import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch


FATAL = ("flat_signal", "nan")
WARNING = ("high_amplitude", "large_jump", "clipping", "bad_channels")


def _array(x):
    x = np.asarray(x)
    if x.ndim not in (2, 3):
        raise ValueError("expected (channels, samples) or (windows, channels, samples)")
    return x.astype(np.float64, copy=False)


def bandpass(x, rate, low=0.5, high=40.0, order=4):
    x = _array(x)
    sos = butter(order, [low, high], btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def notch(x, rate, freq=50.0, quality=30.0):
    x = _array(x)
    freqs = np.atleast_1d(freq).astype(float)
    out = x
    for value in freqs:
        if value <= 0 or value >= rate / 2:
            continue
        b, a = iirnotch(value, quality, fs=rate)
        out = filtfilt(b, a, out, axis=-1)
    return out.astype(np.float32)


def signal_scale(x, method="robust", eps=1e-8):
    x = _array(x)
    if method == "robust":
        center = np.median(x, axis=-1, keepdims=True)
        scale = np.median(np.abs(x - center), axis=-1, keepdims=True) * 1.4826
    elif method == "zscore":
        center = np.mean(x, axis=-1, keepdims=True)
        scale = np.std(x, axis=-1, keepdims=True)
    else:
        raise ValueError("method must be 'robust' or 'zscore'")
    return center, np.maximum(scale, eps)


def normalize(x, method="robust", eps=1e-8, clip=None):
    x = _array(x)
    center, scale = signal_scale(x, method=method, eps=eps)
    out = (x - center) / scale
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32)


def quality(x, flat_eps=1e-6, bad_low=0.1, bad_high=10.0):
    single = np.asarray(x).ndim == 2
    x = _array(x)
    if single:
        x = x[None]

    finite = np.isfinite(x)
    safe = np.where(finite, x, 0.0)
    center, scale = signal_scale(safe)
    centered = safe - center
    normalized = centered / scale
    channel_scale = scale[..., 0]
    median_scale = np.median(channel_scale, axis=1, keepdims=True)
    median_scale = np.maximum(median_scale, 1e-8)

    flat = np.mean(np.all(np.abs(safe) <= flat_eps, axis=1), axis=1)
    amplitude = np.percentile(np.abs(centered), 99.5, axis=(1, 2))
    jump = np.percentile(np.abs(np.diff(centered, axis=-1)), 99.5, axis=(1, 2))
    amplitude_scale = np.percentile(np.abs(normalized), 99.5, axis=(1, 2))
    jump_scale = np.percentile(np.abs(np.diff(normalized, axis=-1)), 99.5, axis=(1, 2))

    minimum = np.min(safe, axis=-1, keepdims=True)
    maximum = np.max(safe, axis=-1, keepdims=True)
    clipped = np.isclose(safe, minimum, atol=flat_eps, rtol=0) | np.isclose(
        safe, maximum, atol=flat_eps, rtol=0
    )
    clipping = np.max(np.mean(clipped, axis=-1), axis=1)

    ratio = channel_scale / median_scale
    bad_channels = np.mean((ratio < bad_low) | (ratio > bad_high), axis=1)
    nan_fraction = 1.0 - np.mean(finite, axis=(1, 2))

    result = {
        "flat_fraction": flat,
        "amplitude_uv": amplitude,
        "jump_uv": jump,
        "amplitude_scale": amplitude_scale,
        "jump_scale": jump_scale,
        "signal_scale_uv": median_scale[:, 0],
        "clipping_fraction": clipping,
        "bad_channel_fraction": bad_channels,
        "nan_fraction": nan_fraction,
    }
    if single:
        return {name: float(value[0]) for name, value in result.items()}
    return result


def _over(value, limit):
    if limit is None:
        return np.zeros_like(value, dtype=bool)
    return value > limit


def _quality_flags(q, max_flat, max_amplitude_uv, max_jump_uv, max_clipping,
                   max_bad_channels, max_nan, max_amplitude_scale=None,
                   max_jump_scale=None):
    return {
        "flat_signal": _over(q["flat_fraction"], max_flat),
        "high_amplitude": _over(q["amplitude_uv"], max_amplitude_uv)
        | _over(q["amplitude_scale"], max_amplitude_scale),
        "large_jump": _over(q["jump_uv"], max_jump_uv)
        | _over(q["jump_scale"], max_jump_scale),
        "clipping": _over(q["clipping_fraction"], max_clipping),
        "bad_channels": _over(q["bad_channel_fraction"], max_bad_channels),
        "nan": _over(q["nan_fraction"], max_nan),
    }


def quality_flags(x, max_flat=0.01, max_amplitude_uv=1000.0, max_jump_uv=250.0,
                  max_clipping=0.01, max_bad_channels=0.25, max_nan=0.0,
                  max_amplitude_scale=None, max_jump_scale=None):
    q = quality(x)
    return _quality_flags(
        q, max_flat, max_amplitude_uv, max_jump_uv, max_clipping,
        max_bad_channels, max_nan, max_amplitude_scale, max_jump_scale
    )


def quality_report(x, max_flat=0.01, max_amplitude_uv=1000.0, max_jump_uv=250.0,
                   max_clipping=0.01, max_bad_channels=0.25, max_nan=0.0,
                   max_amplitude_scale=None, max_jump_scale=None,
                   reject_warnings=()):
    """QC до нормализации; keep учитывает явно выбранные reject_warnings.

    По умолчанию сохранено прежнее поведение: предупреждения не отбраковывают
    окна. Для обучения перечислите нужные причины из WARNING. fatal и warning
    описывают тяжесть замечаний независимо от политики отбора.
    """
    reject_warnings = tuple(reject_warnings)
    if set(reject_warnings) - set(WARNING):
        raise ValueError("unknown warning in reject_warnings")
    single = np.asarray(x).ndim == 2
    q = quality(x)
    flags = _quality_flags(
        q, max_flat, max_amplitude_uv, max_jump_uv, max_clipping,
        max_bad_channels, max_nan, max_amplitude_scale, max_jump_scale
    )

    if single:
        fatal = any(bool(flags[name]) for name in FATAL)
        warning = any(bool(flags[name]) for name in WARNING)
        reasons = [name for name, bad in flags.items() if bad]
        return {
            **q,
            "fatal": fatal,
            "warning": warning,
            "keep": not (fatal or any(bool(flags[name]) for name in reject_warnings)),
            "reasons": reasons,
        }

    n = len(next(iter(flags.values())))
    fatal = np.zeros(n, dtype=bool)
    warning = np.zeros(n, dtype=bool)

    for name in FATAL:
        fatal |= flags[name]
    for name in WARNING:
        warning |= flags[name]
    rejected = fatal.copy()
    for name in reject_warnings:
        rejected |= flags[name]

    reasons = np.array([
        ";".join(name for name, bad in flags.items() if bad[i]) or "ok"
        for i in range(n)
    ], dtype=object)

    return {
        **q,
        "fatal": fatal,
        "warning": warning,
        "keep": ~rejected,
        "reasons": reasons,
    }


def fit_quality_limits(x, quantile=0.995, max_flat=0.01, max_clipping=0.01,
                       max_bad_channels=0.25, max_nan=0.0):
    q = quality(x)
    valid = ((q["flat_fraction"] <= max_flat) & (q["nan_fraction"] <= max_nan)
             & (q["clipping_fraction"] <= max_clipping)
             & (q["bad_channel_fraction"] <= max_bad_channels))
    if not np.any(valid):
        raise ValueError("no valid windows for quality limits")
    return {
        "max_flat": max_flat,
        "max_amplitude_uv": None,
        "max_jump_uv": None,
        "max_clipping": max_clipping,
        "max_bad_channels": max_bad_channels,
        "max_nan": max_nan,
        "max_amplitude_scale": float(np.quantile(q["amplitude_scale"][valid], quantile)),
        "max_jump_scale": float(np.quantile(q["jump_scale"][valid], quantile)),
    }


def clean_mask(x, max_flat=0.01, max_amplitude_uv=1000.0, max_jump_uv=250.0,
               max_clipping=0.01, max_bad_channels=0.25, max_nan=0.0,
               strict=False, max_amplitude_scale=None, max_jump_scale=None):
    flags = quality_flags(
        x,
        max_flat=max_flat,
        max_amplitude_uv=max_amplitude_uv,
        max_jump_uv=max_jump_uv,
        max_clipping=max_clipping,
        max_bad_channels=max_bad_channels,
        max_nan=max_nan,
        max_amplitude_scale=max_amplitude_scale,
        max_jump_scale=max_jump_scale,
    )
    single = np.asarray(x).ndim == 2

    if single:
        fatal = any(bool(flags[name]) for name in FATAL)
        warning = any(bool(flags[name]) for name in WARNING)
        return not (fatal or (strict and warning))

    n = len(next(iter(flags.values())))
    fatal = np.zeros(n, dtype=bool)
    warning = np.zeros(n, dtype=bool)

    for name in FATAL:
        fatal |= flags[name]
    for name in WARNING:
        warning |= flags[name]

    return ~(fatal | warning) if strict else ~fatal


def preprocess(x, rate, low=0.5, high=40.0, notch_freq=50.0, normalize_signal=False,
               normalize_method="robust", clip=None):
    x = notch(x, rate, notch_freq)
    x = bandpass(x, rate, low, high)
    if normalize_signal:
        x = normalize(x, method=normalize_method, clip=clip)
    return x


def preprocess_ecg(x, rate, low=0.5, high=40.0, notch_freq=50.0,
                   normalize_signal=False, normalize_method="robust", clip=None,
                   notch_width=4.0, pad_sec=2.0):
    """ЭКГ: полосовое подавление сети, затем 0.5–40 Гц; результат float32.

    В отличие от узкого notch в preprocess(), по умолчанию подавляется полоса
    48–52 Гц: частота помехи может отклоняться от 50 Гц. Отражение краёв снижает
    переходные процессы; никакие отсчёты не удаляются. Фильтры двунаправленные,
    поэтому это офлайн-обработка целого окна, не потоковый causal-фильтр.
    QC делайте по исходному сигналу И результату до нормализации: фильтрация
    не восстанавливает клиппинг или пропуски. notch_freq=None отключает сеть.
    """
    x = _array(x)
    if not np.isfinite(rate) or not 0 < low < high < rate / 2:
        raise ValueError("expected 0 < low < high < rate / 2")
    if not np.isfinite(pad_sec) or pad_sec < 0:
        raise ValueError("pad_sec must be nonnegative and finite")
    if not np.isfinite(notch_width) or notch_width <= 0:
        raise ValueError("notch_width must be positive and finite")
    padding = min(int(round(pad_sec * rate)), x.shape[-1] - 1)
    if padding < 0:
        raise ValueError("empty signal")
    pads = [(0, 0)] * x.ndim
    pads[-1] = (padding, padding)
    out = np.pad(x, pads, mode="reflect") if padding else x
    if notch_freq is not None:
        for freq in np.atleast_1d(notch_freq):
            edges = [freq - notch_width / 2, freq + notch_width / 2]
            if not 0 < edges[0] < edges[1] < rate / 2:
                raise ValueError("notch band must lie strictly within Nyquist")
            out = sosfiltfilt(butter(4, edges, btype="bandstop", fs=rate,
                                    output="sos"), out, axis=-1)
    out = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=rate,
                            output="sos"), out, axis=-1)
    if padding:
        out = out[..., padding:-padding]
    if normalize_signal:
        out = normalize(out, method=normalize_method, clip=clip)
    return out.astype(np.float32)


def line_noise_power_ratio(x, rate, line_freq=50.0, half_width=2.0,
                           low=0.5, high=40.0, trim_sec=2.0):
    """P(48..52 Гц) / P(0.5..40 Гц), Welch, 4 с, максимум по каналам.

    Это явная новая метрика, НЕ неизвестная реализация line_noise_ratio из
    epi 1.3.0. Отношение мощностей, без единиц. NaN при нулевой мощности
    знаменателя; отсутствие сигнала не считается отсутствием помехи.
    Края длиной trim_sec исключаются только из оценки спектра, не из данных.
    """
    single = np.asarray(x).ndim == 2
    x = _array(x)
    if not (0 < low < high < rate / 2
            and 0 < line_freq - half_width < line_freq + half_width < rate / 2):
        raise ValueError("spectral bands must lie strictly within Nyquist")
    if not np.isfinite(trim_sec) or trim_sec < 0:
        raise ValueError("trim_sec must be nonnegative and finite")
    trim = int(round(trim_sec * rate))
    if x.shape[-1] - 2 * trim < 2 * rate:
        raise ValueError("need at least 2 seconds after spectral edge trimming")
    if trim:
        x = x[..., trim:-trim]
    freq, power = welch(x, fs=rate, nperseg=min(int(round(4 * rate)), x.shape[-1]),
                        axis=-1)
    numerator = power[..., np.abs(freq - line_freq) <= half_width].sum(axis=-1)
    denominator = power[..., (freq >= low) & (freq <= high)].sum(axis=-1)
    ratio = np.full_like(denominator, np.nan)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    result = np.max(ratio, axis=-1)
    return float(result) if single else result


def _longest_run(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def _ecg_flags(report, flags):
    rejected = ~report["keep"]
    reasons = []
    for i, previous in enumerate(report["reasons"]):
        current = [] if previous == "ok" else previous.split(";")
        current += [name for name, flag in flags.items() if flag[i]]
        reasons.append(";".join(current) or "ok")
    for flag in flags.values():
        rejected |= flag
        report["warning"] |= flag
    report.update(keep=~rejected, reasons=np.array(reasons, dtype=object))
    return report


def _single_report(report):
    return {name: (value[0].item() if isinstance(value[0], np.generic) else value[0])
            for name, value in report.items()}


def ecg_raw_quality_report(raw, rate, adc_limits=None, max_constant_sec=0.5, **limits):
    """QC физического исходного сигнала до ресэмплинга; не запускает фильтры.

    Результат нужно объединить по keep с ecg_quality_report обработанного
    окна: границы АЦП после ресэмплинга могут перестать быть плоскими.
    """
    single = np.asarray(raw).ndim == 2
    raw = _array(raw)
    if single:
        raw = raw[None]
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    options = dict(max_amplitude_uv=None, max_jump_uv=None)
    options.update(limits)
    options["reject_warnings"] = WARNING
    report = quality_report(raw, **options)
    safe = np.where(np.isfinite(raw), raw, 0.0)
    constant = np.array([
        max((_longest_run(np.abs(np.diff(ch)) <= 1e-6) + 1) / rate for ch in window)
        for window in safe
    ])
    adc_fraction = np.full(len(raw), np.nan)
    if adc_limits is not None:
        bounds = np.broadcast_to(np.asarray(adc_limits, dtype=float), (raw.shape[1], 2))
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("adc_limits must have finite increasing bounds")
        rail = ((safe <= bounds[None, :, 0, None] + 1e-6)
                | (safe >= bounds[None, :, 1, None] - 1e-6))
        adc_fraction = rail.mean(axis=-1).max(axis=1)
    report.update(constant_sec=constant, adc_rail_fraction=adc_fraction)
    report = _ecg_flags(report, {
        "constant_run": constant > max_constant_sec,
        "adc_clipping": _over(adc_fraction, options.get("max_clipping", 0.01)),
    })
    return _single_report(report) if single else report


def ecg_quality_report(raw, rate, filtered=None, adc_limits=None,
                       max_constant_sec=0.5, max_peak_ratio=8.0,
                       min_peak_uv=1000.0, max_line_noise=0.01, **limits):
    """Консервативный QC ЭКГ до нормализации, без изменения входных отсчётов.

    raw — физические мкВ до фильтрации (лучше до ресэмплинга).
    adc_limits — (нижний, верхний) или (каналы, 2) из reader.channel_limits().
    filtered — те же окна, в мкВ, с той же частотой и формой; None вызывает
    preprocess_ecg(). Пропуски/клиппинг проверяются по raw, остаточная сеть и
    одиночные выбросы — по filtered. Окно исключается целиком, без интерполяции.

    Выброс: пик > min_peak_uv И > max_peak_ratio медианы секундных пиков.
    Регулярные высокие R-зубцы сами по себе не являются причиной отбраковки.
    Пороги — инженерные эвристики для проверки на своих данных, не клиническая
    разметка качества. Обучаемые limits подбираются только на train-фоне.
    """
    single = np.asarray(raw).ndim == 2
    raw = _array(raw)
    if filtered is None:
        filtered = preprocess_ecg(raw, rate)
    filtered = _array(filtered)
    if filtered.shape != raw.shape:
        raise ValueError("raw and filtered must have the same shape")
    if single:
        raw, filtered = raw[None], filtered[None]
    report = ecg_raw_quality_report(raw, rate, adc_limits=adc_limits,
                                    max_constant_sec=max_constant_sec, **limits)
    clean = np.where(np.isfinite(filtered), filtered, 0.0)
    centered = np.abs(clean - np.median(clean, axis=-1, keepdims=True))
    block = int(round(rate))
    n_blocks = centered.shape[-1] // block
    if n_blocks < 2:
        raise ValueError("ECG QC needs at least 2 seconds")
    typical = np.median(centered[..., :n_blocks * block].reshape(
        *centered.shape[:-1], n_blocks, block).max(axis=-1), axis=-1)
    peaks = centered.max(axis=-1)
    peak_ratio = peaks / np.maximum(typical, 1e-8)
    # Короткие сигналы тоже допускаются, но спектр должен содержать >=2 с.
    trim = min(2.0, max(0.0, (raw.shape[-1] / rate - 2.0) / 2))
    residual = line_noise_power_ratio(filtered, rate, trim_sec=trim)
    flags = {
        "isolated_spike": np.any((peaks > min_peak_uv) & (peak_ratio > max_peak_ratio), axis=1),
        "residual_line_noise": residual > max_line_noise,
        "nonfinite_filtered": ~np.all(np.isfinite(filtered), axis=(1, 2)),
    }
    report.update(peak_uv=peaks.max(axis=1), peak_ratio=peak_ratio.max(axis=1),
                  residual_line_noise=residual)
    report = _ecg_flags(report, flags)
    return _single_report(report) if single else report
