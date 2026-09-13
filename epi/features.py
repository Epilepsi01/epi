import numpy as np
from scipy.signal import find_peaks, welch

from .preprocessing import bandpass


BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)

BANDS_70 = BANDS + (("high_gamma", 40.0, 70.0),)


def _eeg(x):
    x = np.asarray(x)
    single = x.ndim == 2
    if single:
        x = x[None]
    if x.ndim != 3:
        raise ValueError("expected (channels, samples) or (windows, channels, samples)")
    return x.astype(np.float64, copy=False), single


def acf(x, rate, lags_sec=None, max_lag_sec=2.0, step_sec=0.02):
    x, single = _eeg(x)
    if lags_sec is None:
        lags_sec = np.arange(step_sec, max_lag_sec + step_sec / 2, step_sec)
    lags_sec = np.asarray(lags_sec, dtype=float)
    indexes = np.rint(lags_sec * rate).astype(int)
    max_lag = int(indexes.max(initial=0))

    z = x - x.mean(axis=-1, keepdims=True)
    z /= x.std(axis=-1, keepdims=True) + 1e-8
    n = z.shape[-1]
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(z, n=nfft, axis=-1)
    values = np.fft.irfft(spectrum * spectrum.conj(), n=nfft, axis=-1)
    values = values[..., :max_lag + 1]
    values /= values[..., :1] + 1e-12
    values = values.mean(axis=1)[..., indexes].astype(np.float32)
    return (values[0] if single else values), lags_sec


def spectral(x, rate, bands=BANDS, nperseg=None):
    x, single = _eeg(x)
    nperseg = nperseg or min(x.shape[-1], int(round(rate * 4)))
    freq, power = welch(x, fs=rate, nperseg=nperseg, axis=-1)
    power = power.mean(axis=1)
    total_mask = (freq >= bands[0][1]) & (freq <= bands[-1][2])
    total = np.trapezoid(power[:, total_mask], freq[total_mask], axis=-1) + 1e-12

    values = []
    names = []
    for name, low, high in bands:
        mask = (freq >= low) & (freq < high)
        band = np.trapezoid(power[:, mask], freq[mask], axis=-1)
        values.append(band / total)
        names.append("power_%s" % name)

    values = np.stack(values, axis=1).astype(np.float32)
    return (values[0] if single else values), names


def entropy(x, rate, low=0.5, high=40.0, nperseg=None):
    x, single = _eeg(x)
    nperseg = nperseg or min(x.shape[-1], int(round(rate * 4)))
    freq, power = welch(x, fs=rate, nperseg=nperseg, axis=-1)
    power = power.mean(axis=1)
    mask = (freq >= low) & (freq <= high)
    power = power[:, mask]
    probability = power / (power.sum(axis=-1, keepdims=True) + 1e-12)
    value = -np.sum(probability * np.log(probability + 1e-12), axis=-1)
    value /= np.log(max(power.shape[-1], 2))
    value = value.astype(np.float32)
    return float(value[0]) if single else value


def eeg_features(x, rate, lags_sec=None, bands=BANDS, entropy_low=0.5,
                 entropy_high=None):
    acf_values, lags = acf(x, rate, lags_sec=lags_sec)
    spectral_values, spectral_names = spectral(x, rate, bands=bands)
    entropy_high = bands[-1][2] if entropy_high is None else entropy_high
    entropy_values = entropy(x, rate, low=entropy_low, high=entropy_high)

    single = np.asarray(x).ndim == 2
    if single:
        values = np.concatenate([acf_values, spectral_values, [entropy_values]])
    else:
        values = np.concatenate([
            acf_values,
            spectral_values,
            np.asarray(entropy_values)[:, None],
        ], axis=1)
    names = ["acf_%.3fs" % lag for lag in lags] + spectral_names + ["spectral_entropy"]
    return values.astype(np.float32), names


def r_peaks(ecg, rate):
    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    filtered = bandpass(ecg[None], rate, 5.0, min(20.0, rate / 2 - 1.0))[0]
    centered = filtered - np.median(filtered)
    scale = np.median(np.abs(centered)) * 1.4826 + 1e-8
    centered = centered / scale
    if abs(np.percentile(centered, 1)) > abs(np.percentile(centered, 99)):
        centered = -centered
    peaks, _ = find_peaks(
        centered,
        distance=max(1, int(round(rate * 0.3))),
        prominence=1.0,
    )
    return peaks


def hrv(ecg, rate):
    peaks = r_peaks(ecg, rate)
    rr = np.diff(peaks) / rate
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    names = ["beats", "mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50"]
    if len(rr) < 2:
        return np.array([len(peaks), np.nan, np.nan, np.nan, np.nan], dtype=np.float32), names

    diff = np.diff(rr)
    values = np.array([
        len(peaks),
        60.0 / rr.mean(),
        rr.std(ddof=1) * 1000.0,
        np.sqrt(np.mean(diff ** 2)) * 1000.0 if len(diff) else np.nan,
        np.mean(np.abs(diff) > 0.05) if len(diff) else np.nan,
    ], dtype=np.float32)
    return values, names
