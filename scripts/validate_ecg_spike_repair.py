"""Validate spike interpolation on real local ECG; never edit source recordings.

Run: python -m scripts.validate_ecg_spike_repair [--audit-windows]
The optional audit uses the saved 400-window manifest, without resampling it.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from epi.preprocessing import (
    ecg_quality_report, ecg_raw_quality_report, preprocess_ecg, repair_ecg_spikes,
)
from epi.reader import NicoletEReader
from .compare_ecg_preprocessing import ecg_quality_report as filter_only_quality_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/reports/ecg_spike_repair"
CASES = [(85, 0, 405, "Одиночный выброс"),
         (210, 0, 4475, "Насыщение АЦП"),
         (31, 1, 1665, "Регулярные высокие зубцы"),
         (44, 1, 100, "Корректно прочитанный сегмент"),
         (202, 0, 3240, "Контроль ЭКГ"),
         (15, 0, 8140, "Контроль ЭКГ")]


def reader_for(recording_id):
    with sqlite3.connect(f"file:{ROOT}/data/catalog.sqlite?mode=ro", uri=True) as conn:
        root = Path(conn.execute("SELECT value FROM meta WHERE key='root'").fetchone()[0])
        row = conn.execute("SELECT rel_path FROM recordings WHERE id=? AND deleted_at IS NULL",
                           (int(recording_id),)).fetchone()
    if row is None:
        raise ValueError(f"recording {recording_id} is absent from this local catalogue")
    return NicoletEReader(root / row[0], type_names={})


def resample(x, rate, target=256):
    ratio = Fraction(target / rate).limit_denominator(10000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype("float32")


def compare_case(reader, segment, start, duration=10):
    reader.select_segment(int(segment))
    native = reader.read_channel("EKG", start, duration)[None]
    original = native.copy()
    rate = reader.sampling_rate
    adc = reader.channel_limits("EKG")
    repaired, info = repair_ecg_spikes(native, rate, adc_limits=adc)
    np.testing.assert_array_equal(native, original)
    assert repaired.shape == native.shape
    np.testing.assert_array_equal(repaired[~info["repaired_mask"]], native[~info["repaired_mask"]])
    assert info["repaired_fraction"] <= .01

    raw, prepared = resample(native, rate), resample(repaired, rate)
    before = preprocess_ecg(raw, 256, repair_spikes=False)
    # Native samples were already repaired. Do not detect another set of
    # candidates after resampling; retain the original mask/provenance.
    after = preprocess_ecg(prepared, 256, repair_spikes=False)
    source_qc = ecg_raw_quality_report(native, rate, adc_limits=adc)
    old_qc = filter_only_quality_report(raw, 256, filtered=before)
    old_source_qc = filter_only_quality_report(native, rate, adc_limits=adc)
    new_qc = ecg_quality_report(prepared, 256, filtered=after, repair_spikes=False)
    keep_before = bool(old_source_qc["keep"] and old_qc["keep"])
    keep_after = bool(source_qc["keep"] and not info["unrepaired_spike"] and new_qc["keep"])
    return dict(native=native, repaired=repaired, raw=raw, before=before, after=after,
                rate=rate, start=start, info=info, source_qc=source_qc,
                old_qc=old_qc, new_qc=new_qc, keep_before=keep_before, keep_after=keep_after)


def case_row(case, recording_id, segment, description=""):
    info = case["info"]
    return dict(recording_id=recording_id, segment=segment, start_sec=case["start"],
                description=description, keep_before=case["keep_before"], keep_after=case["keep_after"],
                repaired_samples=int(info["repaired_mask"].sum()),
                repaired_seconds=float(info["repaired_mask"].sum() / case["rate"]),
                repaired_fraction=info["repaired_fraction"],
                unrepaired_spike=info["unrepaired_spike"],
                peak_before_uv=case["old_qc"]["peak_uv"], peak_after_uv=case["new_qc"]["peak_uv"],
                source_reasons=case["source_qc"]["reasons"], after_reasons=case["new_qc"]["reasons"])


def plot_case(case, title, path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), layout="constrained")
    tn = case["start"] + np.arange(case["native"].shape[-1]) / case["rate"]
    t = case["start"] + np.arange(case["after"].shape[-1]) / 256
    axes[0].plot(tn, case["native"][0], lw=.7, label="Исходная ЭКГ")
    axes[0].plot(tn, case["repaired"][0], lw=.7, label="После замены короткого выброса")
    axes[0].set_title(title + " — до ресэмплинга и частотных фильтров")
    axes[0].legend(loc="upper right")
    axes[1].plot(tn, case["native"][0], lw=.8)
    axes[1].plot(tn, case["repaired"][0], lw=1)
    spans = [r for r in case["info"]["intervals"] if r["repaired"]]
    if spans:
        first = spans[0]
        axes[1].set_xlim(case["start"] + first["start_sec"] - .15,
                         case["start"] + first["stop_sec"] + .15)
    axes[1].set_title("Маска замен отмечена оранжевым: внутри неё истинная ЭКГ неизвестна")
    axes[2].plot(t, case["before"][0], lw=.8, label="Фильтрация с исходным импульсом")
    axes[2].plot(t, case["after"][0], lw=.8, label="Замена импульса → фильтрация")
    axes[2].set_title(f"Текущее QC: до {case['keep_before']}, после {case['keep_after']}")
    axes[2].legend(loc="upper right")
    axes[3].plot(t, (case["before"] - case["after"])[0], lw=.8)
    axes[3].set_title("Разность результатов: влияние исходного выброса распространяется за маску")
    for ax in axes:
        for span in spans:
            ax.axvspan(case["start"] + span["start_sec"], case["start"] + span["stop_sec"],
                       color="orange", alpha=.25)
        ax.set(xlabel="Секунды от начала сегмента", ylabel="мкВ")
        ax.grid(alpha=.2)
    if path:
        fig.savefig(path, dpi=130)
    return fig


def validate_examples():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cases = [], []
    for rid, seg, start, description in CASES:
        reader = reader_for(rid)
        stat_before = reader.path.stat()
        case = compare_case(reader, seg, start)
        stat_after = reader.path.stat()
        assert (stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
        # Also exercise the normal library entry point: default processing and
        # QC agree whether the caller passes an already filtered signal or not.
        filtered, info = preprocess_ecg(case["native"], case["rate"],
                                       adc_limits=reader.channel_limits("EKG"), return_info=True)
        q1 = ecg_quality_report(case["native"], case["rate"], filtered=filtered,
                                adc_limits=reader.channel_limits("EKG"))
        q2 = ecg_quality_report(case["native"], case["rate"],
                                adc_limits=reader.channel_limits("EKG"))
        assert q1["keep"] == q2["keep"]
        assert info["repair_applied"] == case["info"]["repair_applied"]
        rows.append(case_row(case, rid, seg, description))
        cases.append(case)
        print(rows[-1], flush=True)
    assert [r["keep_after"] for r in rows] == [True, False, True, True, True, True]
    assert [r["repaired_samples"] > 0 for r in rows] == [True, False, False, False, False, False]

    spike = cases[0]
    x, rate = spike["native"], spike["rate"]
    # Guard checks use real samples/crops, without generating synthetic signals.
    for options in ({"max_repaired_fraction": 0}, {"max_duration_sec": .001}):
        unchanged, info = repair_ecg_spikes(x, rate, **options)
        np.testing.assert_array_equal(unchanged, x)
        assert info["unrepaired_spike"] and not info["repair_applied"]
        q = ecg_quality_report(x, rate, spike_options=options)
        assert not q["keep"] and "unrepairable_spike" in q["reasons"]
    cropped = x[:, int(np.argmax(np.abs(x[0] - np.median(x)))):]
    unchanged, info = repair_ecg_spikes(cropped, rate)
    np.testing.assert_array_equal(unchanged, cropped)
    assert info["unrepaired_spike"] and not info["repair_applied"]
    assert any(r["reason"] == "window_edge" for r in info["intervals"])

    batch = np.stack([cases[0]["native"], cases[2]["native"]])
    corrected, info = repair_ecg_spikes(batch, rate)
    np.testing.assert_array_equal(corrected[0], spike["repaired"])
    np.testing.assert_array_equal(corrected[1], cases[2]["native"])
    assert info["repair_applied"].tolist() == [True, False]
    batch_filtered = preprocess_ecg(batch, rate)
    batch_qc = ecg_quality_report(batch, rate, filtered=batch_filtered)
    assert batch_qc["keep"].all()
    assert batch_filtered.shape == batch.shape and batch_filtered.dtype == np.float32

    # In a long real window, a short series can exceed the global threshold.
    # Similar nearby peaks must not be interpolated as isolated impulses.
    clustered = compare_case(reader_for(29), 0, 2315, 300)
    lo = int(round((2478 - 2315) * clustered["rate"]))
    hi = int(round((2480 - 2315) * clustered["rate"]))
    assert not clustered["info"]["repaired_mask"][:, lo:hi].any()
    assert any(r["reason"] == "not_isolated" for r in clustered["info"]["intervals"])

    pd.DataFrame(rows).to_csv(OUT / "real_examples.csv", index=False)
    (OUT / "spike_intervals.json").write_text(json.dumps(spike["info"]["intervals"], indent=2))
    import matplotlib.pyplot as plt
    fig = plot_case(spike, "Запись 85, сегмент 0, 405–415 с", OUT / "spike_before_after.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def audit_windows():
    manifest = pd.read_csv(ROOT / "data/reports/preprocessing_v2_20261010/window_manifest.csv")
    rows = []
    for rid, group in manifest.groupby("recording_id", sort=False):
        reader = reader_for(int(rid))
        for w in group.to_dict("records"):
            case = compare_case(reader, w["segment"], w["start_sec"], w["length_sec"])
            row = case_row(case, int(rid), w["segment"])
            row["label"] = w["label"]
            rows.append(row)
        print(f"Checked {len(rows)}/{len(manifest)} windows", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "window_audit.csv", index=False)
    table = frame.groupby("label").agg(n=("label", "size"), before=("keep_before", "sum"),
                                       after=("keep_after", "sum"),
                                       repaired=("repaired_samples", lambda x: (x > 0).sum()))
    table.to_csv(OUT / "retention_by_label.csv")
    print(table.to_string(), flush=True)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-windows", action="store_true")
    args = parser.parse_args()
    validate_examples()
    if args.audit_windows:
        audit_windows()
    print("Real-data spike repair checks passed", flush=True)
