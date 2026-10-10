"""Reproducible real-data QC comparison; source files/catalogue are read-only.

Run from the project root: python -m scripts.compare_ecg_preprocessing
The audit inventory and cached Neo snippets come from the preceding local audit.
No patient names or source paths are exported. No model is trained.
"""
from pathlib import Path
import importlib.util
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from fractions import Fraction

from epi import Catalog, WindowSet
from epi.reader import NicoletEReader
from epi.preprocessing import (
    preprocess, preprocess_ecg, quality_report, ecg_quality_report,
    line_noise_power_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/reports/no_valid_20261010"
OUT = ROOT / "data/reports/preprocessing_v2_20261010"


def resample(x, source_rate):
    ratio = Fraction(256 / source_rate).limit_denominator(10000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype("float32")


def load_baseline(name):
    spec = importlib.util.spec_from_file_location("baseline_" + name, OUT / f"baseline_{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_reader(records, root):
    rows = []
    comparisons = pd.read_csv(AUDIT / "reader_comparison.csv")
    for record in records:
        reader = NicoletEReader(root / record["rel_path"], type_names={})
        for row in comparisons[comparisons.recording_id == record["id"]].to_dict("records"):
            reader.select_segment(int(row["segment"]))
            start = int(round(row["start_sec"] * reader.sampling_rate))
            cache = AUDIT / f"recording_{record['id']:03d}" / f"compare_segment_{int(row['segment']):02d}_{start}.npz"
            if not cache.exists():
                continue
            with np.load(cache) as saved:
                expected = saved["neo"]
                actual = reader.read_channel("EKG", row["start_sec"], len(expected) / reader.sampling_rate)
                error = float(np.max(np.abs(actual - expected)))
                rows.append(dict(recording_id=record["id"], segment=int(row["segment"]),
                                 start_sec=row["start_sec"], samples=len(actual),
                                 before_max_error_uv=row["max_abs_difference_uv"],
                                 after_max_error_uv=error))
                if error > 1e-6:
                    raise AssertionError(f"reader mismatch: recording {record['id']}, segment {row['segment']}")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "reader_validation.csv", index=False)
    print("Neo comparisons:", len(result), "before mismatches:",
          int((result.before_max_error_uv > 1e-6).sum()), "after:",
          int((result.after_max_error_uv > 1e-6).sum()), flush=True)


def compare_windows(records, root):
    before_reader = load_baseline("reader").NicoletEReader
    before_qc = load_baseline("preprocessing").quality_report
    manifest_path = OUT / "window_manifest.csv"
    with Catalog(":memory:") as catalog:
        with sqlite3.connect(f"file:{ROOT}/data/catalog.sqlite?mode=ro", uri=True) as source:
            source.backup(catalog.conn)
        windows = WindowSet(catalog, length_sec=300, stride_sec=5, channels=["EKG"],
                            min_overlap_sec=5, gap_before_sec=300, preictal_sec=600,
                            gap_after_sec=300, merge_gap_sec=120, rate=256)
        if manifest_path.exists():
            selected = pd.read_csv(manifest_path).to_dict("records")
        else:
            windows.build(verbose=False)
            selected = []
            # Fixed, evenly spaced selection of up to 20 labelled windows per
            # recording; no selection by quality. Same intervals for all stages.
            for record in records:
                candidates = catalog.query(
                    "SELECT * FROM windows WHERE recording_id=? AND label>=0 "
                    "ORDER BY segment,start_sec", (record["id"],))
                idx = np.unique(np.linspace(0, len(candidates) - 1,
                                           min(20, len(candidates)), dtype=int))
                for i in idx:
                    selected.append(dict(candidates[i]))
            pd.DataFrame(selected).to_csv(manifest_path, index=False)
        rows = []
        for record in records:
            reader = windows._reader(record["id"])
            legacy = before_reader(root / record["rel_path"], type_names={})
            for window in selected:
                if int(window["recording_id"]) != record["id"]:
                    continue
                reader.select_segment(int(window["segment"]))
                native = reader.read_channel("EKG", window["start_sec"], window["length_sec"])[None]
                loaded = windows.load(window)
                old_filter = preprocess(loaded, 256)
                new_filter = preprocess_ecg(loaded, 256)
                source_qc = ecg_quality_report(native, reader.sampling_rate,
                    adc_limits=reader.channel_limits("EKG"))
                output_qc = ecg_quality_report(loaded, 256, filtered=new_filter)
                # Native ADC clipping is checked before resampling can hide it.
                reasons = sorted(set((source_qc["reasons"] + ";" + output_qc["reasons"]).split(";")) - {"ok"})
                row = dict(patient=record["prefix"], recording_id=record["id"],
                    segment=int(window["segment"]), start_sec=window["start_sec"],
                    length_sec=window["length_sec"], label=window["label"],
                    raw_power_ratio=line_noise_power_ratio(loaded, 256),
                    old_filter_power_ratio=line_noise_power_ratio(old_filter, 256),
                    new_filter_power_ratio=line_noise_power_ratio(new_filter, 256),
                    keep_before_on_correct_read=before_qc(loaded)["keep"],
                    keep_after=source_qc["keep"] and output_qc["keep"],
                    reasons=";".join(reasons) or "ok",
                    native_adc_rail_fraction=source_qc["adc_rail_fraction"],
                    peak_ratio=output_qc["peak_ratio"], peak_uv=output_qc["peak_uv"],
                    constant_sec=source_qc["constant_sec"])
                try:
                    legacy.select_segment(int(window["segment"]))
                    old_raw = resample(legacy.read_channel("EKG", window["start_sec"], window["length_sec"])[None],
                                       legacy.sampling_rate)
                    row.update(legacy_read_ok=True, legacy_keep=before_qc(old_raw)["keep"],
                               legacy_clean_power_ratio=line_noise_power_ratio(preprocess(old_raw, 256), 256),
                               reader_max_diff_uv=float(np.max(np.abs(old_raw - loaded))))
                except (ValueError, IndexError, EOFError, OSError) as exc:
                    row.update(legacy_read_ok=False, legacy_keep=False,
                               legacy_error=type(exc).__name__)
                rows.append(row)
            print("compared recording", record["id"], "windows", sum(r["recording_id"] == record["id"] for r in rows), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "window_comparison.csv", index=False)
    table = frame.groupby(["patient", "recording_id"]).agg(
        n=("label", "size"), n_positive=("label", "sum"),
        raw_median=("raw_power_ratio", "median"),
        raw_p95=("raw_power_ratio", lambda a: np.quantile(a, .95)),
        clean_before=("old_filter_power_ratio", "median"),
        clean_after=("new_filter_power_ratio", "median"),
        n_keep_before=("keep_before_on_correct_read", "sum"),
        n_keep_after=("keep_after", "sum"),
        legacy_read_errors=("legacy_read_ok", lambda a: int((~a).sum())),
    )
    table["noise_reduction_factor"] = table.clean_before / table.clean_after
    accepted = frame[frame.keep_after].groupby(["patient", "recording_id"]).agg(
        clean_after_kept=("new_filter_power_ratio", "median"),
        positive_kept=("label", "sum"))
    table = table.join(accepted).sort_values("raw_median", ascending=False)
    table.to_csv(OUT / "comparison_by_patient.csv")
    print(table.to_string(float_format=lambda v: f"{v:.5g}"), flush=True)
    print("TOTAL", len(frame), "before", int(frame.keep_before_on_correct_read.sum()),
          "after", int(frame.keep_after.sum()), flush=True)
    print("REASONS", frame[~frame.keep_after].reasons.str.split(";").explode().value_counts().to_dict(), flush=True)
    return frame, table


def review_examples(records, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_id = {record["id"]: record for record in records}
    # These intervals were visually reviewed before choosing the QC changes.
    examples = [(85, 0, 405, "isolated transient"), (210, 0, 4475, "ADC saturation"),
                (31, 1, 1665, "regular heart beats"), (44, 1, 100, "recovered segment"),
                (202, 0, 3240, "regular heart beats"), (15, 0, 8140, "regular heart beats")]
    fig, axes = plt.subplots(len(examples), 2, figsize=(15, 3 * len(examples)), layout="constrained")
    metrics = []
    for axs, (rid, seg, start, description) in zip(axes, examples):
        reader = NicoletEReader(root / by_id[rid]["rel_path"], segment=seg, type_names={})
        native = reader.read_channel("EKG", start, 10)[None]
        raw = resample(native, reader.sampling_rate)
        new = preprocess_ecg(raw, 256)
        old = preprocess(raw, 256)
        source = ecg_quality_report(native, reader.sampling_rate, adc_limits=reader.channel_limits("EKG"))
        report = ecg_quality_report(raw, 256, filtered=new)
        keep = source["keep"] and report["keep"]
        reasons = ";".join(sorted(set((source["reasons"] + ";" + report["reasons"]).split(";")) - {"ok"})) or "ok"
        for ax, x, name in zip(axs, [old, new], ["Previous filter", "New filter"]):
            ax.plot(start + np.arange(x.shape[-1]) / 256, x[0], lw=.7)
            ax.set(title=f"{rid} / segment {seg} / {description}\n{name}; new QC: {reasons}",
                   xlabel="Seconds in segment", ylabel="uV")
            ax.grid(alpha=.2)
        metrics.append(dict(recording_id=rid, segment=seg, start_sec=start,
                            description=description, keep=keep, reasons=reasons,
                            peak_ratio=report["peak_ratio"], peak_uv=report["peak_uv"],
                            native_adc_rail_fraction=source["adc_rail_fraction"]))
    fig.savefig(OUT / "signal_examples.png", dpi=125)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(OUT / "reviewed_examples.csv", index=False)
    print("REVIEWED EXAMPLES", json.dumps(metrics, ensure_ascii=False), flush=True)
    assert [row["keep"] for row in metrics] == [False, False, True, True, True, True], \
        "QC regression on previously reviewed real ECG examples"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((AUDIT / "inventory.json").read_text())
    records, root = inventory["records"], Path(inventory["root"])
    validate_reader(records, root)
    compare_windows(records, root)
    review_examples(records, root)


if __name__ == "__main__":
    main()
