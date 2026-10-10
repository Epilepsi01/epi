"""Additional checks using only the local real recordings; no synthetic signals."""
import hashlib
import json
import sqlite3
import warnings

import nbformat
import numpy as np
import pandas as pd
from neo.rawio import NicoletRawIO

from epi import Catalog, WindowSet
from epi.catalog import quick_hash
from epi.reader import NicoletEReader
from epi.preprocessing import (
    preprocess_ecg, ecg_quality_report, ecg_raw_quality_report, normalize,
    fit_quality_limits, quality_report, WARNING,
)
from .compare_ecg_preprocessing import ROOT, OUT, AUDIT, review_examples


def main():
    inventory = json.loads((AUDIT / "inventory.json").read_text())
    from pathlib import Path
    root = Path(inventory["root"])
    checks = []
    # Compare multiple channels in late segments, and verify unchanged sources.
    for rid in [14, 46, 99, 30, 31, 44]:
        record = next(r for r in inventory["records"] if r["id"] == rid)
        path = root / record["rel_path"]
        assert path.stat().st_size == record["file_size"]
        assert path.stat().st_mtime == record["file_mtime"]
        assert quick_hash(path) == record["content_hash"]
        reader = NicoletEReader(path, type_names={})
        with warnings.catch_warnings(record=True):
            backend = NicoletRawIO(filename=str(path))
            backend.parse_header()
            for seg in sorted(set([0, len(reader.segments) - 1])):
                reader.select_segment(seg)
                backend.set_signal_channels(seg)
                n = min(reader.n_samples, int(reader.sampling_rate * 2))
                start = max(0, reader.n_samples - n)
                channels = ["EKG", "C3"]
                actual = reader.read(channels, start / reader.sampling_rate, n / reader.sampling_rate)
                assert len(reader.times(start / reader.sampling_rate, n / reader.sampling_rate)) == n
                for column, name in enumerate(channels):
                    chs = backend.header["signal_channels"]
                    ch = chs[chs["name"] == name][0]
                    stream = int(np.flatnonzero(backend.header["signal_streams"]["id"] == ch["stream_id"])[0])
                    stream_channels = chs[chs["stream_id"] == ch["stream_id"]]
                    idx = int(np.flatnonzero(stream_channels["name"] == name)[0])
                    raw = backend.get_analogsignal_chunk(0, seg, start, start + n,
                                                       stream_index=stream, channel_indexes=[idx])
                    expected = backend.rescale_signal_raw_to_float(raw, dtype="float64",
                        stream_index=stream, channel_indexes=[idx])[:, 0]
                    expected *= {"V": 1e6, "mV": 1000}.get(str(ch["units"]), 1)
                    error = float(np.max(np.abs(actual[:, column] - expected)))
                    assert error < 1e-6, (rid, seg, name, error)
                    checks.append(dict(recording_id=rid, segment=seg, channel=name, max_error_uv=error))
    pd.DataFrame(checks).to_csv(OUT / "multichannel_reader_validation.csv", index=False)
    review_examples(inventory["records"], root)

    # Execute the modified preprocessing/QC notebook cells on a small REAL subset.
    # The rest of that modelling notebook depends on unavailable epi 1.3 APIs.
    manifest = pd.read_csv(OUT / "window_manifest.csv")
    train_meta = pd.concat([manifest[manifest.recording_id == rid].head(2)
                            for rid in [31, 44, 46, 14]]).to_dict("records")
    test_meta = pd.concat([manifest[manifest.recording_id == rid].head(2)
                           for rid in [202, 137]]).to_dict("records")
    with Catalog(":memory:") as catalog:
        with sqlite3.connect(f"file:{ROOT}/data/catalog.sqlite?mode=ro", uri=True) as source:
            source.backup(catalog.conn)
        windows = WindowSet(catalog, channels=["EKG"], length_sec=300, rate=256)
        ns = dict(np=np, windows=windows, TARGET_RATE=256, BATCH=2, LOW=.5, HIGH=40,
                  NOTCH=50, NORMALIZE_METHOD="robust", NORMALIZE_CLIP=None, ECG_SPEC=("EKG", "ECG"),
                  QC_FIXED=dict(max_flat=.01, max_clipping=.01, max_bad_channels=.25, max_nan=0),
                  QC_QUANTILE=.995, preprocess_ecg=preprocess_ecg, ecg_quality_report=ecg_quality_report,
                  ecg_raw_quality_report=ecg_raw_quality_report, normalize=normalize,
                  quality_report=quality_report, fit_quality_limits=fit_quality_limits, WARNING=WARNING,
                  m_dev=train_meta, m_test=test_meta)
        ns["x_dev"] = np.stack([windows.load(row) for row in train_meta])
        ns["x_test"] = np.stack([windows.load(row) for row in test_meta])
        ns["y_dev"] = np.array([row["label"] for row in train_meta])
        digest = hashlib.sha256(ns["x_dev"].tobytes()).hexdigest()
        nb = nbformat.read(ROOT / "notebooks/epi_ecg_forecasting_v2_multiscale.ipynb", as_version=4)
        preprocess_cell = next(c.source for c in nb.cells if c.cell_type == "code"
                               and c.source.startswith("def preprocess_batches"))
        qc_cell = next(c.source for c in nb.cells if c.cell_type == "code"
                      and c.source.startswith("def _base_qc"))
        exec(compile(preprocess_cell, "notebook_preprocessing", "exec"), ns)
        exec(compile(qc_cell.split("def safe_auc")[0], "notebook_qc", "exec"), ns)
        indices = np.arange(len(train_meta))
        limits = ns["fit_qc_limits"](ns["x_dev"], ns["y_dev"], indices)
        kept = ns["keep_with_limits"](ns["x_dev"], indices, limits)
        assert len(kept) > 0
        assert ns["dev_qc"]["keep"][kept].all()
        assert hashlib.sha256(ns["x_dev"].tobytes()).hexdigest() == digest
        assert ns["x_dev_filtered"].shape == ns["x_dev_robust"].shape == ns["x_dev"].shape
        assert np.isfinite(ns["x_dev_filtered"]).all() and np.isfinite(ns["x_dev_robust"]).all()
        # Singletons and batches must expose reasons/keep usable by the model cell.
        assert len(ns["dev_qc"]["reasons"]) == len(ns["y_dev"])
        info = dict(multichannel_checks=len(checks), notebook_dev_windows=len(train_meta),
                    notebook_test_windows=len(test_meta), notebook_train_kept=len(kept),
                    source_files_unchanged=True, real_examples_passed=6)
        (OUT / "additional_validation.json").write_text(json.dumps(info, indent=2))
        print("VALIDATED", info, flush=True)


if __name__ == "__main__":
    main()
