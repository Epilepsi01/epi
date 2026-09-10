import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from epi import Catalog, WindowSet
from epi.evaluation import patient_folds
from epi.windows import WindowReadError


class WindowSamplingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(":memory:")
        self.addCleanup(self.catalog.close)
        self.windows = WindowSet(self.catalog, channels=["C3"], rate=4, length_sec=1)
        conn = self.catalog.conn
        for patient in range(12):
            key = "synthetic-%d" % patient
            fold = "train" if patient < 8 else ("val" if patient < 10 else "test")
            conn.execute("INSERT INTO patients (patient_key) VALUES (?)", (key,))
            conn.execute("INSERT INTO splits VALUES (?, ?)", (key, fold))
        # Две записи одного пациента должны оставаться одной группой.
        for recording in range(1, 14):
            patient = recording - 1 if recording <= 12 else 0
            conn.execute(
                "INSERT INTO recordings (id, patient_key, rel_path) VALUES (?, ?, ?)",
                (recording, "synthetic-%d" % patient, "synthetic-%d.e" % recording),
            )
            for offset in range(4):
                conn.execute(
                    """INSERT INTO windows
                       (id, recording_id, segment, start_sec, length_sec,
                        label, overlap_sec, lead_sec)
                       VALUES (?, ?, 0, ?, 1, ?, 0, ?)""",
                    ((recording - 1) * 4 + offset + 1, recording, offset,
                     int(offset < 2), 10 if offset == 2 else 1000),
                )
        conn.commit()

    @staticmethod
    def signal(window):
        return np.full((1, 4), window["id"], dtype=np.float32)

    def test_groups_match_filtered_and_shuffled_windows_and_exclude_holdouts(self):
        def load(window):
            if window["id"] == 1:
                raise IndexError("synthetic damaged block")
            if window["id"] == 2:
                return np.zeros((1, 4), dtype=np.float32)
            return self.signal(window)

        with patch.object(self.windows, "load", side_effect=load):
            with self.assertLogs("epi.windows", level="WARNING") as logs:
                x, y, groups = self.windows.sample(
                    "train", n=100, hard_ratio=0.5, hard_span_sec=20,
                    return_groups=True, verbose=False,
                )
        self.assertEqual(len(y), 34)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(set(groups), {"synthetic-%d" % i for i in range(8)})
        for signal, label, group in zip(x, y, groups):
            window_id = int(signal[0, 0])
            recording = (window_id - 1) // 4 + 1
            patient = recording - 1 if recording <= 12 else 0
            self.assertEqual(group, "synthetic-%d" % patient)
            self.assertEqual(label, int((window_id - 1) % 4 < 2))
        message = "\n".join(logs.output)
        for detail in ("window_id=1", "recording_id=1", "synthetic-1.e",
                       "segment=0", "start_sec=0", "length_sec=1",
                       "label=1", "target_rate=4", "Traceback",
                       "IndexError: synthetic damaged block", "flat=1, unreadable=1"):
            self.assertIn(detail, message)
        self.assertIsNotNone(logs.records[0].exc_info)
        for train, val in patient_folds(groups, 5, y=y):
            self.assertFalse(set(groups[train]) & set(groups[val]))
        # CV не меняет сохранённый train/val/test.
        self.assertEqual(len(self.catalog.query("SELECT * FROM splits")), 12)

    def test_default_two_arrays_remain_compatible(self):
        with patch.object(self.windows, "load", side_effect=self.signal):
            result = self.windows.sample(n=12, verbose=False)
            with_groups = self.windows.sample(n=12, verbose=False, return_groups=True)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], with_groups[0])
        np.testing.assert_array_equal(result[1], with_groups[1])

    def test_known_read_errors_are_skipped_but_unexpected_errors_propagate(self):
        for error in (IndexError, OSError, EOFError, WindowReadError):
            with self.subTest(error=error):
                with patch.object(self.windows, "load", side_effect=error("broken")):
                    with self.assertLogs("epi.windows", level="WARNING"):
                        with self.assertRaisesRegex(ValueError, "ошибок чтения 36"):
                            self.windows.sample(n=100, verbose=False)
        for error in (TypeError, ValueError, MemoryError):
            with self.subTest(error=error):
                with patch.object(self.windows, "load", side_effect=error("unexpected")):
                    with self.assertRaises(error):
                        self.windows.sample(n=4, verbose=False)

    def test_export_contains_groups_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.windows, "load", side_effect=self.signal):
                with contextlib.redirect_stdout(io.StringIO()):
                    path = self.windows.export(
                        "train", out_dir=directory, n=12, verbose=False,
                    )
                x, y, groups = self.windows.sample(
                    n=12, verbose=False, return_groups=True,
                )
            self.assertEqual(path, Path(directory) / "train.npz")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["x"], x)
                np.testing.assert_array_equal(saved["y"], y)
                np.testing.assert_array_equal(saved["groups"], groups)
                self.assertEqual(saved["channels"].tolist(), ["C3"])
                self.assertEqual(saved["rate"], 4)

    def test_load_preserves_backend_exception_as_cause(self):
        reader = Mock(sampling_rate=4)
        window = self.windows.windows("train", limit=1)[0]
        for error in (IndexError, OSError, EOFError, ValueError):
            reader.read.side_effect = error("damaged block")
            with self.subTest(error=error), patch.object(
                self.windows, "_reader", return_value=reader,
            ):
                with self.assertRaises(WindowReadError) as caught:
                    self.windows.load(window)
                self.assertIsInstance(caught.exception.__cause__, error)

    def test_short_empty_and_wrong_channel_windows_are_rejected(self):
        window = self.windows.windows("train", limit=1)[0]
        for shape in ((2, 1), (0, 1), (4, 2), (4,)):
            for rate in (4, 8):
                reader = Mock(sampling_rate=rate)
                reader.read.return_value = np.ones(shape)
                with self.subTest(shape=shape, rate=rate), patch.object(
                    self.windows, "_reader", return_value=reader,
                ):
                    with self.assertRaises(WindowReadError):
                        self.windows.load(window)

    def test_load_resamples_complete_window(self):
        reader = Mock(sampling_rate=8)
        reader.read.return_value = np.arange(8, dtype=float)[:, None]
        window = self.windows.windows("train", limit=1)[0]
        with patch.object(self.windows, "_reader", return_value=reader):
            data = self.windows.load(window)
        self.assertEqual(data.shape, (1, 4))
        self.assertEqual(data.dtype, np.float32)
        reader.select_segment.assert_called_once_with(window["segment"])


if __name__ == "__main__":
    unittest.main()
