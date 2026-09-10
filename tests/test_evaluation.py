import unittest
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from epi.evaluation import (
    binary_metrics, patient_cv, patient_folds, summarize_folds,
)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fitted_rows = []

    def fit(self, x, y):
        if hasattr(self, "classes_"):
            raise AssertionError("estimator was reused between folds")
        self.classes_ = np.unique(y)
        self.fitted_rows.append(x[:, 0].copy())
        return self

    def predict_proba(self, x):
        score = x[:, 1]
        return np.column_stack([1 - score, score])

    def predict(self, x):
        return (x[:, 1] >= 0.5).astype(int)


class PatientCVTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(["patient-%d" % i for i in range(10)], 8)
        self.y = np.tile([0, 1], 40)
        self.x = np.column_stack([
            np.arange(80), 0.1 + (np.arange(80) % 7) / 8,
        ])
        RecordingClassifier.fitted_rows = []

    def test_every_patient_is_validated_once_without_window_leakage(self):
        for stratified in (False, True):
            with self.subTest(stratified=stratified):
                splits = list(patient_folds(
                    self.groups, 5, y=self.y if stratified else None, seed=42,
                ))
                seen_patients = []
                seen_windows = []
                self.assertEqual(len(splits), 5)
                for train, val in splits:
                    train_groups = set(self.groups[train])
                    val_groups = set(self.groups[val])
                    self.assertFalse(train_groups & val_groups)
                    self.assertEqual(train_groups | val_groups, set(self.groups))
                    self.assertEqual(len(train) + len(val), len(self.groups))
                    self.assertEqual(set(self.y[train]), {0, 1})
                    self.assertEqual(set(self.y[val]), {0, 1})
                    seen_patients.extend(val_groups)
                    seen_windows.extend(val)
                self.assertCountEqual(seen_patients, np.unique(self.groups))
                self.assertCountEqual(seen_windows, range(len(self.groups)))

    def test_unequal_recording_lengths_still_keep_patients_together(self):
        groups = np.repeat(["a", "b", "a", "c", "d", "e"], [6, 4, 8, 12, 2, 10])
        y = np.arange(len(groups)) % 2
        splits = list(patient_folds(groups, 5, y=y))
        for train, val in splits:
            self.assertFalse(set(groups[train]) & set(groups[val]))
        self.assertCountEqual(np.concatenate([v for _, v in splits]), range(len(y)))

    def test_stratified_seed_is_reproducible(self):
        first = list(patient_folds(self.groups, 5, y=self.y, seed=42))
        second = list(patient_folds(self.groups, 5, y=self.y, seed=42))
        other = list(patient_folds(self.groups, 5, y=self.y, seed=7))
        for (train, val), (train_again, val_again) in zip(first, second):
            np.testing.assert_array_equal(train, train_again)
            np.testing.assert_array_equal(val, val_again)
        self.assertTrue(any(not np.array_equal(a[1], b[1])
                            for a, b in zip(first, other)))

    def test_fewer_patients_reduces_folds_with_warning(self):
        with self.assertWarnsRegex(UserWarning, "only 3 patients"):
            splits = list(patient_folds(np.repeat(["a", "b", "c"], 4), 5))
        self.assertEqual(len(splits), 3)
        for _, val in splits:
            self.assertEqual(len(val), 4)

    def test_invalid_groups_and_fold_counts_are_rejected(self):
        for count in (0, 1, 2.5, True, "5"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                list(patient_folds(self.groups, count))
        for groups in ([], ["a"], [["a", "b"]], [None, "a"],
                       ["", "a"], [np.nan, 1], ["a", np.nan]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                list(patient_folds(groups, 2))
        with self.assertRaisesRegex(ValueError, "same length"):
            list(patient_folds(self.groups, y=[0, 1]))

    def test_clones_train_only_and_oof_scores_keep_input_order(self):
        model = RecordingClassifier()
        model.classes_ = np.array([0, 1])  # clone must discard fitted state
        result = patient_cv(model, self.x, self.y, self.groups, seed=42)
        self.assertEqual(len(RecordingClassifier.fitted_rows), 5)
        self.assertEqual(result["n_splits"], 5)
        for row, (train, val), fitted in zip(
            result["folds"], result["indices"], RecordingClassifier.fitted_rows,
        ):
            np.testing.assert_array_equal(fitted, self.x[train, 0])
            self.assertFalse(set(fitted) & set(self.x[val, 0]))
            self.assertEqual(row["val"], len(val))
            self.assertEqual(row["train_positive"], int(self.y[train].sum()))
            self.assertEqual(row["val_positive"], int(self.y[val].sum()))
            self.assertEqual(row["train_patients"] + row["val_patients"], 10)
        np.testing.assert_allclose(result["scores"], self.x[:, 1])
        np.testing.assert_array_equal(result["predictions"], self.x[:, 1] >= 0.5)
        aucs = [row["roc_auc"] for row in result["folds"]]
        self.assertAlmostEqual(result["summary"]["roc_auc"]["mean"], np.mean(aucs))
        self.assertAlmostEqual(result["summary"]["roc_auc"]["std"], np.std(aucs))
        self.assertEqual(result["summary"]["roc_auc"]["n_valid"], 5)
        expected = binary_metrics(self.y, self.x[:, 1])
        self.assertEqual(result["overall"], expected)

    def test_pipeline_runs_and_original_scaler_stays_unfitted(self):
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
        result = patient_cv(pipeline, self.x, self.y, self.groups)
        self.assertTrue(np.isfinite(result["scores"]).all())
        self.assertFalse(hasattr(pipeline[0], "mean_"))
        self.assertFalse(hasattr(pipeline[1], "classes_"))

    def test_single_class_validation_reports_nan_and_valid_fold_count(self):
        groups = np.repeat(["a", "b", "c", "d", "e"], 4)
        y = np.repeat([1, 1, 0, 0, 0], 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = patient_cv(RecordingClassifier(), self.x[:20], y, groups)
        self.assertEqual(sum("validation contains only one class" in str(w.message)
                             for w in caught), 5)
        auc = result["summary"]["roc_auc"]
        self.assertEqual((auc["n_valid"], auc["n_folds"]), (0, 5))
        self.assertTrue(np.isnan(auc["mean"]))
        self.assertTrue(np.isnan(auc["std"]))
        self.assertTrue(np.isfinite(result["overall"]["roc_auc"]))

    def test_single_class_training_fails_before_any_model_is_fitted(self):
        groups = np.repeat(["a", "b", "c"], 4)
        y = np.repeat([1, 0, 0], 4)
        with self.assertRaisesRegex(ValueError, "training set contains only one class"):
            patient_cv(RecordingClassifier(), self.x[:12], y, groups, 3)
        self.assertEqual(RecordingClassifier.fitted_rows, [])

    def test_inconsistent_arrays_and_grey_labels_are_rejected(self):
        for x, y, groups in (
            (self.x[:-1], self.y, self.groups),
            (self.x, self.y, self.groups[:-1]),
            (self.x, np.full(80, -1), self.groups),
            (self.x, np.full(80, 0.5), self.groups),
            (self.x, self.y[:, None], self.groups),
        ):
            with self.subTest(shape=np.shape(y)), self.assertRaises(ValueError):
                patient_cv(RecordingClassifier(), x, y, groups)

    def test_summary_matches_requested_example_and_handles_missing_auc(self):
        aucs = [0.79, 0.61, 0.66, 0.58, 0.69]
        summary = summarize_folds([{"roc_auc": value} for value in aucs])
        auc = summary["roc_auc"]
        self.assertAlmostEqual(auc["mean"], 0.666)
        self.assertAlmostEqual(auc["std"], 0.0728285658)
        summary = summarize_folds([{"roc_auc": 0.8}, {"roc_auc": np.nan}])
        self.assertEqual(summary["roc_auc"], {
            "mean": 0.8, "std": 0.0, "n_valid": 1, "n_folds": 2,
        })
        with self.assertRaises(ValueError):
            summarize_folds([])

    def test_binary_metrics_do_not_invent_missing_class_recall(self):
        result = binary_metrics([0, 0], [0.1, 0.8])
        self.assertTrue(np.isnan(result["sensitivity"]))
        self.assertTrue(np.isnan(result["roc_auc"]))
        self.assertEqual(result["specificity"], 0.5)
        for y, score, pred in (
            ([0, 1], [0.5], None),
            ([0, 1], [np.nan, 0.5], None),
            ([0, 1], [0.1, 0.9], [0, 2]),
        ):
            with self.assertRaises(ValueError):
                binary_metrics(y, score, pred)


if __name__ == "__main__":
    unittest.main()
