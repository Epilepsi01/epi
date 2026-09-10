"""Кросс-валидация по пациентам и метрики бинарной классификации."""

import operator
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity")


def _binary_labels(values, name="y"):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError("%s must be a non-empty 1D array of binary labels 0/1" % name)
    return values.astype(int)


def _patient_groups(groups):
    # Сохранить NaN в смешанном списке строк и чисел: обычный asarray
    # превратил бы его в строку "nan", то есть в ложного нового пациента.
    groups = np.asarray(groups, dtype=object)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("groups must be a non-empty 1D array of patient keys")
    if any(g is None or g != g or (isinstance(g, str) and not g.strip())
           for g in groups):
        raise ValueError("groups must not contain missing patient keys")
    return groups


def binary_metrics(y, score, pred=None, threshold=0.5):
    """ROC AUC, average precision и метрики при заданных предсказаниях.

    AUC не определён для одного класса; recall отсутствующего класса тоже
    возвращается как NaN, а не как искусственный ноль.
    """
    y = _binary_labels(y)
    score = np.asarray(score, dtype=float)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("score must contain one finite value per label")
    pred = ((score >= threshold).astype(int) if pred is None
            else _binary_labels(pred, "pred"))
    if pred.shape != y.shape:
        raise ValueError("pred and y must have the same length")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "roc_auc": roc_auc_score(y, score) if both else np.nan,
        "pr_auc": average_precision_score(y, score) if both else np.nan,
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def patient_folds(groups, n_splits=5, *, y=None, seed=0):
    """Индексы (train, val): все окна пациента всегда в одной части.

    Каждый пациент ровно один раз попадает в val. Если передан y, используется
    StratifiedGroupKFold: он старается сбалансировать классы, сохраняя группы
    целыми. Без y остаётся GroupKFold с балансировкой числа окон.

    seed фиксирует перемешивание при стратификации. Если пациентов меньше
    n_splits, число фолдов сокращается с предупреждением до числа пациентов.
    Группы должны относиться только к тренировочной части; test сюда не входит.
    """
    groups = _patient_groups(groups)
    try:
        if isinstance(n_splits, (bool, np.bool_)):
            raise TypeError
        n_splits = operator.index(n_splits)
    except TypeError as error:
        raise ValueError("n_splits must be an integer >= 2") from error
    if n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    n_patients = len(np.unique(groups))
    if n_patients < 2:
        raise ValueError("at least two patients are required")
    if n_splits > n_patients:
        warnings.warn(
            "requested %d folds but only %d patients are available; using %d folds"
            % (n_splits, n_patients, n_patients),
            UserWarning, stacklevel=2,
        )
        n_splits = n_patients
    dummy = np.zeros(len(groups))
    if y is None:
        yield from GroupKFold(n_splits=n_splits).split(dummy, groups=groups)
    else:
        y = _binary_labels(y)
        if len(y) != len(groups):
            raise ValueError("y and groups must have the same length")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed,
        )
        yield from splitter.split(dummy, y, groups)


def summarize_folds(folds):
    """Невзвешенные mean/std метрик по фолдам, std с ddof=0.

    Неопределённые метрики исключаются отдельно для каждой метрики; n_valid и
    n_folds показывают, сколько фолдов вошло в расчёт. Если годных нет, mean и
    std равны NaN. Можно использовать и со своим циклом обучения.
    """
    folds = list(folds)
    if not folds:
        raise ValueError("at least one fold is required")
    summary = {}
    for name in METRICS:
        if not any(name in row for row in folds):
            continue
        values = np.array([row.get(name, np.nan) for row in folds], dtype=float)
        valid = values[np.isfinite(values)]
        summary[name] = {
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "std": float(valid.std(ddof=0)) if len(valid) else np.nan,
            "n_valid": len(valid),
            "n_folds": len(folds),
        }
    return summary


def _score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x).astype(float)


def patient_cv(model, x, y, groups, n_splits=5, *, seed=0, verbose=False):
    """Обучить свежую копию sklearn-модели на каждом фолде пациентов.

    Передавайте только данные train и соответствующие patient_key. Scaler,
    imputer, отбор признаков и прочие обучаемые преобразования должны входить
    в Pipeline, чтобы fit выполнялся только на train текущего фолда.

    summary -- mean/std по фолдам; overall -- метрики объединённых out-of-fold
    предсказаний (они не равны средним по фолдам). scores/predictions имеют тот
    же порядок, что y; indices содержит пары индексов (train, val). В folds
    сохраняются прежние поля test/patients как синонимы val/val_patients.
    """
    x = np.asarray(x)
    y = _binary_labels(y)
    groups = _patient_groups(groups)
    if x.ndim < 2 or len(x) != len(y) or len(groups) != len(y):
        raise ValueError("x, y and groups must have the same number of windows")

    indices = list(patient_folds(groups, n_splits, y=y, seed=seed))
    # Проверить все фолды до первого fit, чтобы не потратить несколько
    # обучений и только потом обнаружить единственного носителя одного класса.
    for fold, (train, val) in enumerate(indices, start=1):
        if not len(train) or not len(val):
            raise ValueError("fold %d has an empty train or val set" % fold)
        if len(np.unique(y[train])) < 2:
            raise ValueError(
                "fold %d: training set contains only one class; "
                "patient-level CV requires more patients carrying the missing class"
                % fold
            )

    scores = np.full(len(y), np.nan, dtype=float)
    predictions = np.full(len(y), -1, dtype=int)
    folds = []
    for fold, (train, val) in enumerate(indices, start=1):
        if len(np.unique(y[val])) < 2:
            warnings.warn(
                "fold %d: validation contains only one class; "
                "ROC AUC and PR AUC are undefined (NaN)" % fold,
                UserWarning, stacklevel=2,
            )
        fitted = clone(model)
        fitted.fit(x[train], y[train])
        score = _score(fitted, x[val])
        pred = fitted.predict(x[val])
        metrics = binary_metrics(y[val], score, pred)
        scores[val] = score
        predictions[val] = pred
        val_patients = len(np.unique(groups[val]))
        row = {
            "fold": fold,
            "train": len(train),
            "val": len(val),
            "test": len(val),
            "train_patients": len(np.unique(groups[train])),
            "val_patients": val_patients,
            "patients": val_patients,
            "train_positive": int(y[train].sum()),
            "val_positive": int(y[val].sum()),
            **metrics,
        }
        folds.append(row)
        if verbose:
            print("fold %d: train %d patients / %d windows, "
                  "val %d patients / %d windows; AUC = %.3f"
                  % (fold, row["train_patients"], len(train),
                     val_patients, len(val), row["roc_auc"]))

    summary = summarize_folds(folds)
    if verbose:
        auc = summary["roc_auc"]
        print("ROC AUC mean = %.3f, std = %.3f (%d/%d valid folds)"
              % (auc["mean"], auc["std"], auc["n_valid"], auc["n_folds"]))
    return {
        "overall": binary_metrics(y, scores, predictions),
        "summary": summary,
        "folds": folds,
        "n_splits": len(folds),
        "indices": indices,
        "scores": scores,
        "predictions": predictions,
    }
