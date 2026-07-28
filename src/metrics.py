from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Generic, robust helpers (reused from the train.py screening utilities)
# ---------------------------------------------------------------------------


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(np.asarray(y_true))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if int(np.sum(np.asarray(y_true))) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


# Validation metric used to pick the best hyperparameters. AUROC is the default;
# --auprc switches model selection to validation average precision (AUPRC).
SELECTION_SCORERS = {
    "auroc": safe_roc_auc,
    "auprc": safe_average_precision,
}


def positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) for classifiers whose classes_ order may vary."""
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", np.array([0, 1])))
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(len(X), dtype=float)

def threshold_grid(y_prob: np.ndarray) -> np.ndarray:
    """Threshold candidates: regular grid union observed prediction probabilities."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = y_prob[np.isfinite(y_prob)]
    if y_prob.size == 0:
        return np.linspace(0.0, 1.0, 101)
    return np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), y_prob]))

def threshold_metrics(y_true, y_prob: np.ndarray, threshold: float) -> dict:
    """Binary classification metrics at a given probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "ppv": precision,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "predicted_positive_rate": float((tp + fp) / len(y_true)) if len(y_true) else 0.0,
        "flagged_n": int(tp + fp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def threshold_sweep(y_true, y_prob: np.ndarray) -> pd.DataFrame:
    """Return metrics for every candidate probability threshold."""
    rows = [threshold_metrics(y_true, y_prob, t) for t in threshold_grid(y_prob)]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_max_sensitivity_threshold(metrics_df: pd.DataFrame) -> dict:
    """Max sensitivity; tie-break by accuracy, F1, precision, specificity, threshold."""
    sort_cols = ["sensitivity", "accuracy", "f1", "precision", "specificity", "threshold"]
    selected = metrics_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.to_dict().items()
    }


def select_max_f1_threshold(metrics_df: pd.DataFrame) -> dict:
    """Max F1; tie-break by sensitivity, accuracy, precision, threshold."""
    sort_cols = ["f1", "sensitivity", "accuracy", "precision", "threshold"]
    selected = metrics_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.to_dict().items()
    }


def select_max_youden_threshold(metrics_df: pd.DataFrame) -> dict:
    """Max Youden's J (= sensitivity + specificity - 1); tie-break by sensitivity,
    specificity, then higher threshold."""
    df = metrics_df.copy()
    df["youden_j"] = df["sensitivity"] + df["specificity"] - 1.0
    sort_cols = ["youden_j", "sensitivity", "specificity", "threshold"]
    selected = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.to_dict().items()
    }


# Validation-threshold objective -> selector. Chosen via --sensitivity / --youden
# (default: max F1). Each picks the operating point on the validation sweep.
THRESHOLD_SELECTORS = {
    "f1": select_max_f1_threshold,
    "sensitivity": select_max_sensitivity_threshold,
    "youden": select_max_youden_threshold,
}


def print_threshold_summary(label: str, metrics: dict) -> None:
    print(
        f"{label}: threshold={metrics['threshold']:.6f}  "
        f"sensitivity={metrics['sensitivity']:.4f}  "
        f"accuracy={metrics['accuracy']:.4f}  "
        f"specificity={metrics['specificity']:.4f}  "
        f"precision={metrics['precision']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"flagged={metrics['flagged_n']}"
    )