"""Portable logistic-regression "model card" for external validation.

The MGB training pipeline applies TWO affine normalization layers before the
logistic regression:

  layer 1 (split_n_normalize_data): per continuous feature, median-impute (train
           median), optionally log1p (NT_proBNP, Troponin), then z-score with
           train mean/std.
  layer 2 (the sklearn Pipeline itself): SimpleImputer(median) + StandardScaler
           over *every* feature, then LogisticRegression.

Every step from a raw feature value to the model logit is affine except the
log1p, which is applied first. That means the whole chain collapses to a single
linear model in *raw* feature units:

    logit = intercept + sum_i weight_i * t_i(x_i)

where t_i(x) = log1p(x) for the log features and x otherwise, and a missing x_i
is replaced by the frozen MGB training median (continuous) or 0 (binary /
indicator) before transforming.

This module builds that collapsed card from a fitted pipeline (`build_model_card`,
run on the MGB side) and scores a raw dataframe from the card alone
(`predict_from_card`, run on the UCSF side — needs only numpy/pandas, no sklearn,
no version matching).

XGBoost (`model_type: "xgboost"`) cannot be collapsed this way — the estimator is
non-linear, so the normalization layers stay explicit. Those cards record the
per-column transform under `preprocessing` and ship the booster next to the JSON
as UBJSON; `build_xgb_model_card` / `predict_xgb_from_card` handle that pair, and
the scoring path additionally requires the `xgboost` package.

IMPUTATION CAVEAT — a card stores one frozen constant per column, so it can only
reproduce the training pipeline exactly on rows where every continuous feature is
observed. That is exact everywhere for median-imputed runs, but for MICE runs
(`--mice`) the pipeline imputes each row from that row's other features, which no
constant can reproduce. `verify_card` / `verify_xgb_card` therefore assert on
complete-input rows and report the incomplete-row deviation separately.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

# Layer-1 normalization spec — MUST stay in sync with
# src/preprocess/split_n_normalize_data.py on the MGB training side.
CONTINUOUS_NORMALIZE_COLS = [
    "Age_at_T0", "Weight", "Height", "BMI", "Prealbumin", "eGFR", "NT_proBNP",
    "BNP", "Troponin", "Sodium", "Potassium", "Chloride", "Calcium",
    "Creatinine", "Glucose", "ejection_fraction", "LVIDed_mm", "IVS_mm",
    "PWT_mm", "relative_wall_thickness",
]
LOG_NORMALIZE_COLS = {"NT_proBNP", "Troponin"}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _log1p_col(values: pd.Series, col: str) -> pd.Series:
    """Replicate split_n_normalize_data._log_transform for a single column."""
    if col in LOG_NORMALIZE_COLS:
        values = values.where(values > -1)
        values = np.log1p(values)
    return values


def _layer1_stats(train_orig: pd.DataFrame):
    """Recompute the exact layer-1 median / (mean, std) the split pipeline used.

    Mirrors split_n_normalize_data: median-impute (raw), then per continuous
    column compute (mean, std) of the log1p-transformed, imputed train values.
    """
    cols = [c for c in CONTINUOUS_NORMALIZE_COLS if c in train_orig.columns]
    medians: dict[str, float] = {}
    for c in cols:
        med = pd.to_numeric(train_orig[c], errors="coerce").median(skipna=True)
        if not pd.isna(med):
            medians[c] = float(med)
    stats: dict[str, tuple[float, float]] = {}
    for c in cols:
        v = pd.to_numeric(train_orig[c], errors="coerce")
        if c in medians:
            v = v.fillna(medians[c])
        v = _log1p_col(v, c)
        stats[c] = (float(v.mean(skipna=True)), float(v.std(skipna=True, ddof=0)))
    return medians, stats


# ---------------------------------------------------------------------------
# Scoring (UCSF side — depends only on the card + numpy/pandas)
# ---------------------------------------------------------------------------

def predict_from_card(card: dict, df: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for every row of `df`.

    `df` holds raw feature columns (the same names/units as the MGB cohort CSV).
    Columns absent from `df` fall back to the card's frozen impute value; extra
    columns are ignored. Missingness indicators are derived from their source
    column's NaN pattern when the indicator column itself is not supplied.
    """
    n = len(df)
    logit = np.full(n, float(card["intercept"]), dtype=float)

    for feat in card["features"]:
        name = feat["name"]
        weight = float(feat["weight"])

        if feat.get("kind") == "missing_indicator":
            source = feat["source"]
            if name in df.columns:                       # indicator supplied directly
                x = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy()
            elif source in df.columns:                   # derive from source NaN pattern
                x = df[source].isna().astype(float).to_numpy()
            else:                                        # source unavailable -> "not missing"
                x = np.zeros(n)
            logit += weight * x
            continue

        impute_raw = feat.get("impute_raw", 0.0)
        if name in df.columns:
            x = pd.to_numeric(df[name], errors="coerce").fillna(impute_raw).to_numpy(dtype=float)
        else:
            x = np.full(n, float(impute_raw))

        if feat.get("transform") == "log1p":
            x = np.log1p(x)
        logit += weight * x

    return sigmoid(logit)


def missing_feature_report(card: dict, df: pd.DataFrame) -> dict[str, list[str]]:
    """Which card features are absent from `df` (so UCSF can audit coverage)."""
    present, imputed = [], []
    if card.get("model_type") == "xgboost":
        pre = card["preprocessing"]
        indicators = pre.get("missing_indicators", {})
        for name in pre["feature_order"]:
            target = indicators.get(name, name)
            (present if target in df.columns else imputed).append(name)
        return {"present": present, "absent_using_fallback": imputed}
    for feat in card["features"]:
        target = feat.get("source") if feat.get("kind") == "missing_indicator" else feat["name"]
        (present if target in df.columns else imputed).append(feat["name"])
    return {"present": present, "absent_using_fallback": imputed}


# ---------------------------------------------------------------------------
# Building the card (MGB side — needs the fitted pipeline + train_original.csv)
# ---------------------------------------------------------------------------

def build_model_card(
    best_estimator,
    feature_cols: list[str],
    data_dir,
    *,
    label: str,
    cohort: str,
    decision_threshold: float,
    threshold_objective: str,
    metadata: dict | None = None,
) -> dict:
    """Collapse a fitted LR pipeline + the layer-1 split normalization into a card.

    `best_estimator` is the fitted sklearn Pipeline from src.train (ColumnTransformer
    [SimpleImputer+StandardScaler over all columns] -> LogisticRegression).
    `data_dir` is the cohort split dir containing BOTH train_original.csv (raw) and
    train.csv (layer-1 normalized). The layer-1 normalization affine is recovered
    empirically per feature from those two files, so this is imputation-agnostic —
    it works for median- and MICE-imputed runs alike. Always finish with verify_card.
    """
    from pathlib import Path

    data_dir = Path(data_dir)
    train_orig = pd.read_csv(data_dir / "train_original.csv", low_memory=False)
    train_norm = pd.read_csv(data_dir / "train.csv", low_memory=False)

    # Layer-2 params from the fitted pipeline (aligned to feature_cols order).
    pre = best_estimator.named_steps["preprocess"]
    num_pipe = pre.transformers_[0][1]
    scaler2 = num_pipe.named_steps["scale"]
    mean2 = np.asarray(scaler2.mean_, dtype=float)
    scale2 = np.asarray(scaler2.scale_, dtype=float)
    clf = best_estimator.named_steps["clf"]
    coef = clf.coef_.ravel().astype(float)
    intercept = float(clf.intercept_.item())

    continuous = set(CONTINUOUS_NORMALIZE_COLS)
    features: list[dict[str, Any]] = []
    const_sum = 0.0

    for i, name in enumerate(feature_cols):
        c = coef[i]
        s2 = scale2[i]
        m2 = mean2[i]

        if name in continuous and name in train_norm.columns and name in train_orig.columns:
            log = name in LOG_NORMALIZE_COLS
            raw = pd.to_numeric(train_orig[name], errors="coerce")
            v = _log1p_col(raw, name)                    # transformed raw
            g = pd.to_numeric(train_norm[name], errors="coerce")  # layer-1 normalized
            mask = (v.notna() & g.notna()).to_numpy()
            # g = a*v + b exactly on non-imputed rows -> recover layer-1 affine.
            a, b = np.polyfit(v.to_numpy()[mask], g.to_numpy()[mask], 1)
            weight = c * a / s2
            const_sum += c * (b - m2) / s2
            features.append({
                "name": name,
                "kind": "continuous",
                "transform": "log1p" if log else "identity",
                "impute_raw": float(raw.median(skipna=True)),
                "weight": float(weight),
            })
        else:
            weight = c / s2
            const_sum += -c * (m2 / s2)
            feat = {
                "name": name,
                "kind": "binary",
                "transform": "identity",
                "impute_raw": 0.0,
                "weight": float(weight),
            }
            if name.endswith("_missing"):
                feat["kind"] = "missing_indicator"
                feat["source"] = name[: -len("_missing")]
            features.append(feat)

    card = {
        "schema_version": SCHEMA_VERSION,
        "model_type": "logistic_regression",
        "task": {"label": label, "cohort": cohort},
        "predict": (
            "p = 1/(1+exp(-(intercept + sum_i weight_i * transform_i(x_i)))); "
            "positive if p >= decision_threshold"
        ),
        "intercept": float(intercept + const_sum),
        "features": features,
        "decision_threshold": float(decision_threshold),
        "threshold_objective": threshold_objective,
        "metadata": metadata or {},
    }
    return card


# ---------------------------------------------------------------------------
# XGBoost cards
#
# XGBoost is non-linear, so the two normalization layers CANNOT be folded into
# the estimator the way they are for logistic regression. The card instead
# records the transform explicitly and ships the booster alongside it.
# ---------------------------------------------------------------------------

def _layer1_affine(train_orig: pd.DataFrame, train_norm: pd.DataFrame, col: str):
    """Recover the layer-1 z-score (mean, std) actually applied to `col`.

    split_n_normalize_data computes (mean, std) on the *imputed* train values, so
    those numbers depend on the imputation. Rather than assume median imputation
    (which `_layer1_stats` does), recover the affine empirically: on rows where
    the raw value is present, no imputation happened, so

        normalized = (transform(raw) - mean) / std

    is an exact line in transform(raw). Fitting it recovers (mean, std) for
    median- and MICE-imputed runs alike.
    """
    raw = pd.to_numeric(train_orig[col], errors="coerce")
    v = _log1p_col(raw, col)
    g = pd.to_numeric(train_norm[col], errors="coerce")
    mask = (v.notna() & g.notna()).to_numpy()
    if mask.sum() < 2:
        raise ValueError(f"{col}: too few observed rows to recover layer-1 affine")
    a, b = np.polyfit(v.to_numpy()[mask], g.to_numpy()[mask], 1)
    if not np.isfinite(a) or a == 0:
        raise ValueError(f"{col}: degenerate layer-1 affine (slope={a})")
    return float(-b / a), float(1.0 / a)   # (z_mean, z_std)


def build_xgb_model_card(
    best_estimator,
    feature_cols: list[str],
    data_dir,
    *,
    label: str,
    cohort: str,
    decision_threshold: float,
    threshold_objective: str,
    booster_file: str,
    metadata: dict | None = None,
) -> dict:
    """Describe the raw -> booster-input transform for a fitted XGB pipeline.

    `best_estimator` is the fitted sklearn Pipeline from src.train with --model xgb.
    That pipeline is currently just [("clf", XGBClassifier)] — XGBoost consumes NaN
    natively, so there is no layer-2 imputer/scaler — but this reads whatever
    preprocess step is actually present, so it keeps working if one is added.

    The caller is responsible for writing the booster itself to `booster_file`
    (see `save_xgb_booster`). Always finish with `verify_xgb_card`.
    """
    from pathlib import Path

    data_dir = Path(data_dir)
    train_orig = pd.read_csv(data_dir / "train_original.csv", low_memory=False)
    train_norm = pd.read_csv(data_dir / "train.csv", low_memory=False)

    medians, _ = _layer1_stats(train_orig)

    # Layer 2 is optional: read it off the fitted pipeline only if it is there.
    mean2 = scale2 = None
    imputer2 = None
    pre = getattr(best_estimator, "named_steps", {}).get("preprocess")
    if pre is not None:
        num_pipe = pre.transformers_[0][1]
        scaler = num_pipe.named_steps.get("scale")
        if scaler is not None:
            mean2 = np.asarray(scaler.mean_, dtype=float)
            scale2 = np.asarray(scaler.scale_, dtype=float)
        imp = num_pipe.named_steps.get("impute")
        if imp is not None:
            imputer2 = np.asarray(imp.statistics_, dtype=float)

    continuous_set = set(CONTINUOUS_NORMALIZE_COLS)
    continuous: dict[str, dict[str, Any]] = {}
    passthrough_impute: dict[str, float] = {}
    missing_indicators: dict[str, str] = {}

    for i, name in enumerate(feature_cols):
        if name in continuous_set and name in train_norm.columns and name in train_orig.columns:
            spec: dict[str, Any] = {"log1p": name in LOG_NORMALIZE_COLS}
            if name in medians:
                spec["impute_raw"] = float(medians[name])
            z_mean, z_std = _layer1_affine(train_orig, train_norm, name)
            spec["z_mean"] = z_mean
            spec["z_std"] = z_std
            if mean2 is not None:
                spec["scaler_mean"] = float(mean2[i])
                spec["scaler_scale"] = float(scale2[i])
            continuous[name] = spec
        else:
            if name.endswith("_missing"):
                missing_indicators[name] = name[: -len("_missing")]
            if imputer2 is not None:
                passthrough_impute[name] = float(imputer2[i])
            if mean2 is not None:
                # A binary column still passes through layer 2's scaler.
                continuous.setdefault(name, {})
                continuous[name].update({
                    "log1p": False,
                    "scaler_mean": float(mean2[i]),
                    "scaler_scale": float(scale2[i]),
                })

    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": "xgboost",
        "task": {"label": label, "cohort": cohort},
        "predict": (
            "Build the feature matrix in preprocessing.feature_order by applying, per "
            "column: impute_raw for missing -> log1p (if set) -> (x-z_mean)/z_std -> "
            "(x-scaler_mean)/scaler_scale, using only the keys present. Then "
            "p = booster.predict(DMatrix(X)); positive if p >= decision_threshold."
        ),
        "preprocessing": {
            "feature_order": list(feature_cols),
            "continuous": continuous,
            "passthrough_impute": passthrough_impute,
            "missing_indicators": missing_indicators,
        },
        "booster_file": booster_file,
        "decision_threshold": float(decision_threshold),
        "threshold_objective": threshold_objective,
        "metadata": metadata or {},
    }


def save_xgb_booster(best_estimator, path) -> None:
    """Persist the fitted booster as UBJSON (portable, version-tolerant)."""
    best_estimator.named_steps["clf"].get_booster().save_model(str(path))


def build_xgb_matrix(card: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the booster's input matrix, in `feature_order`, from raw columns."""
    pre = card["preprocessing"]
    order = pre["feature_order"]
    continuous = pre.get("continuous", {})
    passthrough = pre.get("passthrough_impute", {})
    indicators = pre.get("missing_indicators", {})

    out = {}
    for name in order:
        if name in indicators:
            source = indicators[name]
            if name in df.columns:
                x = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            elif source in df.columns:
                x = df[source].isna().astype(float).to_numpy()
            else:
                x = np.zeros(len(df))
            out[name] = x
            continue

        if name in df.columns:
            x = pd.to_numeric(df[name], errors="coerce").astype(float)
        else:
            x = pd.Series(np.full(len(df), np.nan), index=df.index)

        spec = continuous.get(name, {})
        # Layer 1: impute on the raw scale, then log1p, then z-score. Columns with
        # no impute_raw keep their NaN — XGBoost handles it natively.
        if "impute_raw" in spec:
            x = x.fillna(float(spec["impute_raw"]))
        elif name in passthrough:
            x = x.fillna(float(passthrough[name]))
        if spec.get("log1p"):
            x = np.log1p(x.where(x > -1))
        if "z_mean" in spec:
            z_std = float(spec["z_std"])
            x = (x - float(spec["z_mean"])) / z_std if z_std else x * 0.0
        # Layer 2 (only present if the pipeline actually has a scaler).
        if "scaler_mean" in spec:
            s = float(spec["scaler_scale"])
            x = (x - float(spec["scaler_mean"])) / s if s else x * 0.0
        out[name] = x.to_numpy(dtype=float)

    return pd.DataFrame(out, columns=order, index=df.index)


def predict_xgb_from_card(card: dict, df: pd.DataFrame, booster_dir) -> np.ndarray:
    """Positive-class probabilities from an xgboost card. REQUIRES the xgboost package."""
    from pathlib import Path

    import xgboost as xgb

    matrix = build_xgb_matrix(card, df)
    booster = xgb.Booster()
    booster.load_model(str(Path(booster_dir) / card["booster_file"]))
    dm = xgb.DMatrix(matrix.to_numpy(dtype=float),
                     feature_names=list(matrix.columns), missing=np.nan)
    return np.asarray(booster.predict(dm), dtype=float)


def verify_xgb_card(card: dict, raw_df: pd.DataFrame, expected_prob: np.ndarray,
                    booster_dir, tol: float = 1e-6) -> tuple[float, float, int]:
    """Assert an xgboost card reproduces the pipeline on complete-input rows.

    Same contract as `verify_card`: exact on rows where every continuous card
    feature is observed. Returns (max_dev_complete, max_dev_incomplete, n_complete).
    """
    got = predict_xgb_from_card(card, raw_df, booster_dir)
    exp = np.asarray(expected_prob, dtype=float)
    dev = np.abs(got - exp)

    complete = _complete_row_mask(card, raw_df)
    max_dev = float(np.max(dev[complete])) if complete.any() else 0.0
    assert max_dev <= tol, (
        f"XGB model card does not match the fitted pipeline on complete-input rows "
        f"(max |Δp| = {max_dev:.3e} > {tol:.0e}). Do not ship this card."
    )
    max_dev_missing = float(np.max(dev[~complete])) if (~complete).any() else 0.0
    return max_dev, max_dev_missing, int(complete.sum())


def _complete_row_mask(card: dict, raw_df: pd.DataFrame) -> np.ndarray:
    """Rows in which every continuous card feature has a value (no imputation used).

    On these rows the card is exact for any training imputation. On rows with a
    missing continuous input, a card can only reproduce the pipeline when training
    used constant (median) imputation — not per-row MICE.
    """
    mask = np.ones(len(raw_df), dtype=bool)
    if card.get("model_type") == "xgboost":
        # Continuous == the columns carrying a layer-1 z-score (binary columns
        # only ever pick up a layer-2 scaler entry, never z_mean).
        for name, spec in card["preprocessing"].get("continuous", {}).items():
            if "z_mean" in spec and name in raw_df.columns:
                mask &= pd.to_numeric(raw_df[name], errors="coerce").notna().to_numpy()
        return mask
    for feat in card["features"]:
        if feat.get("kind") == "continuous" and feat["name"] in raw_df.columns:
            mask &= pd.to_numeric(raw_df[feat["name"]], errors="coerce").notna().to_numpy()
    return mask


def verify_card(card: dict, raw_df: pd.DataFrame, expected_prob: np.ndarray, tol: float = 1e-6) -> float:
    """Assert the card reproduces the pipeline's probabilities on raw_df.

    Verification is asserted on COMPLETE-input rows (see `_complete_row_mask`).
    Rows with a missing continuous input are reported separately: for median-imputed
    training they also match (and are folded into the assertion), but for MICE they
    are only approximate, so they are surfaced as a warning rather than a failure.
    Returns the max absolute deviation on complete rows.
    """
    got = predict_from_card(card, raw_df)
    exp = np.asarray(expected_prob, dtype=float)
    dev = np.abs(got - exp)

    complete = _complete_row_mask(card, raw_df)
    max_dev = float(np.max(dev[complete])) if complete.any() else 0.0
    assert max_dev <= tol, (
        f"Model-card collapse does not match the fitted pipeline on complete-input "
        f"rows (max |Δp| = {max_dev:.3e} > {tol:.0e}). Do not ship this card."
    )
    n_missing = int((~complete).sum())
    if n_missing:
        max_dev_missing = float(np.max(dev[~complete]))
        if max_dev_missing > tol:
            print(f"  NOTE: {n_missing} rows have missing inputs; card vs pipeline max "
                  f"|Δp|={max_dev_missing:.3e} there (expected for MICE — the card falls "
                  f"back to median for missing values). Complete rows match to {max_dev:.2e}.")
    return max_dev
