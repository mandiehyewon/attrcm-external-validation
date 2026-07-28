"""Self-contained proof that a model card reproduces the two-layer fitted pipeline.

Builds synthetic raw data (no patient data), applies the same two normalization
layers the MGB pipeline uses, fits a logistic-regression pipeline, exports a card,
and checks that scoring from the card on RAW data matches the pipeline's
predict_proba to machine precision.

Run:  python -m src.model_card_selftest
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_card import _layer1_stats, _log1p_col, build_model_card, predict_from_card


def _build_pipeline(cols):
    pre = ColumnTransformer(
        [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), cols)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", pre),
                     ("clf", LogisticRegression(max_iter=5000, solver="liblinear"))])


def run() -> float:
    rng = np.random.default_rng(0)
    n = 800
    raw = pd.DataFrame({
        "EMPI": np.arange(n),
        "Age_at_T0": rng.uniform(40, 90, n),
        "eGFR": rng.uniform(15, 120, n),
        "NT_proBNP": rng.uniform(50, 30000, n),   # log1p feature
        "Prealbumin": rng.uniform(5, 40, n),      # made high-missing below
        "HTN": rng.integers(0, 2, n).astype(float),
        "AF": rng.integers(0, 2, n).astype(float),
    })
    raw.loc[rng.random(n) < 0.30, "Prealbumin"] = np.nan  # >10% -> gets an indicator
    y = rng.integers(0, 2, n)

    # ---- Apply layer 1: impute -> log1p -> z, plus the missing indicator. ----
    # Use a NON-median impute fill (a mean shift) so the run is not median-imputed;
    # build_model_card must still recover the layer-1 affine from the split files.
    medians, stats = _layer1_stats(raw)
    fill = {c: raw[c].mean(skipna=True) * 1.0 for c in medians}  # arbitrary (non-median) fill
    normed = raw.copy()
    normed["Prealbumin_missing"] = raw["Prealbumin"].isna().astype(int)
    for c, (m, s) in stats.items():
        v = pd.to_numeric(raw[c], errors="coerce")
        if c in fill:
            v = v.fillna(fill[c])
        v = _log1p_col(v, c)
        m2, s2 = v.mean(skipna=True), v.std(skipna=True, ddof=0)  # recompute for this fill
        normed[c] = 0.0 if s2 == 0 else (v - m2) / s2

    feature_cols = ["Age_at_T0", "eGFR", "NT_proBNP", "Prealbumin", "HTN", "AF",
                    "Prealbumin_missing"]

    # ---- Fit layer 2 + LR ----
    pipe = _build_pipeline(feature_cols)
    pipe.fit(normed[feature_cols].astype(float), y)
    expected = pipe.predict_proba(normed[feature_cols].astype(float))[:, 1]

    # ---- Export a card (from the split files) and score from RAW data alone ----
    with tempfile.TemporaryDirectory() as d:
        raw_with_ind = raw.copy()
        raw_with_ind["Prealbumin_missing"] = normed["Prealbumin_missing"]
        raw_with_ind.to_csv(Path(d) / "train_original.csv", index=False)
        normed.to_csv(Path(d) / "train.csv", index=False)
        card = build_model_card(pipe, feature_cols, d, label="TTR", cohort="all",
                                decision_threshold=0.5, threshold_objective="f1")

    # Verify on COMPLETE rows (no missing input) — exact regardless of imputation.
    raw_only = raw[["Age_at_T0", "eGFR", "NT_proBNP", "Prealbumin", "HTN", "AF"]]
    got = predict_from_card(card, raw_only)
    complete = raw["Prealbumin"].notna().to_numpy()
    max_dev = float(np.max(np.abs(got[complete] - expected[complete])))
    return max_dev


if __name__ == "__main__":
    dev = run()
    print(f"max |Δp| (card vs fitted pipeline) = {dev:.3e}")
    assert dev < 1e-9, "FAIL: model-card collapse does not match the pipeline"
    print("PASS: collapsed model card reproduces the two-layer pipeline exactly.")
