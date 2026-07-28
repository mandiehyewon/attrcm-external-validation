"""Reconstruct a portable model card from an already-trained LR results directory.

Use this when you have a finished `src.train` LR run and its normalized splits, and
want the model card WITHOUT re-training. It reconstructs the full raw -> probability
map from the saved artifacts and refuses to write a card unless that map reproduces
the run's own `test_predictions.csv` to < 1e-6.

It is imputation-agnostic: each feature's layer-1 normalization affine is recovered
directly from (train_original.csv, train.csv), so it works for median- and
MICE-imputed runs alike. Only logistic-regression runs are supported (linear).

Run (on the machine that holds the data — reads patient-level CSVs locally):

    python -m src.reconstruct_card_from_results \
        --results-dir /home/hyewonj/women_amyloidosis/hf_data/results_mice_f1_nomiss/all \
        --splits-dir  /home/hyewonj/women_amyloidosis/hf_data/normalized_mice/all \
        --cohort all \
        --out    models/model_card_lr_all.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_card import (
    CONTINUOUS_NORMALIZE_COLS,
    LOG_NORMALIZE_COLS,
    SCHEMA_VERSION,
    predict_from_card,
    sigmoid,
)

_PROB_COLS = ["risk_score", "y_prob", "proba", "predicted_proba", "prob", "score", "y_score"]


def _find_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find a {what} column in predictions "
                     f"(looked for {candidates}; got {list(df.columns)}).")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def reconstruct(results_dir: Path, splits_dir: Path, cohort: str) -> dict:
    results_dir, splits_dir = Path(results_dir), Path(splits_dir)

    meta = json.loads((results_dir / "logistic_regression_results.json").read_text())
    if str(meta.get("model", "lr")).lower() not in ("lr", "logistic_regression"):
        raise SystemExit(f"This tool only supports logistic-regression runs; "
                         f"results say model={meta.get('model')}.")
    feature_cols = meta["feature_cols"]
    threshold = float(meta["decision_threshold"])
    objective = meta.get("threshold_selection_metric", meta.get("threshold_objective", "f1"))

    or_ci = pd.read_csv(results_dir / "logistic_or_ci.csv").set_index("variable")
    coef = or_ci["coefficient"].to_dict()

    train = pd.read_csv(splits_dir / "train.csv", low_memory=False)          # layer-1 normalized
    train_orig = pd.read_csv(splits_dir / "train_original.csv", low_memory=False)  # raw

    continuous = set(CONTINUOUS_NORMALIZE_COLS)
    features: list[dict] = []
    for name in feature_cols:
        if name not in coef:
            raise SystemExit(f"Feature '{name}' missing from logistic_or_ci.csv.")
        c = float(coef[name])
        g = pd.to_numeric(train[name], errors="coerce")
        mean2 = float(g.mean())
        scale2 = float(g.std(ddof=0)) or 1.0

        if name in continuous:
            log = name in LOG_NORMALIZE_COLS
            raw = pd.to_numeric(train_orig[name], errors="coerce")
            v = np.log1p(raw.where(raw > -1)) if log else raw
            mask = v.notna() & g.notna()
            # g = a*v + b exactly on non-imputed rows -> recover layer-1 affine.
            a, b = np.polyfit(v[mask].to_numpy(), g[mask].to_numpy(), 1)
            weight = c * a / scale2
            const = c * (b - mean2) / scale2
            features.append({
                "name": name, "kind": "continuous",
                "transform": "log1p" if log else "identity",
                "impute_raw": float(raw.median(skipna=True)),
                "weight": float(weight), "_const": float(const),
            })
        else:
            weight = c / scale2
            const = -c * mean2 / scale2
            feat = {"name": name, "kind": "binary", "transform": "identity",
                    "impute_raw": 0.0, "weight": float(weight), "_const": float(const)}
            if name.endswith("_missing"):
                feat["kind"] = "missing_indicator"
                feat["source"] = name[: -len("_missing")]
            features.append(feat)

    # Recover the effective intercept from complete test rows, then verify on all.
    test_orig = pd.read_csv(splits_dir / "test_original.csv", low_memory=False)
    preds = pd.read_csv(results_dir / "test_predictions.csv", low_memory=False)
    if len(test_orig) != len(preds):
        raise SystemExit(f"Row mismatch: test_original.csv ({len(test_orig)}) vs "
                         f"test_predictions.csv ({len(preds)}).")
    prob_col = _find_col(preds, _PROB_COLS, "predicted-probability")
    logit_obs = _logit(pd.to_numeric(preds[prob_col], errors="coerce").to_numpy())

    # linear part sum_i weight_i * transform_i(raw_i) with card imputation
    card_stub = {"intercept": 0.0, "features": features}
    lin = _logit(predict_from_card(card_stub, test_orig))  # sigmoid(0+lin) -> logit == lin
    ok = np.isfinite(logit_obs) & np.isfinite(lin)
    intercept = float(np.median(logit_obs[ok] - lin[ok]))

    card = {
        "schema_version": SCHEMA_VERSION,
        "model_type": "logistic_regression",
        "task": {"label": "TTR", "cohort": cohort},
        "predict": ("p = 1/(1+exp(-(intercept + sum_i weight_i * transform_i(x_i)))); "
                    "positive if p >= decision_threshold"),
        "intercept": intercept,
        "features": [{k: v for k, v in f.items() if k != "_const"} for f in features],
        "decision_threshold": threshold,
        "threshold_objective": objective,
        "metadata": {
            "reconstructed_from": str(results_dir),
            "splits_dir": str(splits_dir),
            "n_features": len(features),
            "note": "Reconstructed from saved LR artifacts; verified vs test_predictions.csv.",
        },
    }

    # Hard verification against the run's own saved predictions.
    p_hat = predict_from_card(card, test_orig)
    p_obs = pd.to_numeric(preds[prob_col], errors="coerce").to_numpy()
    max_dev = float(np.nanmax(np.abs(p_hat - p_obs)))
    card["metadata"]["verified_max_abs_prob_diff"] = max_dev
    if max_dev > 1e-6:
        raise SystemExit(
            f"VERIFICATION FAILED: reconstructed card deviates from saved predictions "
            f"by max |Δp| = {max_dev:.3e} (> 1e-6). Card NOT written. This usually means "
            f"a preprocessing detail differs from what this tool assumes — tell the human.")
    print(f"Verified: reconstructed card reproduces test_predictions.csv "
          f"(max |Δp| = {max_dev:.2e}).")
    return card


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="e.g. .../results_mice_f1_nomiss/all")
    p.add_argument("--splits-dir", required=True, help="matching splits, e.g. .../normalized_mice/all")
    p.add_argument("--cohort", default="all", choices=["all", "male", "female"])
    p.add_argument("--out", required=True, help="output card path, e.g. models/model_card_lr_all.json")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    card = reconstruct(args.results_dir, args.splits_dir, args.cohort)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(card, indent=2))
    print(f"Wrote {args.out}  ({len(card['features'])} features, "
          f"threshold={card['decision_threshold']:.6f}, objective={card['threshold_objective']}).")
