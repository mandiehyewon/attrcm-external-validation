"""External validation of an MGB-trained model card on a UCSF cohort.

The entire input CSV is treated as a test set (no training, no splitting). The
model card carries the frozen MGB preprocessing + coefficients + decision
threshold; this script only applies them.

Usage
-----
    python -m src.evaluate_external \
        --card    models/model_card_female.json \
        --data    hf_data/ucsf_cohort.csv \
        --out     hf_data/external_female/ \
        --label   TTR            # optional; omit for scoring-only

Outputs (under --out)
---------------------
    predictions.csv   EMPI (if present), risk_score, predicted_at_threshold
    metrics.json      external-validation metrics (only if --label is present
                      in the data with both classes)
    coverage.json     which card features were present vs imputed as fallback
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_card import (
    missing_feature_report,
    predict_from_card,
    predict_xgb_from_card,
)


def score_from_card(card: dict, df: pd.DataFrame, card_path: Path) -> np.ndarray:
    """Dispatch on the card's model_type.

    Linear cards score from the JSON alone; xgboost cards additionally load the
    booster named by card["booster_file"], resolved next to the card.
    """
    model_type = card.get("model_type", "logistic_regression")
    if model_type == "xgboost":
        return predict_xgb_from_card(card, df, Path(card_path).parent)
    if model_type == "logistic_regression":
        return predict_from_card(card, df)
    raise ValueError(f"Unsupported model_type '{model_type}' in {card_path}")


def _n_features(card: dict) -> int:
    if card.get("model_type") == "xgboost":
        return len(card["preprocessing"]["feature_order"])
    return len(card["features"])


def evaluate(card_path: str, data_path: str, out_dir: str, label_col: str | None) -> dict:
    card = json.loads(Path(card_path).read_text())
    df = pd.read_csv(data_path, low_memory=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Model card:  {card_path}")
    print(f"  type: {card.get('model_type', 'logistic_regression')}")
    print(f"  task: label={card['task']['label']} cohort={card['task']['cohort']}  "
          f"features={_n_features(card)}  threshold={card['decision_threshold']:.6f}")
    print(f"Data:        {data_path}  ({len(df)} rows)")

    # Feature coverage audit — external validity depends on this.
    coverage = missing_feature_report(card, df)
    if coverage["absent_using_fallback"]:
        print(f"  WARNING: {len(coverage['absent_using_fallback'])} card feature(s) "
              f"absent from the data, using frozen fallback (median / 0):")
        print("    " + ", ".join(coverage["absent_using_fallback"]))
    else:
        print("  All card features present in the data.")
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))

    # Score.
    threshold = float(card["decision_threshold"])
    risk = score_from_card(card, df, Path(card_path))
    pred = (risk >= threshold).astype(int)

    preds = pd.DataFrame({"risk_score": risk, "predicted_at_threshold": pred})
    for id_col in ("EMPI", "MRN", "PatientID", "patient_id"):
        if id_col in df.columns:
            preds.insert(0, id_col, df[id_col].values)
            break
    preds.to_csv(out / "predictions.csv", index=False)
    print(f"Predictions -> {out / 'predictions.csv'}  "
          f"(flagged {int(pred.sum())} / {len(pred)} = {100*pred.mean():.2f}%)")

    # Metrics — only if a usable label is present.
    if label_col and label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce")
        keep = y.notna().to_numpy()
        y = y[keep].astype(int).to_numpy()
        risk_k = risk[keep]
        if len(np.unique(y)) < 2:
            print(f"Label '{label_col}' has a single class in this cohort; "
                  "skipping discrimination metrics.")
            return card

        # src.eval.metrics here on the MGB side; src.metrics on the UCSF branch.
        try:
            from src.eval.metrics import (
                safe_average_precision,
                safe_roc_auc,
                threshold_metrics,
            )
        except ImportError:
            from src.metrics import (
                safe_average_precision,
                safe_roc_auc,
                threshold_metrics,
            )
        m = threshold_metrics(y, risk_k, threshold)
        m["roc_auc"] = safe_roc_auc(y, risk_k)
        m["average_precision"] = safe_average_precision(y, risk_k)
        m["n"] = int(len(y))
        m["n_positive"] = int(y.sum())
        m["prevalence"] = float(y.mean())
        m["decision_threshold"] = threshold
        m["mean_predicted_risk"] = float(risk_k.mean())
        (out / "metrics.json").write_text(json.dumps(m, indent=2))

        print("\n--- External-validation metrics ---")
        print(f"  n = {m['n']}   positives = {m['n_positive']} "
              f"({100*m['prevalence']:.2f}%)")
        print(f"  AUROC        {m['roc_auc']:.3f}")
        print(f"  AUPRC        {m['average_precision']:.3f}")
        print(f"  Sensitivity  {m['sensitivity']:.3f}   Specificity {m['specificity']:.3f}")
        print(f"  PPV          {m['ppv']:.3f}   NPV        {m['npv']:.3f}")
        print(f"  F1           {m['f1']:.3f}")
        print(f"Metrics -> {out / 'metrics.json'}")
    elif label_col:
        print(f"Label column '{label_col}' not found in data; scoring-only mode.")
    else:
        print("No --label given; scoring-only mode (no metrics computed).")

    return card


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--card", required=True, help="Path to the model_card.json.")
    p.add_argument("--data", required=True, help="UCSF cohort CSV (all rows treated as test).")
    p.add_argument("--out", default="external_validation", help="Output directory.")
    p.add_argument("--label", default=None, help="Outcome column for metrics (e.g. TTR). Optional.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    evaluate(args.card, args.data, args.out, args.label)
