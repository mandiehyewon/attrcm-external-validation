#!/usr/bin/env bash
#
# External-validation driver.
#
# Validate a UCSF cohort against a frozen MGB model card:
#     bash run.sh <card.json> <cohort.csv> [out_dir] [label_col]
#   e.g.
#     bash run.sh models/model_card_female.json hf_data/ucsf_cohort.csv external_female TTR
#
# Smoke test (no real data) — score the committed synthetic fixture with the dummy
# card and confirm the metrics match the committed baseline exactly:
#     bash run.sh --smoke
#
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--smoke" ]]; then
  echo "=== SMOKE TEST: card math + evaluator pipeline (no patient data) ==="
  python -m src.model_card_selftest
  echo
  python -m src.evaluate_external \
    --card models/model_card_lr_all.json --data fixtures/synthetic_cohort.csv \
    --out /tmp/ext_smoke --label TTR
  python - <<'PY'
import json
exp = json.load(open("fixtures/expected_metrics.json"))
got = json.load(open("/tmp/ext_smoke/metrics.json"))
keys = ["roc_auc", "average_precision", "sensitivity", "specificity", "ppv", "npv", "f1"]
bad = [k for k in keys if abs(exp[k] - got[k]) > 1e-9]
if bad:
    raise SystemExit(f"SMOKE TEST FAILED: metrics differ from baseline: {bad}")
print("\nSMOKE TEST PASSED: metrics reproduce fixtures/expected_metrics.json exactly.")
PY
  exit 0
fi

CARD="${1:?usage: bash run.sh <card.json> <cohort.csv> [out_dir] [label_col]}"
DATA="${2:?usage: bash run.sh <card.json> <cohort.csv> [out_dir] [label_col]}"
OUT="${3:-external_validation}"
LABEL="${4:-TTR}"

python -m src.evaluate_external --card "$CARD" --data "$DATA" --out "$OUT" --label "$LABEL"
