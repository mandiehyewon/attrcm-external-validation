# External Validation — ATTR-CM Sex-Specific Model (UCSF)

External validation of a logistic-regression model for **transthyretin cardiac
amyloidosis (ATTR-CM)** in heart-failure patients, trained on the MGB cohort and
applied to a UCSF cohort.

This branch is **validation-only**. The model is trained and frozen on the MGB
side and shipped as a small, self-contained **model card** (JSON). On the UCSF
side there is **no training and no data splitting** — the entire UCSF cohort is
scored as one test set.

- **Label:** `TTR` (any TTR amyloidosis; positive = 1)
- **Cohorts:** `all`, `male`, `female` (a separate card per cohort)

> **IRB / PHI:** the input cohort CSV is patient data governed by an IRB
> protocol. It is never committed to this repository. Keep it under `hf_data/`
> (git-ignored) or another local path.

---

## 1. Setup

```bash
pip install -r requirements.txt
```

Python 3.10. Dependencies: numpy, pandas, scikit-learn, xgboost. Scoring a
logistic-regression card needs only numpy/pandas (the JSON is self-contained, no
sklearn version matching); `xgboost` is needed only to score `model_type:
"xgboost"` cards.

## 2. What you receive

- **Model cards** under `models/`, produced on the MGB side — one per cohort for
  each model: `model_card_lr_<cohort>.json` (logistic regression) and
  `model_card_xgb_<cohort>.json` + `model_card_xgb_<cohort>.ubj` (XGBoost).
- The evaluator, `src/evaluate_external.py`.

## 3. Provide the UCSF cohort CSV

One row per patient, with the columns below. Place it anywhere local (e.g.
`hf_data/ucsf_cohort.csv`).

**Required**

| Column | Meaning |
|--------|---------|
| `TTR` | Outcome label, 0/1 (needed for metrics; omit for scoring-only) |
| `Gender_Legal_Sex` | `Male` / `Female` — only to pick which cohort card to apply |

**Features** (supply what you have; any listed feature absent from your CSV is
scored with the card's frozen MGB fallback and reported in `coverage.json`):

Continuous — `Age_at_T0, eGFR, NT_proBNP, BNP, Troponin, Prealbumin, Sodium,
Potassium, Chloride, ejection_fraction, LVIDed_mm, IVS_mm, PWT_mm,
relative_wall_thickness` (plus `Weight, Height, BMI, Calcium, Creatinine, Glucose`
if available).

Binary 0/1 — `Race_White, Race_Black, Race_Asian, Race_Other, Race_Unknown,
HTN, CAD, AF, VT, OrthoHypo, AS, AVReplace, CTS, PolyNeuro, Neuropathy, SPST,
BTR, KneeReplace, HipReplace, Proteinuria, Fatigue, Dyslipidemia, Hypothyroidism,
Diabetes, COPD, NAC_Stage_1, NAC_Stage_2, NAC_Stage_3`.

Notes:
- Units must match MGB: `NT_proBNP`/`Troponin` are raw (the card applies `log1p`
  internally); other continuous features are raw values.
- Do **not** normalize or impute — the card carries the frozen MGB normalization
  and imputation. Missing cells are handled automatically.
- Missingness indicators (`{col}_missing`) are derived by the evaluator from each
  source column's missing pattern; you don't supply them.

## 4. Run the validation

Pick a card that matches the cohort you are scoring. Two equivalent ways:

**Shell**
```bash
bash run.sh models/model_card_lr_female.json hf_data/ucsf_cohort.csv external_female
```

**Notebook** — `jupyter notebook notebook/run_validation.ipynb` (edit the paths in
the first cell, run top to bottom).

**Directly**
```bash
python -m src.evaluate_external \
    --card  models/model_card_lr_female.json \
    --data  hf_data/ucsf_cohort.csv \
    --out   external_female \
    --label TTR                       # omit --label for scoring-only
```

### Outputs (under `--out`)

| File | Contents |
|------|----------|
| `predictions.csv` | per-patient `risk_score` + `predicted_at_threshold` (id column carried through if present) |
| `metrics.json` | AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1 at the card's frozen threshold, plus n / prevalence |
| `coverage.json` | which card features were present vs scored with the frozen fallback |

### Check the tooling on synthetic data

A synthetic 4000-patient cohort with the exact MGB column schema ships under
`fixtures/`. The smoke test scores it with `models/model_card_lr_all.json` and
confirms the metrics reproduce the committed baseline `fixtures/expected_metrics.json`
**exactly** — a deterministic check that your environment reproduces our scoring
(the metric *values* are meaningless on synthetic labels; only their exact
reproduction matters):

```bash
bash run.sh --smoke
```

You can also run the card/pipeline equivalence check on its own:

```bash
python -m src.model_card_selftest    # card math == fitted pipeline (max |Δp| ~1e-16)
```

---

## 5. Model-card format

A card is a single JSON file. The whole raw→probability chain (MGB split-level
z-normalization, the pipeline's imputer/scaler, and the logistic-regression
coefficients) is collapsed into one linear model in **raw feature units**:

```
p = sigmoid( intercept + Σ_i  weight_i · transform_i(x_i) )
predicted positive  if  p ≥ decision_threshold
```

`transform_i` is `log1p` for `NT_proBNP`/`Troponin` and identity otherwise; a
missing `x_i` is replaced by `impute_raw` (the frozen MGB median for continuous
features, 0 for binary) before transforming.

```json
{
  "schema_version": "1.0",
  "model_type": "logistic_regression",
  "task": {"label": "TTR", "cohort": "female"},
  "predict": "p = 1/(1+exp(-(intercept + sum_i weight_i * transform_i(x_i)))); positive if p >= decision_threshold",
  "intercept": -4.812,
  "features": [
    {"name": "NT_proBNP", "kind": "continuous", "transform": "log1p", "impute_raw": 2450.0, "weight": 0.4127},
    {"name": "Age_at_T0", "kind": "continuous", "transform": "identity", "impute_raw": 74.0, "weight": 0.0331},
    {"name": "HTN", "kind": "binary", "transform": "identity", "impute_raw": 0.0, "weight": 0.211},
    {"name": "Prealbumin_missing", "kind": "missing_indicator", "source": "Prealbumin", "transform": "identity", "impute_raw": 0.0, "weight": 0.642}
  ],
  "decision_threshold": 0.0173,
  "threshold_objective": "sensitivity",
  "metadata": {"trained_on": "MGB HF cohort", "n_train": 15042, "sklearn_version": "1.5.0", "created_utc": "..."}
}
```

See `models/model_card_lr_female.json` for a full example.

---

## 6. Generating cards (MGB side)

Cards are produced from the fitted training pipeline with
`src.model_card.build_model_card`, which collapses both normalization layers
+ the LR coefficients and **self-verifies** the result against the fitted pipeline
before writing. In the training code, right after the pipeline is fit and the
decision threshold chosen:

```python
from src.model_card import build_model_card, verify_card
import json, pandas as pd

card = build_model_card(
    best_estimator, feature_cols, cohort_split_dir,   # dir with train_original.csv
    label="TTR", cohort="female",
    decision_threshold=best_threshold, threshold_objective="sensitivity",
    metadata={"trained_on": "MGB HF cohort", "n_train": len(y_train)},
)
# Fails loudly if the collapsed card does not reproduce the pipeline on raw test data:
verify_card(card, pd.read_csv(cohort_split_dir / "test_original.csv"), test_probs, tol=1e-6)
json.dump(card, open("models/model_card_lr_female.json", "w"), indent=2)
```

Then commit the resulting `models/model_card_*.json` to this branch and hand it to
UCSF. No raw data or trained binary is shared — only the coefficients in the card.

---

## 7. Repository layout

```
run.sh                          validation driver (card + CSV -> metrics)
requirements.txt                dependencies
models/                         model cards: model_card_{lr,xgb}_{all,male,female}.json (+ .ubj)
fixtures/                       synthetic_cohort.csv + expected_metrics.json (smoke test)
notebook/run_validation.ipynb   interactive validation (same steps as run.sh)
src/
  evaluate_external.py          UCSF-side CLI: score + metrics + coverage
  model_card.py                 card format: build (MGB) + predict (UCSF)
  model_card_selftest.py        equivalence proof (card == fitted pipeline)
  metrics.py                    AUROC/AUPRC/sensitivity/... helpers
```

## XGBoost model cards

Cards now come in two flavours, distinguished by `card["model_type"]`.
`src/evaluate_external.py` dispatches automatically:

| `model_type` | Scorer | Extra files | Dependencies |
|--------------|--------|-------------|--------------|
| `logistic_regression` | `predict_from_card` | none — the JSON is self-contained | numpy, pandas |
| `xgboost` | `predict_xgb_from_card` | `model_card_xgb_<cohort>.ubj` beside the JSON | numpy, pandas, **xgboost** |

XGBoost is non-linear, so the normalization layers cannot be folded into the
estimator the way they are for logistic regression. Those cards keep the transform
explicit under `preprocessing` — `feature_order`, per-column `continuous`
(`log1p`, `impute_raw`, `z_mean`/`z_std` for layer 1, `scaler_mean`/`scaler_scale`
for layer 2 when present), `passthrough_impute`, `missing_indicators` — and ship
the native booster as UBJSON, which is portable and version-tolerant.

Scoring an xgboost card needs the optional `xgboost` dependency (commented out in
`requirements.txt`):

```bash
pip install "xgboost==2.0.3"
python -m src.evaluate_external \
    --card models/model_card_xgb_female.json \
    --data hf_data/ucsf_cohort.csv \
    --out  hf_data/external_xgb_female/ \
    --label TTR
```

Keep the `.ubj` next to its `.json`; the booster is resolved relative to the card.
