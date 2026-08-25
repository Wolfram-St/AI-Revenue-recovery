# Day 2 Baseline Recovery Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate the first reproducible recovery-probability model using only decision-time information from the hardened 19-column dataset.

**Architecture:** Validate the synthetic dataset, construct a strict decision-time feature matrix, split events chronologically 70/15/15, train an XGBoost baseline, calibrate its probabilities, and evaluate both predictive quality and recovered-revenue impact. The model predicts general recoverability only; it does not choose an intervention and does not make action-specific causal claims.

**Tech Stack:** Python 3.12+, pandas, numpy, scikit-learn, XGBoost, pytest, Docker.

**Spec:** Day 2 design approved in chat and the Day 1.5 contracts in `docs/DAY1.md`, `data/schema.json`, and `docs/EVALUATION_PROTOCOL.md`.

## Global Constraints

- Use only decision-time predictive features from the canonical 19-column dataset.
- `event_timestamp` is metadata for chronological splitting, not a predictive feature unless explicitly justified later.
- `recovered` is the only Day 2 baseline label.
- `recovery_time_hours` and reserved action-aware fields are excluded from baseline training.
- Use chronological 70/15/15 train/validation/test splitting.
- Report ROC-AUC, PR-AUC, and probability calibration.
- Report recovered INR and policy-constrained business metrics where the baseline can support them without claiming causal action effects.
- Do not implement LangGraph, an intervention optimizer, API, frontend, or autonomous payment execution in Day 2.
- Preserve reproducibility with explicit random seeds.

---

### Task 1: Validate the canonical dataset

**Files:**
- Create: `data/validate_dataset.py`
- Test: `tests/test_data_validation.py`

**Interfaces:**
- Consumes: `data.generate_dataset.generate_dataset`, `data.schema.json` contract.
- Produces: `validate_dataset(df) -> dict[str, object]` returning row count, column count, missing-value summary, duplicate identifier counts, class balance, timestamp monotonicity, and contract violations.

- [ ] **Step 1: Write failing validation tests**

```python
def test_validation_accepts_canonical_dataset():
    from data.generate_dataset import generate_dataset
    from data.validate_dataset import validate_dataset

    report = validate_dataset(generate_dataset(500, seed=42))
    assert report["valid"] is True
    assert report["column_count"] == 19
    assert report["missing_cells"] == 0
```

```python
def test_validation_rejects_duplicate_attempt_ids():
    from data.generate_dataset import generate_dataset
    from data.validate_dataset import validate_dataset

    df = generate_dataset(50, seed=42)
    df.loc[1, "attempt_id"] = df.loc[0, "attempt_id"]
    report = validate_dataset(df)
    assert report["valid"] is False
    assert report["duplicate_attempt_ids"] > 0
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_data_validation.py -q`
Expected: FAIL because `validate_dataset` does not exist yet.

- [ ] **Step 3: Implement the minimal validator**

Implement checks for the exact 19-column contract, null cells, duplicate `attempt_id`/`payment_id`, binary `recovered`, allowed failure categories, and nondecreasing UTC event timestamps. Return a structured report rather than printing from library code.

- [ ] **Step 4: Run the focused tests and verify pass**

Run: `python -m pytest tests/test_data_validation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/validate_dataset.py tests/test_data_validation.py
git commit -m "feat: validate day 2 recovery dataset"
```

---

### Task 2: Build the chronological split and decision-time feature matrix

**Files:**
- Create: `data/splits.py`
- Create: `ml/features.py`
- Test: `tests/test_temporal_split.py`
- Test: `tests/test_features.py`

**Interfaces:**
- `chronological_split(df, train_fraction=0.70, validation_fraction=0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- `build_feature_matrix(df) -> tuple[pd.DataFrame, pd.Series]`

The split must sort by `event_timestamp`, preserve temporal order, and never randomly shuffle observations. The feature builder must exclude identifiers, `event_timestamp`, `recovered`, `recovery_time_hours`, and any reserved action-aware columns.

- [ ] **Step 1: Write failing split and leakage tests**

```python
def test_chronological_split_preserves_time_order():
    from data.generate_dataset import generate_dataset
    from data.splits import chronological_split

    train, validation, test = chronological_split(generate_dataset(100, seed=1))
    assert train["event_timestamp"].max() < validation["event_timestamp"].min()
    assert validation["event_timestamp"].max() < test["event_timestamp"].min()
```

```python
def test_feature_matrix_excludes_outcome_and_metadata_columns():
    from data.generate_dataset import generate_dataset
    from ml.features import build_feature_matrix

    X, y = build_feature_matrix(generate_dataset(100, seed=1))
    forbidden = {"event_timestamp", "recovered", "recovery_time_hours", "recovery_action", "action_outcome", "recovered_amount_inr"}
    assert forbidden.isdisjoint(X.columns)
    assert len(X) == len(y) == 100
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_temporal_split.py tests/test_features.py -q`
Expected: FAIL because the split and feature-builder modules do not exist yet.

- [ ] **Step 3: Implement minimal chronological splitting**

Sort a copy by `event_timestamp`, compute integer boundaries from the requested fractions, and return disjoint train/validation/test frames without shuffling.

- [ ] **Step 4: Implement the feature matrix**

Separate categorical and numeric columns explicitly. Keep `recovered` as the label. Do not perform target encoding, aggregate future history, or fit preprocessing on validation/test data in this task.

- [ ] **Step 5: Run focused tests and verify pass**

Run: `python -m pytest tests/test_temporal_split.py tests/test_features.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add data/splits.py ml/features.py tests/test_temporal_split.py tests/test_features.py
git commit -m "feat: add temporal split and decision-time features"
```

---

### Task 3: Train the baseline recovery model

**Files:**
- Create: `ml/train.py`
- Modify: `requirements.txt`
- Test: `tests/test_training.py`

**Interfaces:**
- `train_baseline(train_df, validation_df, seed=42) -> tuple[object, dict[str, object]]`
- `predict_recovery_probability(model, df) -> np.ndarray`

Use XGBoost for the baseline. Fit preprocessing only on the training data. Handle categorical columns through an explicit preprocessing pipeline and produce probability estimates in `[0, 1]`. Save a reproducible model artifact only after tests pass.

- [ ] **Step 1: Write failing training tests**

```python
def test_baseline_returns_probabilities_in_range():
    from data.generate_dataset import generate_dataset
    from data.splits import chronological_split
    from ml.train import train_baseline, predict_recovery_probability

    train, validation, _ = chronological_split(generate_dataset(300, seed=42))
    model, metadata = train_baseline(train, validation, seed=42)
    probabilities = predict_recovery_probability(model, validation)
    assert len(probabilities) == len(validation)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert metadata["model"] == "xgboost"
```

- [ ] **Step 2: Run focused test and verify failure**

Run: `python -m pytest tests/test_training.py -q`
Expected: FAIL because the training module does not exist yet.

- [ ] **Step 3: Add the minimum XGBoost/scikit-learn dependencies**

Update `requirements.txt` with pinned major/minor-compatible dependencies already supported by the Docker environment, including `xgboost` and `scikit-learn`. Do not add a framework that is not required by this baseline.

- [ ] **Step 4: Implement the training pipeline**

Use a `ColumnTransformer` for numeric/categorical preprocessing and an XGBoost classifier with a fixed seed. Fit only on training data. Return model metadata including seed, feature names, train/validation row counts, and positive-class rate.

- [ ] **Step 5: Run focused test and verify pass**

Run: `python -m pytest tests/test_training.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/train.py tests/test_training.py requirements.txt
 git commit -m "feat: add baseline recovery model"
```

---

### Task 4: Calibrate probabilities and evaluate the baseline

**Files:**
- Create: `ml/evaluate.py`
- Create: `docs/DAY2.md`
- Create: `docs/DAY2_RESULTS.md`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- `evaluate_predictions(y_true, probabilities, amounts) -> dict[str, float]`
- `calculate_revenue_metrics(y_true, probabilities, amounts, threshold=0.5) -> dict[str, float]`
- `calibrate_model(model, validation_df) -> object`

Report ROC-AUC, PR-AUC, Brier score, calibration data, revenue at risk, recovered revenue under a clearly labeled thresholded simulation, and intervention counts. Do not describe the thresholded simulation as causal incremental recovery because the baseline dataset does not yet encode action assignment.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_evaluation_returns_required_metrics():
    import numpy as np
    from ml.evaluate import evaluate_predictions

    result = evaluate_predictions(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.8, 0.7, 0.2]),
        np.array([100., 200., 300., 400.]),
    )
    assert {"roc_auc", "pr_auc", "brier_score"}.issubset(result)
```

```python
def test_revenue_metrics_are_nonnegative():
    import numpy as np
    from ml.evaluate import calculate_revenue_metrics

    result = calculate_revenue_metrics(
        np.array([0, 1]), np.array([0.2, 0.9]), np.array([100., 1000.]), threshold=0.5
    )
    assert result["revenue_at_risk_inr"] >= 0
    assert result["predicted_recoverable_revenue_inr"] >= 0
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_evaluation.py -q`
Expected: FAIL because evaluation functions do not exist yet.

- [ ] **Step 3: Implement evaluation metrics**

Use scikit-learn metrics. Calculate business metrics from test predictions and actual labels without treating predictions as recovered money. Keep predicted recoverable revenue and actual recovered revenue as separate fields.

- [ ] **Step 4: Add validation-based probability calibration**

Use a scikit-learn calibration method fitted without touching the test labels. Compare uncalibrated and calibrated Brier scores on the held-out test set.

- [ ] **Step 5: Run focused tests and verify pass**

Run: `python -m pytest tests/test_evaluation.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full Day 2 test suite**

Run: `python -m pytest -q`
Expected: all existing Day 1/Day 1.5 tests plus Day 2 tests pass.

- [ ] **Step 7: Generate the Day 2 results report**

Record the exact dataset seed, split sizes, model configuration, metrics, calibration comparison, and revenue metrics. If the model score is suspiciously high, inspect leakage and simulator dependence before presenting it.

- [ ] **Step 8: Commit**

```bash
git add ml/evaluate.py tests/test_evaluation.py docs/DAY2.md docs/DAY2_RESULTS.md
git commit -m "feat: evaluate and calibrate recovery baseline"
```

---

### Task 5: Final Day 2 verification gate

**Files:**
- Modify: `docs/DAY2.md`
- Modify: `docs/DAY2_RESULTS.md`

**Interfaces:**
- Consumes: all Day 2 modules and tests.
- Produces: a documented GO/NO-GO decision for downstream intervention-aware modeling.

- [ ] **Step 1: Re-run the complete test suite in Docker**

Run: `docker compose run --rm <test-service> python -m pytest -q` using the repository's actual compose service name.
Expected: zero failures.

- [ ] **Step 2: Re-generate the dataset from the fixed seed**

Run: `python data/generate_dataset.py` inside the project container and validate that the generated contract remains 5,000 rows and 19 columns.

- [ ] **Step 3: Confirm the model never consumes forbidden columns**

Run the feature-contract test and inspect the final feature list. The test must confirm that `event_timestamp`, `recovered`, `recovery_time_hours`, and reserved action-aware columns are absent from `X`.

- [ ] **Step 4: Record the Day 2 gate**

Set the report to `GO` only if validation, tests, temporal split, model training, calibration, and evaluation all pass. Otherwise record `NO-GO` with the exact failed gate and do not start intervention optimization.

- [ ] **Step 5: Commit final verification documentation**

```bash
git add docs/DAY2.md docs/DAY2_RESULTS.md
git commit -m "docs: close day 2 baseline verification"
```
