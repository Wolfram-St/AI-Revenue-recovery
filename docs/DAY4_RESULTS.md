# Day 4 Results — Simulated Treatment Outcomes

All numbers below are fresh outputs from the canonical run: dataset
regenerated from seed 42, chronological 70/15/15 split, baseline retrained
with seed 42, sigmoid calibration fit on validation only, calibrated
probabilities predicted over the FULL 5,000-row frame, treatment assignment /
outcomes / timeline generated under `master_seed 20260826`.

## Run configuration

| Item | Value |
| --- | --- |
| Dataset seed | 42 |
| Rows | 5,000 (contract-exact) × 19 columns |
| Split | chronological 70/15/15 (train 3,500 / validation 750 / test 750) |
| Splits used for the model | train (fit) + validation (calibration); test labels untouched; predictions requested over the full 5,000-row frame |
| Model | XGBoost (`n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9`) |
| Preprocessing | numeric passthrough + one-hot (`handle_unknown="ignore"`), fitted on train only |
| Calibration | sigmoid `CalibratedClassifierCV` over frozen pipeline, fit on validation |
| Probability range fed to stage 1 | min 0.237922 / max 0.722157 / mean 0.479985 over 5,000 rows |
| Policy version | `1.0` (`config/treatment_policy.yaml`, loaded validated) |
| Master seed | 20260826 |
| Seed streams | `Generator.spawn` children in fixed order: 0 assignment · 1 outcomes · 2 temporal |
| Train positive-class rate | 0.4983 |

## Engine-of-record output (canonical run)

Stage-1 safety gate: STOP 1,188 rows forced to safety-censored CONTROL —
R001 opt-out 120 · R002 fraud 86 · R003 hard_decline 857 · R004 attempt-count
125 — leaving 3,812 eligible rows to the randomized draw.

### OBSERVED SIMULATED OUTCOME — arms (n = 5,000)

| Arm | count | randomized | safety_censored | recovery_rate | recovered_amount_inr_total |
| --- | --- | --- | --- | --- | --- |
| CONTROL | 1,931 | 743 | 1,188 | 0.368721 (randomized-only: 0.585464) | ₹2,250,916.71 |
| RETRY_NOW | 1,132 | 1,132 | 0 | 0.718198 | ₹2,431,755.14 |
| RETRY_LATER | 973 | 973 | 0 | 0.630010 | ₹1,909,702.03 |
| REQUEST_UPDATE | 581 | 581 | 0 | 0.685026 | ₹1,399,049.37 |
| HUMAN_REVIEW | 383 | 383 | 0 | 0.650131 | ₹866,887.40 |

Total simulated recovered revenue: **₹8,858,310.65**.

CONTROL's aggregate rate deliberately includes safety-censored rows
(descriptive); every difference below uses the randomized-only baseline.

### OBSERVED SIMULATED OUTCOME — naive differences vs randomized CONTROL

These are **not causal estimates**: eligibility selection routes
context-dependent rows into safety-censored CONTROL, so compared arms draw
from different strata; treated-minus-control numbers are adjusted-for-nothing
naive differences on simulated outcomes. All four treated comparisons had ≥ 30
cases on both sides (`count_caveat` null everywhere).

| Treated arm | recovery_rate_difference | revenue_per_case_difference_inr |
| --- | --- | --- |
| RETRY_NOW | +0.1327 | ₹268.19 |
| RETRY_LATER | +0.0445 | ₹82.69 |
| REQUEST_UPDATE | +0.0996 | ₹527.99 |
| HUMAN_REVIEW | +0.0647 | ₹383.41 |

Control-vs-best-arm comparison: randomized CONTROL rate 0.585464 vs best
treated arm RETRY_NOW rate 0.718198 → difference **+0.1327** on OBSERVED
SIMULATED OUTCOMES, non-causal per the note above.

Honest ordering check at finite n: observed aggregate rates order RETRY_NOW >
REQUEST_UPDATE > HUMAN_REVIEW > RETRY_LATER. Ground-truth main effects order
RETRY_NOW (+0.60) > REQUEST_UPDATE (+0.45) > RETRY_LATER (+0.35) >
HUMAN_REVIEW (+0.25). The first two match; RETRY_LATER falls below
HUMAN_REVIEW in this run — consistent with its designed fatigue interaction
(−0.25 logit for `attempt_number >= 3`) plus sampling noise at n = 973, and
reported plainly rather than smoothed over.

### Overlap diagnostics (OBSERVED SIMULATED OUTCOME)

eligible_count = 3,812 · safety_censored_count = 1,188.
Assignment-probability ranges are constant per configured arm (randomized
rows): CONTROL [0.20, 0.20] · RETRY_NOW [0.30, 0.30] · RETRY_LATER
[0.25, 0.25] · REQUEST_UPDATE [0.15, 0.15] · HUMAN_REVIEW [0.10, 0.10];
safety-censored CONTROL rows exactly [0.00, 0.00] (no stage-2 draw).
Propensity-under-assignment ranges cover every row of each arm:

| Arm | propensity_under_assignment [min, max] |
| --- | --- |
| CONTROL | [0.064042, 0.820066] |
| RETRY_NOW | [0.336145, 0.931203] |
| RETRY_LATER | [0.238400, 0.898626] |
| REQUEST_UPDATE | [0.346334, 0.899257] |
| HUMAN_REVIEW | [0.327557, 0.842119] |

Ranges overlap heavily across arms within the eligible stratum; positivity
holds there by construction (every eligible row kept positive probability for
every arm). Full-population estimands are not supported because eligibility
was context-dependent.

### Invariant checks

| Check | Result |
| --- | --- |
| Temporal-ordering violations | **0** (treated `failure < treatment < outcome`; controls `failure < outcome` with null treatment_timestamp; 0 non-null control timestamps) |
| Per-arm observed delays | RETRY_NOW 0.25 h · REQUEST_UPDATE 2.0 h · HUMAN_REVIEW 4.0 h · RETRY_LATER 24.0 h (min == max == config for every arm) |
| Resolution-window hours | min 1.019433 / max 47.970612 (inside configured [1, 48]) |
| Amount-bounds violations | **0** of 5,000 (unrecovered ⇒ 0.00 exactly; recovered ⇒ round(amount, 2) exactly; no negatives; no payout exceeds settled amount) |
| Dataset validator | `valid: true`, 5,000 × 12, zero violations, `classification_complete: true` |
| Leakage guard | Day 2 feature builder applied to the joined frame selects exactly its whitelist columns; zero of the new treatment/outcome/ground-truth fields present; original `recovered` label untouched by the simulator |
| Determinism rerun | SHA-256 digests identical for assignment frame, outcome frame, and timeline columns across a fresh full rerun |

## Example records (verbatim canonical-run JSON)

Randomized RETRY_NOW treated row:

```json
{
  "attempt_id": "ATT-000001",
  "payment_id": "PAY-000001",
  "customer_id": "CUS-0090",
  "context": {
    "event_timestamp": "2026-01-01 00:00:00+00:00",
    "amount_inr": 2995.85,
    "failure_category": "temporary_decline",
    "failure_code": "T002",
    "payment_method": "upi",
    "device_type": "android",
    "country": "IN",
    "attempt_number": 2,
    "successful_payment_count": 1,
    "failed_payment_count": 1,
    "historical_recovery_count": 0,
    "customer_opted_out": 0,
    "fraud_risk": 0,
    "model_recovery_probability": 0.4228084865369152
  },
  "assignment": {
    "assigned_action": "RETRY_NOW",
    "arm_source": "randomized",
    "assignment_probability": 0.3
  },
  "ground_truth": {
    "base_recovery_propensity": 0.617209,
    "action_effect_logit": 1.0,
    "propensity_under_assignment": 0.814228
  },
  "timing": {
    "treatment_timestamp": "2026-01-01 00:15:00+00:00",
    "outcome_timestamp": "2026-01-01 09:56:53.639817457+00:00"
  },
  "outcome": {
    "simulated_recovered": 1,
    "simulated_recovered_amount_inr": 2995.85
  }
}
```

Safety-censored CONTROL row (R001 opt-out ⇒ forced CONTROL before any random
draw; `treatment_timestamp` is the documented null):

```json
{
  "attempt_id": "ATT-000004",
  "payment_id": "PAY-000004",
  "customer_id": "CUS-0439",
  "context": {
    "event_timestamp": "2026-01-01 00:45:00+00:00",
    "amount_inr": 620.91,
    "failure_category": "temporary_decline",
    "failure_code": "T001",
    "payment_method": "card",
    "device_type": "web",
    "country": "IN",
    "attempt_number": 1,
    "successful_payment_count": 5,
    "failed_payment_count": 2,
    "historical_recovery_count": 1,
    "customer_opted_out": 1,
    "fraud_risk": 0,
    "model_recovery_probability": 0.6035313217037251
  },
  "assignment": {
    "assigned_action": "CONTROL",
    "arm_source": "safety_censored",
    "assignment_probability": 0.0
  },
  "ground_truth": {
    "base_recovery_propensity": 0.712753,
    "action_effect_logit": 0.0,
    "propensity_under_assignment": 0.712753
  },
  "timing": {
    "treatment_timestamp": null,
    "outcome_timestamp": "2026-01-01 23:32:53.027990332+00:00"
  },
  "outcome": {
    "simulated_recovered": 0,
    "simulated_recovered_amount_inr": 0.0
  }
}
```

In both records the ground-truth block shows the world the simulator built
(base propensity, effect under the assigned arm, pre-noise propensity), not
anything measured from a real provider.

## SIMULATED GROUND TRUTH table

Known by construction from `config/treatment_policy.yaml` v1.0
(`master_seed 20260826`):

| Parameter | Value |
| --- | --- |
| noise_sigma_logit | 0.5 |
| arm_probabilities | CONTROL 0.20 · RETRY_NOW 0.30 · RETRY_LATER 0.25 · REQUEST_UPDATE 0.15 · HUMAN_REVIEW 0.10 |
| main_effects_logit | CONTROL 0.0 · RETRY_NOW +0.6 · RETRY_LATER +0.35 · REQUEST_UPDATE +0.45 · HUMAN_REVIEW +0.25 |
| interaction 1 | RETRY_NOW where `failure_category == temporary_decline` → effect_logit +0.4 |
| interaction 2 | RETRY_LATER where `attempt_number >= 3` → effect_logit −0.25 |
| base intercept | −0.35 |
| category_effects | temporary_decline +0.95 · payment_method_issue +0.25 · authentication_required +0.05 · unknown −0.15 · hard_decline −1.45 |
| other base terms | successful_payment_count_log1p +0.11 · historical_recovery_count_min5 +0.16 · attempt_number_prior_offset −0.28 · fraud_risk −0.22 · amount_log1p_per_k −0.10 · method_upi +0.12 · device_android +0.10 |

Outcome rule: `simulated_recovered ~ Bernoulli(sigmoid(base_logit +
effect_logit + Normal(0, 0.5)))`; revenue is full-or-zero.

## Verification evidence

Focused counts per suite file (fresh runs):

| Suite file | Tests |
| --- | --- |
| tests/test_treatment_config.py | 48 |
| tests/test_treatment_assignment.py | 29 |
| tests/test_outcomes.py | 32 |
| tests/test_treatment_dataset.py | 43 |
| tests/test_temporal_treatment.py | 26 |
| tests/test_reporting.py | 21 |
| Day 4 subtotal | 199 |
| Pre-existing (Day 1/2/3 suites) | 263 |
| **Total** | **462** |

Full-suite tails (fresh runs, both environments):

- Local: `.venv\Scripts\python -m pytest -q` → `462 passed in 12.79s`
- Docker: `docker compose run --rm --no-deps app python -m pytest -q` →
  `462 passed in 17.08s` (wall time varies by environment)

Artifact check: `data/treatment_outcomes.csv` was NOT created — the module
APIs return frames only; the `python -m simulation.generate` CLI wrapper that
writes the CSV lands with Day 5 or as immediate follow-up work. Git status
shows only the three new Day 4 documentation files.

## Day 4 gate: **GO**

| Gate check | Evidence | Result |
| --- | --- | --- |
| Canonical pipeline executed with real captured outputs | 5,000-row run end-to-end (model → probabilities → assignment → outcomes → timeline → dataset → validator → reporting); all tables above verbatim | PASS |
| Two-stage assignment with labeled CONTROL sources | 3,812 randomized / 1,188 safety_censored; STOP causes itemized (R001–R004); safety-censored `assignment_probability = 0.0` enforced by validator | PASS |
| Temporal semantics correct | 0 ordering violations; delays exact per config; control-null rule holds; windows inside [1, 48] h | PASS |
| Revenue bounds hold | 0 amount-bounds violations; total ₹8,858,310.65 full-or-zero | PASS |
| Ground truth vs observed label discipline | SIMULATED GROUND TRUTH / OBSERVED SIMULATED OUTCOME labels embedded in all reporting output; no causal estimators anywhere | PASS |
| Baseline boundary held | leakage guard: Day 2 feature matrix contains zero new fields even on the merged frame; `P(recovered \| context)` contract untouched | PASS |
| Determinism | byte-identical rerun (SHA-256 digests) under master_seed 20260826, spawn children 0/1/2 | PASS |
| Full test suite green | 199 new + 263 pre-existing = 462 passed locally AND in Docker | PASS |
| Documentation complete | DAY4.md covers policy, confounding (verbatim D1), overlap/positivity, limitations (in-sample caveat, horizon asymmetry), reproducibility; three mandated sentences verbatim | PASS |
| No forbidden artifacts | `data/treatment_outcomes.csv` absent (CLI deferred to Day 5/follow-up); git tree clean apart from docs | PASS |

**GO for Day 5 action-aware modeling**, inside the unchanged boundary: the
observations are synthetic `(context, assigned_action, simulated_recovered)`
triplets usable for future action-aware predictive models; nothing in Day 4
estimates real-world causal effects, the Day 2 baseline remains
`P(recovered | context)`, and any Day 5 model must stay labeled against the
stored SIMULATED GROUND TRUTH rather than retrofitting causal meaning onto
observed differences.
