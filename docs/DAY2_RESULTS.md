# Day 2 Results — Baseline Recovery Model

All numbers below are fresh outputs from the canonical run described here.

## Run configuration

| Item | Value |
| --- | --- |
| Dataset seed | 42 |
| Rows | 5,000 (contract-exact) |
| Columns | 19 (contract-exact) |
| Split | chronological 70/15/15 |
| Split sizes | train 3,500 / validation 750 / test 750 |
| Model | XGBoost (`n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9`) |
| Preprocessing | numeric passthrough + one-hot (`handle_unknown="ignore"`), fitted on train only |
| Seed | 42 |
| Train positive-class rate | 0.4983 |

## Predictive metrics (held-out latest 15%, n = 750)

| Metric | Uncalibrated | Calibrated (sigmoid, fit on validation) |
| --- | --- | --- |
| ROC-AUC | 0.6445 | 0.6445 (ranking unchanged by design) |
| PR-AUC | 0.5796 | 0.5796 |
| Brier score | 0.2440 | **0.2331** |

Calibration improved the Brier score by ~4.5% relative while preserving
ranking. The test-period base recovery rate is 0.4613.

## Score sanity check (protocol §7)

ROC-AUC ≈ 0.64 is moderate, not suspiciously high. This matches the known
irreducible noise in the synthetic outcome generator (σ = 0.75 Gaussian term
in the logit). No leakage indicators: features are decision-time only, split
is chronological, preprocessing fitted exclusively on training rows.

## Revenue metrics — labeled THRESHOLDED SIMULATION (threshold = 0.5)

These are **not** causal incremental-recovery estimates. The dataset records
whether a payment eventually recovered, not which intervention caused it.

| Metric | Uncalibrated probabilities | Calibrated probabilities |
| --- | --- | --- |
| Revenue at risk (test) | ₹2,274,852.48 | ₹2,274,852.48 |
| Interventions selected | 393 | 350 |
| Predicted recoverable revenue | ₹716,037.44 | ₹549,625.49 |
| Actual recovered among selected | ₹544,653.25 | ₹494,771.90 |
| Recovery rate among selected | 0.5598 | 0.5771 |
| Recovered share of revenue at risk | 0.2394 | 0.2175 |
| False-positive interventions | 173 | 148 |
| Missed recoverable cases | 126 | 144 |

Interpretation: selecting cases with p ≥ 0.5 concentrates on payments that
recovered at 56–58% versus the 46% period base rate. Calibrated
probabilities produce fewer, cleaner interventions (higher precision,
lower recall).

## Verification evidence

- Full suite: 38 passed locally and in Docker (`docker compose run --rm app python -m pytest -q`).
- Feature-contract test confirms forbidden columns absent from the model input.
- Dataset regeneration check recorded in the Day 2 gate report.

## Day 2 gate: **GO**

| Gate check | Evidence | Result |
| --- | --- | --- |
| Dataset valid | regenerated from seed 42 inside container: 5,000 rows × 19 columns, `valid: True` | PASS |
| Chronological split | 3,500 / 750 / 750 with strict time-ordering tests | PASS |
| Feature pipeline leakage-safe | container run confirms forbidden columns absent; preprocessing fitted on train only | PASS |
| Baseline trains reproducibly | same-seed predictions identical (unit-tested); full suite green in Docker | PASS |
| Calibration acceptable | Brier 0.2440 → 0.2331 on held-out test | PASS |
| Business metrics calculated | thresholded-simulation revenue table above, labeled non-causal | PASS |

**GO for intervention-aware work**: policy engine, opportunity scoring by
expected recovered value, and controlled workflow may proceed. Action-specific
probabilities still require action-assigned observations or a documented
simulated treatment policy — that boundary is unchanged.
