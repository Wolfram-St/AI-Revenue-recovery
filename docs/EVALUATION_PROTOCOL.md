# RecoverAI Evaluation Protocol

This protocol is frozen before Day 2 model training.

## 1. Dataset split

Use the synthetic `event_timestamp` for a strict chronological split:

- Train: earliest 70%
- Validation: next 15%
- Test: latest 15%

Never randomly mix future observations into training.

`event_timestamp` is evaluation metadata, not a predictive feature by default.

## 2. Model diagnostics

Report:

- ROC-AUC
- PR-AUC
- Brier score or equivalent probability-calibration metric
- Calibration curve

Accuracy alone is not a sufficient model metric.

## 3. Business metrics

The primary product metric is recovered revenue, not classification accuracy.

Required reporting:

- Revenue at risk (INR)
- Recovered revenue (INR)
- Recovery rate
- Recovered revenue / revenue at risk
- Number of interventions
- Intervention success rate
- False-positive intervention count
- Expected recovered revenue vs actual recovered revenue

## 4. Decision metric

For a candidate action A, the eventual optimizer should estimate:

`ExpectedRecoveryValue(A) = P(recovery | context, A) * amount_inr - intervention_cost - risk_penalty`

The optimizer must not interpret an overall `P(recovery | context)` as a causal estimate for every action.

## 5. Action-aware modeling boundary

The Day 2 baseline may predict general recoverability using `recovered` as the label.

Before implementing the intervention optimizer, the dataset must contain action-aware observations or an explicitly simulated treatment policy. This prevents unsupported claims such as `P(recovery | retry_later)` from a dataset that only records whether recovery eventually happened.

## 6. Policy constraints

Any business metric must be calculated after applying deterministic policy constraints. A model recommendation that violates an exclusion rule is not an eligible recovery action.

## 7. Synthetic-data validity

The synthetic outcome generator must:

- contain stochastic noise
- create overlapping positive and negative examples
- include feature interactions
- avoid making the label a direct copy of policy rules

A very high score is not automatically evidence of a strong model. If performance is suspiciously high, inspect leakage and simulator dependence before presenting it.
