# RecoverAI — Day 1 Foundation

## Goal
Finalize and harden the dataset, database schema, deterministic business rules, and evaluation contract for the AI Revenue Recovery MVP.

## Scope boundary
Day 1 + Day 1.5 foundation only. No ML training, recovery scoring, LangGraph workflow, API, frontend, optimizer, or autonomous payment execution is included yet.

## Dataset
The project uses a synthetic Razorpay-like payment-attempt dataset so no real customer or payment information is required.

Hardened dataset contract:
- 5,000 payment-attempt records
- 19 columns, including ordered `event_timestamp` metadata
- 17 decision-time predictive features
- 1 recovery label: `recovered`
- 1 temporal-evaluation metadata field: `event_timestamp`
- Post-intervention outcome fields are kept outside the predictive dataset
- Customer history fields are defined as point-in-time snapshots
- Synthetic outcomes contain stochastic noise and are not direct copies of policy rules

Primary Day 2 baseline target:
- `recovered` — whether the failed payment was ultimately recovered

Important rule:
- Features used for prediction must describe information available at decision time. Post-intervention outcomes must never be used as model features.

## Database schema
Core tables:

1. `customers`
2. `payments`
3. `payment_attempts`
4. `recovery_cases`
5. `recovery_actions`
6. `recovery_outcomes`
7. `audit_logs`

The separation is intentional:

`payment event -> recovery case -> proposed/authorized action -> outcome -> audit record`

## Financial safety boundary
The architecture uses this rule throughout the project:

> AI recommends. Policy engine authorizes.

The eventual ML/agent layer may rank recovery opportunities and recommend interventions, but deterministic policy rules remain the final authorization gate for any recovery action.

## Action-aware modeling boundary
The baseline model predicts general recoverability. The eventual intervention optimizer must not claim action-specific causal probabilities from the baseline label alone.

Before optimizing between `RETRY_NOW`, `RETRY_LATER`, `REQUEST_UPDATE`, and other actions, the dataset must contain action-aware observations or a clearly documented simulated treatment policy.

## Evaluation boundary
Evaluation is frozen in `docs/EVALUATION_PROTOCOL.md`:
- chronological 70/15/15 split using `event_timestamp`
- ROC-AUC and PR-AUC diagnostics
- probability calibration
- recovered INR as the primary business metric
- policy-constrained intervention metrics

## Day 1.5 hardening completed
- [x] Point-in-time feature semantics
- [x] Ordered event timestamp for temporal evaluation
- [x] Normalized action vocabulary
- [x] Explicit rule precedence
- [x] Action-aware optimizer boundary
- [x] Stochastic synthetic outcome mechanism
- [x] Frozen evaluation protocol
- [x] Automated dataset contract tests

## Day 2 boundary
Day 2 begins with data validation, preprocessing, feature engineering, and the first recovery-probability model. Do not implement intervention optimization or agent orchestration until the baseline model and evaluation pass the agreed gates.
