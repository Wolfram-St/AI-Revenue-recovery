# RecoverAI — Day 1 Foundation

## Goal
Finalize the dataset, database schema, and deterministic business rules for the AI Revenue Recovery MVP.

## Scope boundary
Day 1 only. No ML training, recovery scoring, LangGraph workflow, API, frontend, or autonomous payment execution is included yet.

## Dataset
The project uses a synthetic Razorpay-like payment-attempt dataset so no real customer or payment information is required.

Target dataset:
- 5,000 payment-attempt records
- 18 columns
- Customer history represented through aggregate behavioral fields
- Failure reason and payment-method context
- Retry history
- Recovery outcome label
- No target leakage into future ML features

Primary learning target for Day 2:
- `recovered` — whether the failed payment was ultimately recovered

Important rule:
- Features used for prediction must describe information available at decision time. Post-intervention outcome fields must never be used as model features.

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

## Day 1 completion criteria
- [x] Dataset specification finalized
- [x] Schema finalized
- [x] Business rules finalized
- [x] Synthetic-data generation contract documented
- [x] Target-leakage boundary documented

## Day 2 boundary
Day 2 begins with preprocessing, exploratory validation, feature design, and the first recovery-probability model. Do not implement Day 2 work as part of this foundation commit.
