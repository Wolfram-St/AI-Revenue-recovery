# Day 1 / Day 1.5 Dataset Contract

The canonical dataset is a synthetic payment-attempt dataset for development, evaluation, and end-to-end recovery simulation.

## Contract

- Target rows: 5,000
- Columns: 19
- Currency: INR
- Data source: synthetic only
- `event_timestamp`: ordered synthetic metadata used for temporal evaluation
- Predictive features: decision-time information only
- Label: `recovered`
- Post-intervention outcome fields are not predictive features

## Required feature groups

### Payment context
- payment/attempt identifiers
- amount
- payment method
- attempt number

### Customer history
- tenure
- historical successful payments
- historical failed payments
- historical recoveries
- opt-out state

Historical aggregates are point-in-time snapshots. They must exclude events after the current failed attempt.

### Failure context
- failure code
- failure category
- issuer response
- fraud-risk indicator

### Environment
- device type
- country

### Evaluation metadata
- ordered `event_timestamp`

`event_timestamp` is used to create chronological train/validation/test splits and is not automatically used as a model feature.

### Outcome
- recovery indicator (`recovered`)
- recovery latency is an outcome-only field reserved for later evaluation

## Leakage rule
Outcome fields describe what happened after an intervention and cannot be used as features when training the original recovery decision.

For Day 2, the training pipeline must construct the decision-time feature matrix first, then attach the recovery label separately.

## Synthetic-data realism rule
The outcome generator intentionally uses noisy latent probabilities and feature interactions. Policy rules must not be copied directly into the label-generation logic.

## Failure categories

The initial taxonomy is intentionally small:

- `temporary_decline`
- `hard_decline`
- `payment_method_issue`
- `authentication_required`
- `unknown`

This taxonomy can be extended only when the data demonstrates a real need.
