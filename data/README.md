# Day 1 Dataset Contract

The canonical Day 1 dataset is a synthetic payment-attempt dataset for development and evaluation.

## Contract

- Target rows: 5,000
- Columns: 18
- Currency: INR
- Data source: synthetic only
- Intended use: ML experimentation and end-to-end recovery simulation

## Required feature groups

### Payment context
- payment/attempt identifiers
- amount
- payment method
- timestamp
- attempt number

### Failure context
- failure code
- failure category
- issuer response
- hard-decline indicator
- fraud-risk indicator

### Customer history
- tenure
- historical successful payments
- historical failed payments
- historical recoveries
- opt-out state

### Environment
- device type
- country

### Outcome
- recovery indicator
- recovery latency

## Leakage rule
Outcome fields describe what happened after an intervention and cannot be used as features when training a model that makes the original recovery decision.

For Day 2, the training pipeline must create a decision-time feature set first, then attach the outcome label separately.

## Failure categories

The initial taxonomy is intentionally small:

- `temporary_decline`
- `hard_decline`
- `payment_method_issue`
- `authentication_required`
- `unknown`

This taxonomy can be extended only when the data demonstrates a real need.
