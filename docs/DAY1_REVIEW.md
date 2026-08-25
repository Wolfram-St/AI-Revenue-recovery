# Day 1.5 Foundation Review

## Gate result

**GO for Day 2 foundation work, with one explicit boundary:** Day 2 may begin with dataset validation, preprocessing, feature engineering, and the baseline recovery-probability model. Intervention optimization and action-specific causal claims remain blocked until action-aware outcome data is available.

## Repairs completed

1. Added ordered `event_timestamp` metadata for chronological evaluation.
2. Defined point-in-time semantics for customer history aggregates.
3. Normalized recovery actions to `RETRY_NOW`, `RETRY_LATER`, `REQUEST_UPDATE`, `HUMAN_REVIEW`, and `STOP`.
4. Defined deterministic rule resolution: highest priority wins and STOP has precedence.
5. Separated predictive features, label, metadata, and post-intervention outcomes in the dataset schema.
6. Reworked synthetic recovery generation to include stochastic noise and interactions rather than copying policy rules.
7. Reserved action-aware fields for the future intervention optimizer.
8. Froze the evaluation protocol around chronological splits, calibration, and recovered INR.
9. Added automated dataset contract tests.

## Remaining intentional limitations

### Action-aware learning
The baseline label `recovered` answers whether recovery occurred, not which intervention caused it. The optimizer must therefore wait for action-aware observations or a clearly documented simulated treatment policy.

### Synthetic data
The dataset is suitable for demonstrating architecture and model workflow, not for claiming production-level model performance. Any unusually high metric must trigger leakage and simulator-dependence checks.

### Razorpay integration
The current dataset is provider-inspired and synthetic. A later provider adapter should translate Razorpay events into the canonical RecoverAI schema rather than coupling the model directly to provider-specific payloads.

## Day 2 entry criteria

Day 2 can proceed only with these rules intact:

- chronological train/validation/test split
- no post-intervention feature leakage
- no future customer-history leakage
- model metrics reported alongside recovered INR
- no action-specific causal claims from the baseline model
