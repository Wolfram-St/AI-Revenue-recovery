# Day 3 Results — Recovery Opportunity Engine

All numbers below are fresh outputs from the canonical run: dataset
regenerated from seed 42, chronological 70/15/15 split, baseline retrained
with seed 42, sigmoid calibration fit on validation only, engine executed
over the held-out test split.

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
| Calibration | sigmoid `CalibratedClassifierCV` over frozen pipeline, fit on validation (test labels untouched) |
| Seed | 42 |
| Train positive-class rate | 0.4983 |

## Engine output (held-out latest 15%, n = 750)

Test-split base recovery rate: **0.4613** (matches Day 2).

| Metric | Value |
| --- | --- |
| case_count | 750 |
| candidate_count (worth intervening → policy consulted) | 750 |
| noop_count (terminal no-op STOP, policy not consulted) | 0 |
| HUMAN_REVIEW | 1 |
| REQUEST_UPDATE | 106 |
| RETRY_LATER (all residual default) | 445 |
| RETRY_NOW | 6 |
| STOP | 192 |
| stop_count | 192 |
| human_review_count | 1 |
| total_candidate_erv_inr | ₹1,038,149.21 |
| Divergent traces (scoring INTERVENE vs authorized STOP) | 192 |
| Residual-default RETRY_LATER cases among candidates | 445 |

Every test row cleared strict ERV positivity against the illustrative ₹10
cost, so all 750 rows reached the policy engine as candidates and the
terminal no-op path never fired on this split. All 445 RETRY_LATER
authorizations were residual defaults (`matched_rule_id=None`) because the
shipped rules authorize no RETRY_LATER directly; all 192 STOPs carried an
INTERVENE scoring recommendation that policy overrode.

## Top-5 ranked opportunities (test split)

| Rank | attempt_id | amount_inr | recovery_probability | expected_recovery_value_inr | authorized_action |
| --- | --- | --- | --- | --- | --- |
| 1 | ATT-004419 | 29,377.00 | 0.430329 | ₹11,162.92 | HUMAN_REVIEW (R006) |
| 2 | ATT-004531 | 20,931.06 | 0.471477 | ₹9,858.52 | REQUEST_UPDATE (R005) |
| 3 | ATT-004533 | 20,531.61 | 0.456338 | ₹9,359.36 | RETRY_LATER (residual) |
| 4 | ATT-004449 | 13,018.62 | 0.669353 | ₹8,704.05 | RETRY_LATER (residual) |
| 5 | ATT-004271 | 12,396.87 | 0.636017 | ₹7,874.61 | RETRY_LATER (residual) |

Value dominates probability exactly as designed: rank 1 pairs the largest
amount with a below-average probability, outranking smaller high-probability
rows.

## Example decision traces

Three fully expanded traces from the canonical run (verbatim engine output).
The first is the required policy-overrides-AI divergence case — present in
the test split itself, so no validation-slice fallback was needed.

### Trace A — policy overrides AI (INTERVENE → STOP via R003 hard_decline)

```json
{
  "attempt_id": "ATT-004252",
  "payment_id": "PAY-004252",
  "customer_id": "CUS-0386",
  "event_timestamp": "2026-02-14T06:45:00+00:00",
  "amount_inr": 5826.92,
  "failure_category": "hard_decline",
  "recovery_probability": 0.34314196604790814,
  "expected_recovery_value_inr": 1989.46,
  "scoring_recommendation": "INTERVENE",
  "authorized_action": "STOP",
  "authorization_reason": "Hard declines should not be repeatedly retried automatically.",
  "matched_rule_id": "R003",
  "matched_rule_name": "hard_decline",
  "rule_priority": 90,
  "is_stop": true,
  "evaluated_rules": [
    ["R001", false],
    ["R002", false],
    ["R003", true],
    ["R004", false],
    ["R005", false],
    ["R006", false],
    ["R007", false],
    ["R008", false]
  ],
  "model_contract": "P(recovered | context)",
  "probability_is_action_conditional": false
}
```

### Trace B — clean automated candidate (RETRY_NOW via R007)

```json
{
  "attempt_id": "ATT-004395",
  "payment_id": "PAY-004395",
  "customer_id": "CUS-0509",
  "event_timestamp": "2026-02-15T18:30:00+00:00",
  "amount_inr": 1193.01,
  "failure_category": "temporary_decline",
  "recovery_probability": 0.7025779436067491,
  "expected_recovery_value_inr": 828.18,
  "scoring_recommendation": "INTERVENE",
  "authorized_action": "RETRY_NOW",
  "authorization_reason": "Temporary failure with sufficiently high predicted recovery probability.",
  "matched_rule_id": "R007",
  "matched_rule_name": "retry_now_eligible",
  "rule_priority": 50,
  "is_stop": false,
  "evaluated_rules": [
    ["R001", false],
    ["R002", false],
    ["R003", false],
    ["R004", false],
    ["R005", false],
    ["R006", false],
    ["R007", true],
    ["R008", false]
  ],
  "model_contract": "P(recovered | context)",
  "probability_is_action_conditional": false
}
```

### Trace C — residual-default RETRY_LATER candidate (no rule matched)

```json
{
  "attempt_id": "ATT-004251",
  "payment_id": "PAY-004251",
  "customer_id": "CUS-0419",
  "event_timestamp": "2026-02-14T06:30:00+00:00",
  "amount_inr": 3760.93,
  "failure_category": "authentication_required",
  "recovery_probability": 0.49246219820187537,
  "expected_recovery_value_inr": 1842.12,
  "scoring_recommendation": "INTERVENE",
  "authorized_action": "RETRY_LATER",
  "authorization_reason": "Residual default RETRY_LATER applied because no policy rule matched this case.",
  "matched_rule_id": null,
  "matched_rule_name": null,
  "rule_priority": null,
  "is_stop": false,
  "evaluated_rules": [
    ["R001", false],
    ["R002", false],
    ["R003", false],
    ["R004", false],
    ["R005", false],
    ["R006", false],
    ["R007", false],
    ["R008", false]
  ],
  "model_contract": "P(recovered | context)",
  "probability_is_action_conditional": false
}
```

In every trace `scoring_recommendation` and `authorized_action` remain
separate fields, the model contract reads literally
`"P(recovered | context)"`, and `probability_is_action_conditional` is
exactly `false`.

## Score sanity check

- Moderate discrimination carries through from Day 2: ROC-AUC ≈ 0.64 matches
  the documented irreducible noise in the synthetic outcome generator
  (σ = 0.75 Gaussian term in the logit). The top-5 opportunities carry
  moderate probabilities (0.43–0.67), not suspiciously near-certainties.
- ERV ranking behaves as designed under that noise: amount dominates — the
  highest-probability rows do not crowd out high-value rows (rank 1:
  ₹29,377 at p = 0.430).
- All 750 rows passed the strict positivity worth gate because simulator
  amounts dwarf the illustrative ₹10 cost; this is a property of the
  uncalibrated constant, not evidence of engine quality, and is recorded as
  such in DAY3.md limitations.
- These metrics measure the SYNTHETIC simulator, not production. No Razorpay
  or any other production performance is claimed or implied anywhere in this
  document. Expected Recovery Value is a prioritization score, not an
  incremental causal revenue estimate, and total_candidate_erv_inr is not
  forecast revenue.

## Verification evidence

Focused counts per suite file (fresh runs):

| Suite file | Tests |
| --- | --- |
| tests/test_policy_config.py | 44 |
| tests/test_policy_engine.py | 31 |
| tests/test_scoring.py | 75 |
| tests/test_audit_trace.py | 55 |
| tests/test_engine_integration.py | 20 |
| Day 3 subtotal | 225 |
| Pre-existing (generator 5, validation 11, temporal split 6, features 4, training 4, evaluation 8) | 38 |
| **Total** | **263** |

Full-suite tails:

- Local: `.venv\Scripts\python -m pytest -q` → `263 passed in 6.67s`
- Docker: `docker compose run --rm --no-deps app python -m pytest -q` →
  `263 passed in 10.35s` (wall time varies by environment)

Dataset regeneration check: seed 42 reproduces 5,000 × 19 inside the
container; `data/payment_attempts.csv` remains git-ignored (git status shows
only the two new Day 3 docs).

## Day 3 gate: **GO**

| Gate check | Evidence | Result |
| --- | --- | --- |
| Gate 8: action-aware modeling stays blocked / boundary held | every trace enforces `model_contract = "P(recovered \| context)"` and `probability_is_action_conditional = False`; 55 audit-trace tests pass; uplift/bandit modeling untouched | PASS (boundary held) |
| Gate 9: policy engine implemented & tested | restricted AST grammar, precedence + tie-break, STOP dominance, residual RETRY_LATER covered by 44 config + 31 engine tests; full suite green locally and in Docker | PASS |
| Gate 10 groundwork: workflow composed | engine chains features → probability → scoring → policy → trace deterministically (20 integration tests incl. fixed-seed determinism); LangGraph composition explicitly deferred to later phases | PASS (groundwork) |

**GO for Day 4 simulated treatment outcomes**, with the unchanged caveat:
outcome simulation must be introduced as an explicitly labeled synthetic
treatment policy before any realized-revenue claim. The boundary is
unchanged: the model estimates `P(recovered | context)` only, Expected
Recovery Value ranks but does not cause, and no component may claim an
action caused a recovery until action-assigned observations exist.
