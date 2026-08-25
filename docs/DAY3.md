# Day 3 — Recovery Opportunity Engine

Status: implemented on branch `feature/day3-recovery-engine`.

## What was implemented

1. `recovery/policy.py` — loads the frozen `config/business_rules.yaml` v1.1
   (validated, never mutated), parses rule conditions through a restricted
   AST whitelist parser (no `eval`), and authorizes exactly one action per
   decision-time context under deterministic precedence.
2. `recovery/scoring.py` — pure opportunity-scoring module combining the Day 2
   calibrated estimate `P(recovered | context)` with `amount_inr` and
   deterministic cost/risk constants into an Expected Recovery Value ranking
   signal.
3. `recovery/audit.py` — frozen, JSON-serializable decision trace per case
   whose field names mirror `db/schema.sql`.
4. `recovery/engine.py` — deterministic composition: features → probability →
   scoring → policy → trace, returning per-row traces plus a summary. No
   LangGraph, API, frontend, database writes, autonomous execution, or
   multi-agent orchestration.

## Architecture flow

```
Model → P(recovered | context) → Opportunity scoring → scoring_recommendation → Policy Engine → authorized_action → Decision Trace
```

## Policy architecture

- **Loading** (`load_policy_config`): `yaml.safe_load` of the shipped file;
  validates unique non-empty rule ids, integer priorities, actions within the
  canonical vocabulary (`RETRY_NOW`, `RETRY_LATER`, `REQUEST_UPDATE`,
  `HUMAN_REVIEW`, `STOP` — exact-set match against the config's declared
  vocabulary), boolean `stop_precedence`, and optional boolean `enabled`
  flags. Returns frozen dataclasses; nothing prints; nothing mutates the YAML
  artifact.
- **Authorization** (`decide_action`): evaluates every enabled rule in config
  order and resolves one winner. Disabled rules are never evaluated but still
  appear in `evaluated_rules` as `(id, False)`. The final authorized action is
  re-checked against the canonical vocabulary before return.

### Restricted condition grammar

Conditions are parsed with `ast.parse(..., mode="eval")` — which executes
nothing — and then walked against a strict whitelist:

- The root is a **single comparison or comparisons joined by `and`**.
- Each comparison pits one column name against a literal (int, float, quoted
  string) or another column name, using only `== != >= <= > <`.
- Lowercase `true` / `false` are reserved boolean literals and cannot be used
  as column names.
- Everything else raises `ValueError` loudly, naming the rejected construct:
  calls, attributes, arithmetic, lambdas, comprehensions, imports,
  subscripts, `or`, and chained comparisons are all rejected at parse time —
  never silently evaluated to False. Policy text is never executed
  dynamically.
- A **256-character cap** on condition text is enforced before parsing.
- NaN semantics: a NaN operand makes `==`, `>`, `<`, `>=`, and `<=`
  comparisons evaluate False, while `!=` evaluates True against NaN; within
  an `and`-chain this silently drops the affected clause. Only an entirely
  missing column raises `KeyError`. The shipped rules use no `!=`, so every
  shipped condition treats NaN as non-matching today; validate upstream if
  NaN must be treated as a stop.

### Precedence law

1. Among matched rules, **highest priority wins**.
2. Equal priorities break to the **lowest rule id**.

Both are implemented in one sort key (`-priority, id`) so resolution is
deterministic for identical inputs.

### STOP dominance

Flag-gated by `rule_resolution.stop_precedence: true`: when any matched rule
authorizes STOP, the candidate set narrows to matched STOP rules, so **STOP
wins even against a strictly higher positive priority**. When the flag is
false no narrowing occurs and pure priority applies. Unit-tested with a
priority-99 positive rule vs priority-10 STOP rule.

### Residual default RETRY_LATER

When no rule matches, `RESIDUAL_DEFAULT_ACTION = "RETRY_LATER"` is authorized
(`matched_rule_id=None`, reason states the residual default was applied). The
constant exists because no ERV-based NOW-vs-LATER optimizer exists yet; it
keeps the action vocabulary closed instead of inventing an action. All 445
RETRY_LATER authorizations in the canonical test run were residual defaults.

## Opportunity scoring

- **Pure module**: never calls the model, never invokes policy, performs no
  I/O, reads no wall clock, draws no randomness. Identical inputs produce
  identical outputs.
- **Frozen formula**:

  ```
  expected_recovery_value_inr =
      P(recovered | context) * amount_inr
      - intervention_cost_inr
      - risk_penalty_inr
  ```

  where `intervention_cost_inr` defaults to `RETRY_INTERVENTION_COST_INR`
  and `risk_penalty_inr` equals
  `UNKNOWN_CATEGORY_RISK_FRACTION * amount_inr` iff
  `failure_category == "unknown"`, else `0.0`.
- **Constants documented illustrative**: `RETRY_INTERVENTION_COST_INR = 10.0`
  and `UNKNOWN_CATEGORY_RISK_FRACTION = 0.05` are simulation parameters
  pending product calibration, not measured production costs.
- **Risk penalty**: a deterministic uncertainty surcharge applied only to the
  `unknown` failure category; identical rows differing only in this category
  differ in ERV by exactly `0.05 * amount_inr` (unit-tested).
- **Cost basis**: automated-retry cost, because both automated actions are
  retries. REQUEST_UPDATE / HUMAN_REVIEW economics belong to policy, not
  scoring. Nothing here estimates per-action recovery probabilities.
- **Strict worth rule**: `worth_intervening` compares the *unrounded* ERV
  against zero — zero or negative ERV is never worth intervening, while
  displayed values carry paise rounding (`round(..., 2)`).
- **Dense label-precedence ranks**: ranks number 1..n over eligible rows
  ordered by displayed ERV descending; ties break by index label ascending
  (equal to `attempt_id` ascending when the frame is labeled by attempt_id),
  never by positional order, so shuffling rows does not change ranks.
  Non-worth rows carry `pd.NA` under nullable `Int64`. Sub-paise differences
  that round to equal displayed values fall back to label precedence;
  monotone rounding bounds any distortion at half a paise. Mixed-type index
  labels raise `ValueError` at rank time.

## Decision trace

`recovery/audit.py` emits a frozen `DecisionTrace` with exactly **18 fields**:
`attempt_id`, `payment_id`, `customer_id`, `event_timestamp`, `amount_inr`,
`failure_category`, `recovery_probability`,
`expected_recovery_value_inr`, `scoring_recommendation`, `authorized_action`,
`authorization_reason`, `matched_rule_id`, `matched_rule_name`,
`rule_priority`, `is_stop`, `evaluated_rules`, `model_contract`,
`probability_is_action_conditional`.

- Field names mirror `db/schema.sql` vocabulary so a later persistence layer
  maps 1:1 onto `recovery_cases.*`,
  `recovery_actions.authorization_reason`, and `audit_logs.event_payload`.
- Constructor-enforced invariants: `model_contract` must be exactly
  `"P(recovered | context)"`; `probability_is_action_conditional` must be
  exactly `False`; `is_stop` must equal `(authorized_action == "STOP")`;
  forbidden post-intervention columns (`recovered`, `recovery_time_hours`,
  `recovery_action`, `action_outcome`, `recovered_amount_inr`) appear nowhere.
- **Byte-deterministic JSON**: `traces_to_json` serializes with fixed field
  order; identical inputs yield byte-identical output across processes.
- **No wall clock**: the only timestamp is the row's own `event_timestamp`,
  normalized once to canonical ISO-8601 UTC from row metadata.

## AI recommends vs policy authorizes

The two concepts stay separate end-to-end and are never collapsed:

- `scoring_recommendation` is `INTERVENE` / `NO_INTERVENTION` from the ERV
  sign — a recommendation only.
- `authorized_action` comes exclusively from the policy engine — except for
  the candidate rule below.

**Candidate rule** (`recovery/engine.py`): the policy engine is consulted
ONLY for rows with `worth_intervening=True`; the calibrated probability is
injected into the context so rules R006–R008 can reference it. Rows that are
not worth intervening are terminal no-ops: `authorized_action="STOP"`,
`is_stop=True`, `matched_rule_id=None`, `evaluated_rules=()`, and a reason
quoting the exact non-positive ERV — the engine does not manufacture a
positive authorization when there is no economic case. Two mechanical
invariants hold: (a) a negative or zero ERV can never become an automated
positive action; (b) every authorized action comes either from a real policy
decision or this documented terminal. In the canonical test run the
recommendation and authorization diverged on 192 cases (INTERVENE recommended,
STOP authorized) — divergence is legitimate and recorded verbatim.

## Model contract and causal boundary

The model estimates general recoverability: `P(recovered | context)`.
The current system does not estimate P(recovered | context, action).
Every trace carries that contract literally in `model_contract` and asserts
`probability_is_action_conditional=False`.

Expected Recovery Value is a prioritization score, not an incremental causal revenue estimate. It combines a general recoverability estimate with
deterministic constants; it is not realized revenue and supports no claim
that any action caused a recovery.

Action-specific probabilities remain unavailable because the dataset records
whether a payment eventually recovered — not which intervention caused it.
There are no action-assigned observations to learn from; uplift/bandit
modeling stays a non-goal, and simulated treatment outcomes are deferred to
Day 4 (see gate below).

## Known limitations

- Scoring constants (₹10 cost, 0.05 unknown-risk fraction) are illustrative
  and uncalibrated; with ₹10 against simulator-scale amounts, virtually every
  row clears strict positivity (canonical test run: 750/750 candidates,
  `noop_count=0`), so the worth gate currently filters almost nothing.
- RETRY_LATER remains a residual placeholder; NOW-vs-LATER optimization does
  not exist yet (all 445 test-split RETRY_LATERs were residual defaults).
- NaN-valued policy columns make equality/ordering comparisons evaluate
  False (`!=` would evaluate True, but no shipped rule uses it).
- All numbers measure the synthetic simulator, not any production system; no
  production performance is claimed anywhere.
- Ranking ties below one paise fall back to label precedence (bounded at half
  a paise of ordering distortion).

## Test evidence

Day 3 suites (focused counts, real): `tests/test_policy_config.py` 44 ·
`tests/test_policy_engine.py` 31 · `tests/test_scoring.py` 75 ·
`tests/test_audit_trace.py` 55 · `tests/test_engine_integration.py` 20 —
225 total; plus 38 pre-existing dataset/Day 1/Day 2 tests = **263 passed**
locally and identically inside Docker (tails in DAY3_RESULTS.md).

## Docker evidence

`docker compose run --rm --no-deps app python -m pytest -q` → `263 passed in
10.35s`. Dataset regenerated from seed 42 inside the container reproduces
5,000 × 19; `data/payment_attempts.csv` stays git-ignored.

## Deferred work

LangGraph workflow composition · API/frontend/dashboard · database writes ·
NOW-vs-LATER ERV optimizer · product calibration of scoring constants ·
persistence layer mapping traces onto schema tables · Day 4 simulated
treatment outcomes (explicitly labeled synthetic treatment policy first).

## What comes next

Day 4: simulated treatment outcomes behind the labeled-synthetic boundary,
enabling the first (still synthetic) action-comparison analysis without ever
retrofitting causal meaning onto `P(recovered | context)`.
