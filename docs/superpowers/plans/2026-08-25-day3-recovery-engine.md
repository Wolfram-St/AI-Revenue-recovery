# Day 3 Recovery Opportunity Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Recovery Opportunity Engine: rank failed payments by payment-value-aware expected recovery value, authorize actions through a deterministic precedence-based policy engine, and emit an explainable decision trace for every case — without claiming any action-conditional probability.

**Architecture:** New `recovery/` package with three pure modules. `recovery/scoring.py` combines the Day 2 calibrated estimate `P(recovered | context)` with `amount_inr` and deterministic cost/risk constants to produce an `expected_recovery_value_inr` ranking signal. `recovery/policy.py` loads the frozen `config/business_rules.yaml`, evaluates its rule conditions through a small safe parser (no `eval`), and resolves one authorized action per case under documented precedence. `recovery/audit.py` emits a JSON-serializable decision trace per case whose field names mirror `db/schema.sql`. `recovery/engine.py` chains features → probability → score → policy → trace and returns a deterministic summary. The policy engine can always override the AI recommendation.

**Tech Stack:** Python 3.12+, pandas, numpy, scikit-learn (existing calibrated pipeline), PyYAML (new, pinned), pytest, Docker.

**Spec:** Master loop contract (`P(recovered | context)` only), `docs/DAY1.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/DAY2.md`, `config/business_rules.yaml` v1.1, `db/schema.sql`.

## Global Constraints

- The model output remains exactly `P(recovered | context)`. No code, test, metric, or document may claim or imply `P(recovered | context, action)`.
- ERV is a **ranking signal computed from the general recoverability estimate plus deterministic constants**. It is never described as an action-conditional causal estimate and never as realized money.
- Candidate actions normalize to `RETRY_NOW`, `RETRY_LATER`, `REQUEST_UPDATE`, `HUMAN_REVIEW`, `STOP`.
- `config/business_rules.yaml` is treated as a frozen Day 1 artifact: loaded, never mutated. If a gap appears, it is handled in code with a named constant and documented (e.g., the `RETRY_LATER` residual default).
- Rule resolution: highest applicable priority wins; STOP dominates positive actions even against a numerically higher positive priority; ties break deterministically by rule id.
- No LangGraph, API, frontend, database writes, autonomous execution, or multi-agent orchestration.
- Every task follows strict TDD: failing test → minimal implementation → focused pass → related pass → Docker verification → commit.

---

### Task 1: Policy configuration loading and safe condition evaluation

**Files:**
- Create: `recovery/policy.py` (loader + condition parser/evaluator only)
- Modify: `requirements.txt`
- Test: `tests/test_policy_config.py`

**Interfaces:**
- `load_policy_config(path="config/business_rules.yaml") -> PolicyConfig`
- `PolicyConfig.rules` — tuple of parsed rules `(id, name, priority, action, reason, condition_ast)`
- `evaluate_condition(condition_ast, context: dict) -> bool`
- `CANONICAL_ACTIONS: frozenset[str]`

The condition grammar accepted by the parser is exactly what the shipped YAML uses: comparisons (`==`, `!=`, `>=`, `<=`, `>`, `<`) between a column name and a numeric/string/boolean literal, joined by `and`. Anything outside the grammar raises `ValueError` — deliberately, because policy text must never be executed dynamically.

- [ ] **Step 1: Write failing config tests**

```python
def test_loads_shipped_business_rules():
    from recovery.policy import load_policy_config

    policy = load_policy_config("config/business_rules.yaml")
    assert [rule.id for rule in policy.rules] == [f"R{i:03d}" for i in range(1, 9)]
    assert policy.version == "1.1"
    assert policy.stop_precedence is True


def test_condition_evaluates_deterministically():
    from recovery.policy import evaluate_condition, load_policy_config

    policy = load_policy_config("config/business_rules.yaml")
    rule = {r.id: r for r in policy.rules}["R004"]
    assert evaluate_condition(rule.condition_ast, {"attempt_number": 4}) is True
    assert evaluate_condition(rule.condition_ast, {"attempt_number": 3}) is False


def test_unknown_operator_is_rejected_not_executed():
    import pytest
    from recovery.policy import parse_condition

    with pytest.raises(ValueError):
        parse_condition("attempt_number =~ '4'")


def test_rule_with_noncanonical_action_is_rejected(tmp_path):
    # a copied config with action: AUTO_RETRY must fail loudly
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_policy_config.py -q`
Expected: FAIL — `recovery.policy` does not exist.

- [ ] **Step 3: Add PyYAML dependency**

`requirements.txt`: add `pyyaml>=6,<7` (justification: parse the frozen `config/business_rules.yaml`; no other config mechanism exists). Rebuild the Docker image before Docker verification.

- [ ] **Step 4: Implement loader + parser minimally**

Tokenizer + recursive-descent parser producing a nested AST of `(column, op, literal)` and `("and", left, right)` nodes; literal types restricted to int/float/quoted-string/true/false. Loader validates: unique ids, monotonic priorities not required but priorities must be ints, action ∈ canonical vocabulary, version captured. Library code returns objects; nothing prints.

- [ ] **Step 5: Run focused tests and verify pass**

Run: `python -m pytest tests/test_policy_config.py -q`
Expected: PASS.

- [ ] **Step 6: Run related suites and Docker verification, then commit**

```bash
git add recovery/policy.py tests/test_policy_config.py requirements.txt
docker compose run --rm --no-deps app python -m pytest -q   # after image rebuild
git commit -m "feat: load policy rules with safe condition evaluation"
```

---

### Task 2: Deterministic policy decision with precedence

**Files:**
- Modify: `recovery/policy.py`
- Test: `tests/test_policy_engine.py`

**Interfaces:**
- `decide_action(context: dict, policy: PolicyConfig) -> PolicyDecision`
- `PolicyDecision`: `authorized_action`, `matched_rule_id`, `matched_rule_name`, `priority`, `reason`, `is_stop`, `evaluated_rules` (ordered `(id, matched)` tuples)

Precedence law (deterministic, tested):
1. Evaluate all rules against the context.
2. Among matched rules, select highest priority; ties break by lowest rule id.
3. **Stop dominance:** if any matched rule has `action == STOP` and the config declares `stop_precedence: true`, the selected action must be `STOP` — even if a positive rule matched with strictly higher priority.
4. If no rule matches, return the documented residual default `RETRY_LATER` (named constant `RESIDUAL_DEFAULT_ACTION` in code, docstring explains it is a placeholder until the ERV-based NOW-vs-LATER optimizer exists; `matched_rule_id` is `None`, reason states the default was applied).
5. Missing context columns raise `KeyError` with the column name (loud, never silently false).

- [ ] **Step 1: Write failing engine tests**

```python
def test_every_shipped_rule_fires_on_a_crafted_context():
    # one crafted context dict per R001-R008 asserting the exact action
    # R001 opted-out->STOP, R002 fraud->STOP, R003 hard_decline->STOP,
    # R004 attempt_number>=4->STOP, R005 payment_method_issue->REQUEST_UPDATE,
    # R006 amount>25000 and p<0.70->HUMAN_REVIEW, R007 temporary+p>=0.70->RETRY_NOW,
    # R008 p<0.20->STOP


def test_stop_overrides_numerically_higher_positive_priority():
    # custom temp config: positive rule priority 99, STOP rule priority 10, both match
    # decide_action must return STOP


def test_ties_break_by_lowest_rule_id():
    # two matching rules with equal priority -> lowest id wins


def test_residual_default_is_retry_later_when_nothing_matches():
    # authentication_required, p=0.45, attempt_number=1 -> RETRY_LATER, matched_rule_id is None


def test_missing_column_raises_keyerror():
```

- [ ] **Step 2: Verify failure, then implement `decide_action`**

- [ ] **Step 3: Focused pass, then full local suite**

Run: `python -m pytest tests/test_policy_engine.py -q` then `python -m pytest -q`

- [ ] **Step 4: Docker verification and commit**

```bash
git add recovery/policy.py tests/test_policy_engine.py
docker compose run --rm --no-deps app python -m pytest -q
git commit -m "feat: authorize actions through precedence-based policy engine"
```

---

### Task 3: Opportunity scoring and Expected Recovery Value

**Files:**
- Create: `recovery/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- `score_opportunities(df, probabilities) -> pd.DataFrame` indexed like `df`, columns: `recovery_probability`, `expected_recovery_value_inr`, `is_worth_intervention` (bool), `opportunity_rank` (int or null)
- Module constants (documented as illustrative simulation parameters pending product calibration): `RETRY_INTERVENTION_COST_INR = 10.0`, `UNKNOWN_CATEGORY_RISK_FRACTION = 0.05`

Honest semantics (enforced by docstring + tests referencing them):
- `expected_recovery_value_inr = P(recovered | context) * amount_inr − RETRY_INTERVENTION_COST_INR − risk_penalty`, where `risk_penalty = UNKNOWN_CATEGORY_RISK_FRACTION * amount_inr` iff `failure_category == "unknown"` (deterministic uncertainty surcharge).
- The cost basis is the automated-retry cost because both automated actions are retries; REQUEST_UPDATE / HUMAN_REVIEW economics belong to policy, not scoring. Nothing here estimates per-action recovery probabilities.
- Rows with negative ERV are marked `is_worth_intervention=False` and receive no rank; ranks ascend from 1 ordered by ERV descending, ties by `attempt_id` ascending.

- [ ] **Step 1: Write failing scoring tests**

```python
def test_value_dominates_probability_for_prioritization():
    # ₹50,000 @ p=0.55 must outrank ₹500 @ p=0.95


def test_negative_expected_value_is_not_worth_intervention():
    # tiny amount, low probability -> False, rank is null


def test_unknown_category_carries_risk_penalty():
    # identical rows except failure_category unknown vs temporary_decline:
    # unknown ERV == temporary ERV - 0.05 * amount


def test_ranking_is_deterministic_and_dense_from_one():
    # shuffled input frame produces identical ranks; ranks are 1..n over eligible rows


def test_row_count_preserved_and_index_aligned():
```

- [ ] **Step 2: Verify failure, implement, focused pass**

- [ ] **Step 3: Full local suite, Docker verification, commit**

```bash
git add recovery/scoring.py tests/test_scoring.py
docker compose run --rm --no-deps app python -m pytest -q
git commit -m "feat: rank recovery opportunities by expected recovery value"
```

---

### Task 4: Decision trace and audit record

**Files:**
- Create: `recovery/audit.py`
- Test: `tests/test_audit_trace.py`

**Interfaces:**
- `build_decision_trace(row, probability, erv, scoring_recommendation, policy_decision) -> dict`
- `traces_to_json(traces) -> str` (round-trip safe)

Trace fields mirror `db/schema.sql` vocabulary so a later persistence layer maps 1:1 (`recovery_cases.*`, `recovery_actions.authorization_reason`, `audit_logs.event_payload`):
`attempt_id`, `payment_id`, `customer_id`, `event_timestamp`, `amount_inr`, `failure_category`,
`recovery_probability`, `expected_recovery_value_inr`,
`scoring_recommendation` (`AUTOMATED_RETRY` / `NO_INTERVENTION` from ERV sign),
`authorized_action`, `authorization_reason`, `triggered_rule`, `priority`, `evaluated_rules`,
`model_contract` (literal `"P(recovered | context)"`), `probability_is_action_conditional` (literal `False`).

Determinism: the only timestamp in a trace is the row's own `event_timestamp` (data metadata); no wall-clock reads inside library code.

- [ ] **Step 1: Write failing trace tests** — JSON round-trip; demo-narrative completeness (every field the Day 23 demo narrative requires: amount, failure context, probability, ERV, recommendation, policy result, reasons); forbidden post-intervention columns absent (`recovered`, `recovery_time_hours`, `recovery_action`, `action_outcome`, `recovered_amount_inr`).

- [ ] **Step 2: Verify failure, implement, focused pass**

- [ ] **Step 3: Full suite, Docker verification, commit**

```bash
git add recovery/audit.py tests/test_audit_trace.py
docker compose run --rm --no-deps app python -m pytest -q
git commit -m "feat: record explainable policy-gated decision traces"
```

---

### Task 5: Recovery engine integration

**Files:**
- Create: `recovery/engine.py`
- Test: `tests/test_engine_integration.py`

**Interfaces:**
- `run_recovery_engine(df, model, policy_config=None) -> EngineResult`
- `EngineResult`: `traces: list[dict]`, `summary: dict` (`case_count`, `action_counts`, `worth_intervention_count`, `total_expected_recovery_value_inr` over eligible rows, `stop_count`, `human_review_count`)

Pipeline per row: `build_feature_matrix` → `predict_recovery_probability` → `score_opportunities` → `decide_action` → `build_decision_trace`. The engine enforces the core invariant mechanically: **no row whose authorized action is `STOP` carries an automated action, and `scoring_recommendation` never reaches the customer-facing `authorized_action` without the policy gate.**

- [ ] **Step 1: Write failing integration tests**

```python
def test_engine_runs_end_to_end_on_validation_slice():
    # chronological_split(generate_dataset(300, seed=42)); train_baseline(seed=42)
    # result traces count == len(validation); summary totals consistent


def test_engine_output_is_deterministic_for_fixed_seed():
    # two runs produce identical summaries and traces


def test_no_stopped_case_receives_automated_action():
    # for every trace: if authorized_action == STOP -> scoring_recommendation never equals AUTOMATED_RETRY-with-authorized-automated-action


def test_summary_reports_action_distribution_and_total_erv():
    # sum(action_counts.values()) == case_count; total_erv == sum of eligible ERVs rounded to paise
```

- [ ] **Step 2: Verify failure, implement, focused pass, full local suite**

- [ ] **Step 3: Docker verification and commit**

```bash
git add recovery/engine.py tests/test_engine_integration.py
docker compose run --rm --no-deps app python -m pytest -q
git commit -m "feat: integrate recovery opportunity engine end to end"
```

---

### Task 6: Day 3 documentation and verification gate

**Files:**
- Create: `docs/DAY3.md`, `docs/DAY3_RESULTS.md`

- [ ] **Step 1: Run the engine over the canonical held-out test split (seed 42)** and record exact outputs: action distribution, stop/human-review counts, total ERV INR, top-ranked opportunities, and at least two fully expanded example traces (one policy-overrides-AI case if present, one clean automated case).

- [ ] **Step 2: Re-run full suite in Docker; regenerate dataset from seed 42; confirm CSV stays ignored.**

- [ ] **Step 3: Write `DAY3.md` (what/why/assumptions/limitations) and `DAY3_RESULTS.md` (verified numbers, gate decision).**

Gate decision recorded as GO/NO-GO for Day 4 (simulated treatment outcomes) with the unchanged caveat: outcome simulation must be introduced as an explicitly labeled synthetic treatment policy before any realized-revenue claim.

- [ ] **Step 4: Commit**

```bash
git add docs/DAY3.md docs/DAY3_RESULTS.md
git commit -m "docs: close day 3 recovery engine verification"
```

---

## Explicit non-goals (this phase)

LangGraph · API · frontend · dashboard · database writes · autonomous execution · multi-agent orchestration · uplift/bandit modeling · threshold optimization · action-conditional probabilities.

## Boundary statement carried into every artifact

> The Day 2 model estimates general recoverability: `P(recovered | context)`.
> Expected Recovery Value ranks opportunities; it is not `P(recovered | context, action)`
> and not realized revenue. Until action-assigned observations exist, no component
> may claim an action caused a recovery.
