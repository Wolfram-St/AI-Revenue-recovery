# Day 7 — Bounded Recovery Portfolio Optimizer: Design Specification

**Status:** Design only. No implementation has started. Awaiting review and approval.

---

## 1. Status and Purpose

This document is the architectural design specification for Day 7 of the RecoverAI project, named **Bounded Recovery Portfolio Optimizer**.

It records the verified repository state, defines the problem, specifies the chosen architecture and all material design decisions, documents constraints the optimizer must enforce, justifies the solver strategy, defines the policy interaction contract, establishes leakage boundaries, specifies evaluation methodology, and proposes GO/NO-GO gates.

This is **not** an implementation plan. No source files, test files, or dependency changes are created by this document. Implementation tasks will be defined in a subsequent planning artifact, conditional on review approval.

---

## 2. Context and Evidence Entering Day 7

### 2.1 Repository State Verified

Branch: `feature/day7-optimizer`
Base commit: `06e6b4d` (Merge PR \#5 `feature/day6-decision-engine` into `main`)
Branch divergence from `origin/main`: zero commits ahead (clean base).
Working tree: clean, no staged or unstaged changes.
`git merge-base HEAD origin/main` returns `06e6b4df678e2ddc5a8ff54a5272b32504697f35`.

### 2.2 Days 1-6 Summary

| Day | Deliverable | Key Contract |
|-----|-------------|--------------|
| 1 | Synthetic dataset + schema | 5,000 rows x 19 cols; `recovered` label; post-intervention fields in `FORBIDDEN_FEATURES` |
| 2 | Baseline model | `P(recovered | context)` only; NOT action-conditional; Brier 0.2331; ROC-AUC 0.6445 on test |
| 3 | Recovery Opportunity Engine | `Model -> ERV scoring -> Policy -> authorized_action -> DecisionTrace`; AI recommends; policy authorizes; STOP dominant |
| 4 | Simulated Treatment Outcomes | Explicit synthetic treatment policy; master_seed 20260826; 5 arms; two-stage hybrid assignment; CONTROL = no-intervention reference |
| 5 | Action-Aware Models | One calibrated XGBoost per arm on randomized stratum; `P(recovered | context, action=a)`; target `simulated_recovered` |
| 6 | Decision Engine Evidence | OPTIMIZER_JUSTIFIED (match rate 0.765, relative regret 0.138, policy safety passed, 3 disjoint CI pairs); per-arm preferred over pooled; verdict advisory and synthetic-world-only |

### 2.3 Confirmed Contracts from Repository Inspection

**Action vocabulary** (`recovery/policy.py`, `config/business_rules.yaml`):
```
RETRY_NOW, RETRY_LATER, REQUEST_UPDATE, HUMAN_REVIEW, STOP
```
STOP is dominant when `stop_precedence: true` (currently `true`).

**ARM_ORDER** (`ml/action_model.py`):
```python
ARM_ORDER = ("CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")
```
TREATED_ARMS = `("RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")`.

**Decision-time feature whitelist** (`ml/features.py`):
- Numeric: `amount_inr`, `attempt_number`, `customer_tenure_days`, `successful_payment_count`, `failed_payment_count`, `historical_recovery_count`, `customer_opted_out`, `fraud_risk`
- Categorical: `payment_method`, `failure_code`, `failure_category`, `issuer_response`, `device_type`, `country`
- FORBIDDEN: identifiers, `event_timestamp`, `recovered`, `recovery_time_hours`, `recovery_action`, `action_outcome`, `recovered_amount_inr`

**Incremental revenue formula** (`ml/incremental.py`, `ml/decision_policy.py`):
```
IncrementalRevenue(i, a) =
    (P_hat_a(i) - P_hat_CONTROL(i)) * amount_inr(i)
    - RETRY_INTERVENTION_COST_INR
    - risk_penalty(i)
```
where `risk_penalty(i) = UNKNOWN_CATEGORY_RISK_FRACTION * amount_inr(i)` iff `failure_category == "unknown"` else 0.0.

Constants (from `recovery/scoring.py`):
- `RETRY_INTERVENTION_COST_INR = 10.0` (illustrative, uncalibrated)
- `UNKNOWN_CATEGORY_RISK_FRACTION = 0.05` (illustrative, uncalibrated)

**Probability injection choice** (Day 6, `ml/decision_policy.py`): `BoundedRecommender` injects the top arm's model probability as `recovery_probability` into `decide_action`. This choice is preserved in Day 7.

**Existing audit trace** (`recovery/audit.py`, `DecisionTrace`): 18 frozen fields. Day 7 introduces a parallel portfolio-level trace structure and does NOT replace or modify `DecisionTrace`.

**Day 6 limitations carried forward:**
- HUMAN_REVIEW has approximately 52 randomized test rows; bootstrap AUC CI half-width approximately 0.17.
- Relative regret margin is thin: 0.1383 against threshold 0.15; per-seed sd approximately 0.010.

**`BoundedRecommender` (Day 6)**: operates per-row, row-independently. Day 7 extends this to portfolio-level allocation.

---

## 3. Problem Statement

The Day 6 `BoundedRecommender` selects the best action independently per row: for each payment attempt it picks the treated arm with the highest incremental revenue estimate and routes it through `decide_action`. This is greedy and row-independent.

Day 7 addresses a strictly more general problem: **portfolio-level allocation across a batch of recovery opportunities subject to explicit resource constraints**.

When intervention resources are limited, independent per-row selection may produce an infeasible or suboptimal portfolio:
- If every row independently prefers HUMAN_REVIEW, the staffing capacity is exceeded.
- If total interventions exceed budget, lower-value interventions consume budget that could serve higher-value opportunities.
- A per-row greedy approach cannot reallocate budget from low-value rows to high-value rows.

The portfolio optimizer solves this jointly across all rows before any action is submitted to the policy engine.

---

## 4. Scope

Day 7 builds:

1. **Portfolio optimizer module** (`ml/portfolio_optimizer.py`)
2. **Greedy baseline** (`ml/portfolio_greedy.py`)
3. **Portfolio audit** (`ml/portfolio_audit.py`)
4. **Portfolio evaluation** (`ml/portfolio_evaluation.py`)
5. **Tests** (`tests/test_portfolio_optimizer.py`, `tests/test_portfolio_greedy.py`, `tests/test_portfolio_audit.py`, `tests/test_portfolio_evaluation.py`)
6. **Documentation** (`docs/DAY7.md`, `docs/DAY7_RESULTS.md`)

Day 7 does NOT build a CLI, API, frontend, database writer, or production deployment artifact.

---

## 5. Non-Goals

Explicitly out of scope:

- Uplift models, incremental causal estimators, contextual bandits, reinforcement learning, online learning
- LangGraph workflow composition
- HTTP API or frontend
- Database writes or persistence
- Real-world or production deployment
- Multi-step sequential allocation
- Per-segment fairness constraints beyond those documented
- Action quotas other than the HUMAN_REVIEW capacity bound
- Any claim of real-world revenue gains
- Razorpay or provider integration
- Autonomous payment execution
- Model weight optimization (action-value models consumed as-is)
- Pooled model as primary (per-arm models remain preferred per Day 6 Gate A)

---

## 6. Existing Contracts and Invariants

The following are binding and non-negotiable in Day 7:

**I-1. AI recommends. Policy authorizes.**
The optimizer is a recommendation layer. Every allocated action must pass through `decide_action` before it becomes an authorized action. The optimizer cannot return an authorized action directly.

**I-2. STOP dominance.**
When `stop_precedence: true`, any matching STOP rule overrides any positive recommendation. The optimizer cannot circumvent this.

**I-3. Separate semantic fields.**
`optimizer_recommendation` and `authorized_action` must be stored as distinct fields. They must never be silently collapsed.

**I-4. Decision-time inputs only.**
The optimizer may only consume features, probabilities, costs, and risk signals available at decision time. Realized outcomes, future timestamps, and simulator ground-truth fields are forbidden.

**I-5. Model contract.**
The optimization signal is `P(recovered | context, action=a)` from Day 5 per-arm models. The Day 2 baseline is NOT the optimization signal.

**I-6. Synthetic-world-only.**
All incremental revenue estimates are MODEL ESTIMATE quantities in the synthetic world. No real-world revenue claim is made.

**I-7. Determinism.**
Identical inputs and configuration produce byte-identical allocation output. No wall-clock reads, no randomness in the optimizer core.

---

## 7. Chosen Architecture

```
Candidate frame (decision-time context + identifiers)
        |
        v
Action Value Computation
(IncrementalValue per (row, treated_arm) via per-arm bundle)
        |
        v
Eligibility Gate
(policy pre-screen: unconditional-STOP rows removed from allocation)
        |
        v
Portfolio Optimizer
(constraint-aware ranked greedy allocation over eligible rows)
        | optimizer_recommendation per row
        v
Policy Engine (decide_action per row)
        | authorized_action per row
        v
Portfolio Audit Trace
(optimizer_recommendation + authorized_action + per-row details)
        |
        v
Portfolio Summary
(totals, constraint utilization, divergence counts)
```

**Unchanged from Days 1-6:**
- `recovery.policy.decide_action` remains the sole authorization boundary
- `config/business_rules.yaml` is not modified
- `ml/action_model.py`, `ml/incremental.py`, `ml/decision_policy.py`, `recovery/*` are not modified
- All production modules are read-only in Day 7

---

## 8. End-to-End Data Flow

### 8.1 Input

| Input | Type | Source |
|-------|------|--------|
| `candidate_frame` | `pd.DataFrame` | Decision-time context rows (Day 2 feature whitelist + identifiers) |
| `action_bundle` | `ActionModelBundle` | Calibrated per-arm bundle from Day 5 |
| `policy_config` | `PolicyConfig` | Loaded from `config/business_rules.yaml` |
| `optimizer_config` | `OptimizerConfig` (frozen dataclass) | Caller-supplied: `budget_limit`, `human_review_capacity` |

### 8.2 Computation Steps

1. **Predict all-arm probabilities**: `predict_all_actions(bundle, candidate_frame)` -> DataFrame with one column per arm.

2. **Compute incremental values**: For each `(row i, treated arm a)`: apply the incremental value formula (Section 9.2). CONTROL is the reference and not a candidate arm.

3. **Pre-screen eligibility**: call `decide_action` per row without `recovery_probability` injection to identify unconditional STOP cases (rules R001-R004: opt-out, fraud, hard-decline, retry limit >= 4). These rows receive NO_INTERVENTION before the optimizer runs, without consuming budget. The pre-screen does not replace the post-allocation `decide_action` call.

4. **Optimizer allocation**: solve the constrained allocation problem over eligible rows (see Section 11).

5. **Policy authorization**: for every row (allocated and not), call `decide_action`. For allocated rows, inject `P_hat_top_arm(i)` as `recovery_probability`. Record `optimizer_recommendation` and `authorized_action` separately.

6. **Build portfolio audit**: freeze per-row and portfolio-level records.

### 8.3 Output

A `PortfolioAllocation` frozen structure containing per-row `PortfolioEntry` objects and a `PortfolioSummary` (see Section 19).

---

## 9. Action-Value Formulation

### 9.1 Choice: Incremental Value Relative to CONTROL

**Decision: incremental value relative to CONTROL.**

Rationale from repository evidence:
1. The Day 5/6 incremental revenue formula in `ml/decision_policy.py` and `ml/incremental.py` already computes `(P_hat_a - P_hat_CONTROL) * amount - cost - risk_penalty`. The portfolio optimizer reuses this formula exactly.
2. The `BoundedRecommender` (Day 6) uses this formulation for per-row selection. Day 7 extends it to portfolio level.
3. Absolute expected value (`P_hat_a * amount - cost`) would allocate budget to rows where intervention adds no marginal value over doing nothing.
4. CONTROL is the no-intervention outcome. `P_hat_CONTROL` estimates recovery probability under no action. `P_hat_a - P_hat_CONTROL` is the estimated marginal lift from arm `a`.

### 9.2 Formula

For eligible row `i` and treated arm `a` in TREATED_ARMS:

```
IncrementalValue(i, a) =
    (P_hat_a(i) - P_hat_CONTROL(i)) * amount_inr(i)
    - RETRY_INTERVENTION_COST_INR
    - risk_penalty(i)
```

where:
- `P_hat_a(i)` = `predict_action_probability(bundle, row_i, arm=a)`
- `P_hat_CONTROL(i)` = `predict_action_probability(bundle, row_i, arm="CONTROL")`
- `RETRY_INTERVENTION_COST_INR = 10.0` (imported from `recovery.scoring`, illustrative)
- `risk_penalty(i) = UNKNOWN_CATEGORY_RISK_FRACTION * amount_inr(i)` iff `failure_category == "unknown"` else `0.0`

This formula is identical to Day 6's `BoundedRecommender.recommend()`. No new formula is invented.

**Disclosed cost simplification** (inherited from Day 5/6): a single retry-cost constant is applied uniformly to all treated arms including REQUEST_UPDATE and HUMAN_REVIEW. Because cost and risk-penalty are arm-independent per row, argmax arm reduces to argmax `(P_hat_a - P_hat_CONTROL) * amount_inr`.

### 9.3 CONTROL/No-Intervention Value

CONTROL is not a candidate action in the allocation. Its incremental value is 0 by definition. Rows assigned NO_INTERVENTION contribute 0 to the optimizer objective.

### 9.4 Negative Incremental Value

If `IncrementalValue(i, a) <= 0` for all treated arms `a`, row `i` receives NO_INTERVENTION unconditionally, regardless of budget remaining. Allocating a negative-value arm reduces portfolio value and wastes budget.

### 9.5 Numeric Representation

INR values are Python float64. No conversion to integer paise is required. Floating-point ties in the sort are broken deterministically by `attempt_id` (see Section 15). No external solver is used.

---

## 10. CONTROL/No-Intervention Semantics

| Context | Meaning |
|---------|---------|
| Day 4/5 training | An arm in the randomized treatment assignment; CONTROL rows received no intervention and form the reference stratum |
| Day 7 optimizer output | An allocation outcome: a row the optimizer does not assign a treated arm |

These are compatible: `P_hat_CONTROL(i)` estimates recovery probability under no intervention, which is exactly what happens to rows allocated NO_INTERVENTION.

**Explicit Day 7 CONTROL semantics:**
- Every row has exactly one outcome: a treated arm OR NO_INTERVENTION.
- Rows force-stopped by policy pre-screen are recorded with `no_intervention_reason = "policy_pre_screen: STOP"`.
- Rows assigned NO_INTERVENTION do not consume budget or HUMAN_REVIEW capacity.
- All rows (including NO_INTERVENTION) are submitted to `decide_action` post-allocation.

---

## 11. Portfolio Optimization Formulation

### 11.1 Decision Variables

For each eligible row `i` and treated arm `a` in TREATED_ARMS:

```
x(i, a) in {0, 1}
```

`x(i, a) = 1` means arm `a` is recommended for row `i`.

### 11.2 Objective

```
maximize SUM_i SUM_a  x(i, a) * max(IncrementalValue(i, a), 0)
```

The `max(..., 0)` clamp ensures only positive-value interventions are attractive. Non-positive-value arms are excluded before the optimizer runs (C5).

### 11.3 Constraints

**C1. At-most-one action per row:**
```
SUM_a x(i, a) <= 1   for all i
```

**C2. No-intervention option:**
When `x(i, a) = 0` for all `a`, row `i` receives NO_INTERVENTION. Always feasible.

**C3. Global intervention budget:**
```
SUM_i SUM_a x(i, a) <= budget_limit
```
`budget_limit` is a non-negative integer or None (unconstrained). Zero -> all rows NO_INTERVENTION.

**C4. HUMAN_REVIEW capacity:**
```
SUM_i x(i, HUMAN_REVIEW) <= human_review_capacity
```
`human_review_capacity` is a non-negative integer or None (unconstrained). Zero -> HUMAN_REVIEW excluded.

**C5. Non-negative value gate:**
```
x(i, a) = 0   if IncrementalValue(i, a) <= 0
```
Enforced before the optimizer runs.

**C6. Policy eligibility pre-screen:**
Rows where `decide_action` returns STOP from context-only rules (R001-R004) are excluded from optimizer allocation.

No additional constraints are added. YAGNI applies: no constraint is included without documented justification.

### 11.4 Formal Problem Class

This is a two-dimensional binary knapsack: items are (row, arm) pairs with value (incremental INR) and two resource dimensions: total budget count (all items weight 1) and HUMAN_REVIEW-only count. The general 2D 0-1 knapsack is NP-hard. For n <= 750 rows and 4 treated arms, both exact DP and heuristic greedy approaches are tractable.

---

## 12. Constraint Detail

### C1: At-most-one action per row

Enforced structurally by the greedy: each row is allocated at most once. The `allocated` set check prevents any row from appearing twice.

### C2: No-intervention option

Every row has NO_INTERVENTION as a zero-cost, zero-revenue fallback. The feasibility set is always non-empty.

### C3: Global intervention budget

Default: unconstrained. Zero budget -> all rows NO_INTERVENTION, `optimizer_status = "budget_exhausted_before_allocation"`.

### C4: HUMAN_REVIEW capacity

Default: unconstrained. Zero -> HUMAN_REVIEW excluded; rows best served by HUMAN_REVIEW receive second-best positive-value arm or NO_INTERVENTION.

### C5: Non-negative value gate

Applied per (row, arm) pair before the optimizer. Rows with no positive-value arm receive `no_intervention_reason = "non_positive_value"`.

### C6: Policy eligibility pre-screen

The pre-screen calls `decide_action` without `recovery_probability`. Rules R001-R004 fire on context columns alone. Rules R006-R008 reference `recovery_probability` and will not fire during pre-screen; they are handled in post-allocation `decide_action`.

Pre-screened rows appear in the portfolio output with `optimizer_recommendation = "NO_INTERVENTION"`, `no_intervention_reason = "policy_pre_screen: STOP"`, and `authorized_action = "STOP"`. They count in `pre_screen_stopped` and are excluded from `eligible_count`.

---

## 13. Solver Alternatives Considered

### Option A: Specialized Deterministic Greedy

Sort all candidate (row, arm) pairs by `IncrementalValue` descending (deterministic secondary key on `attempt_id`). Iterate in sorted order, assigning each pair if: (a) row not yet allocated, (b) budget not exhausted, (c) HUMAN_REVIEW capacity not exhausted if arm is HUMAN_REVIEW.

**Optimality for single-constraint case (C3 only):** Exactly optimal - equivalent to fractional knapsack with equal item weights.
**Optimality for two-constraint case (C3 + C4):** Not guaranteed optimal. The gap depends on data distribution; bounded but not proven zero.
**Simplicity:** Maximum. Pure Python standard library. Trivially auditable, deterministic, zero new dependencies.
**Selected as primary solver and as greedy baseline for comparison.**

### Option B: Exact 2D Dynamic Programming

2D DP table of size `(budget + 1) x (human_review_capacity + 1)` over n items. Complexity `O(n * budget * capacity)`.

**Optimality:** Exact.
**Tractability:** For small budgets (budget <= 200, capacity <= 50): approximately 750 * 201 * 51 = 7.7M operations - feasible. For large budgets (budget -> 750): approximately 750 * 751 * 751 = 423M entries - impractical without memory optimization.
**Problem:** The optimizer must handle unconstrained budgets. DP degrades for large budgets. A fallback to greedy for large budgets adds complexity.
**Not selected as primary.** Documented as extension path for cases requiring exact optimality at small constraint values. The interface supports a pluggable solver strategy.

### Option C: OR-Tools CP-SAT

Express as a binary integer program in OR-Tools CP-SAT (SAT-based branch-and-bound).

**Correctness:** Exact.
**Dependency cost:** `ortools` is not in `requirements.txt`. Adding it for a synthetic-world prototype with n <= 750 is not justified.
**Determinism:** Version-specific; harder to guarantee across environments than pure Python sort.
**Justification burden not met:** Python stdlib is sufficient for the greedy. No mathematical property of the problem requires a SAT solver for n <= 750.
**Not selected.**

---

## 14. Solver Decision

**Chosen solver: Deterministic ranked greedy with HUMAN_REVIEW quota enforcement.**

**Rationale:**
1. **Portfolio sizes are small (n <= 750)**. Greedy-vs-optimal gap is bounded, measurable, and documented honestly.
2. **No new dependency is justified.** Python stdlib and numpy suffice. OR-Tools for n <= 750 violates YAGNI.
3. **Greedy is exactly optimal for the dominant single-constraint case** (budget only, large HUMAN_REVIEW capacity).
4. **Exact DP is documented as a future extension** for small constraint values. The interface supports pluggable solver strategies.
5. **Auditability.** The greedy sort is trivially inspectable. The audit trace records each item's sort rank.
6. **Determinism.** Pure-Python sort with a deterministic three-level key is guaranteed deterministic. No solver version dependency.

**Algorithm:**

```
Input: eligible candidate (row, arm) pairs with IncrementalValue > 0
       budget_limit (int or None), human_review_capacity (int or None)

1. Build candidate list: all (i, a, value) where IncrementalValue(i, a) > 0
2. Sort candidates by:
   - Primary: value descending
   - Secondary: attempt_id ascending (lexicographic)
   - Tertiary: ARM_ORDER index ascending
3. Initialize: allocated = {}; budget_used = 0; hr_used = 0
4. For each (i, a, value, rank) in sorted candidates:
   a. If i already in allocated: skip (at-most-one per row)
   b. If budget_limit is not None and budget_used >= budget_limit: skip
   c. If a == "HUMAN_REVIEW" and human_review_capacity is not None
      and hr_used >= human_review_capacity: skip (not break; continue for non-HR arms)
   d. Allocate: allocated[i] = (a, value, rank)
      budget_used += 1
      if a == "HUMAN_REVIEW": hr_used += 1
5. Eligible rows not in allocated -> NO_INTERVENTION
   (reason: "non_positive_value" if no positive arm existed;
            "budget_exhausted" if budget was the binding constraint;
            "human_review_capacity_exhausted" if HR cap was binding and no fallback)
```

**Key behavior:** Step 4c does NOT break when HUMAN_REVIEW capacity is exhausted. It skips the HUMAN_REVIEW item and continues scanning lower-ranked candidates. A row whose best arm is HUMAN_REVIEW may appear again in the candidate list with a different arm (its second-best). Step 4a prevents double-allocation.

**Budget exhaustion:** When `budget_used >= budget_limit`, step 4b skips all remaining items. The loop continues to record sort ranks but allocates nothing. In practice the loop can break early for efficiency; the result is identical.

---

## 15. Determinism and Numeric Representation

### 15.1 Sort Key

- Primary: `IncrementalValue(i, a)` descending
- Secondary: `attempt_id` ascending, lexicographic (format `ATT-NNNNNN`; lexicographic equals numeric)
- Tertiary: ARM_ORDER index ascending (RETRY_NOW=1, RETRY_LATER=2, REQUEST_UPDATE=3, HUMAN_REVIEW=4)

Guaranteed deterministic because `attempt_id` is unique per row (primary key), making the secondary key resolve all remaining ties after the primary.

### 15.2 Float Safety

`IncrementalValue` is computed in float64. Sub-paise differences are always resolved deterministically by the secondary key. No paise rounding before comparison. The deterministic key guarantees reproducibility regardless of sub-paise floating-point behavior.

### 15.3 Constraint Comparison

Budget and HUMAN_REVIEW capacity comparisons use integer arithmetic (counts of allocated items). No float arithmetic in constraint enforcement.

### 15.4 No Wall-Clock Reads

The optimizer reads no wall clock and draws no randomness. Identical bundle and frame produce identical probability predictions.

---

## 16. Policy Interaction and Override Contract

### 16.1 Two-Layer Semantics

Every row has two distinct action fields:

| Field | Source | Meaning |
|-------|--------|---------|
| `optimizer_recommendation` | Optimizer allocation | Arm selected (or `"NO_INTERVENTION"`) |
| `authorized_action` | `decide_action()` | Action authorized by the deterministic policy engine |

These are always stored separately and may legitimately differ.

### 16.2 Post-Allocation Policy Authorization

After optimizer allocation, every row is submitted to `decide_action`:
- **Allocated rows**: `recovery_probability` set to `P_hat_top_arm(i)` (Day 6 injection convention).
- **NO_INTERVENTION rows**: `recovery_probability` set to `P_hat_CONTROL(i)`.

Both `optimizer_recommendation` and `authorized_action` are recorded for every row.

### 16.3 Resource Accounting on Policy Override

When the policy overrides an allocated action to STOP, **the optimizer's budget accounting is NOT retroactively adjusted**.

**Explicit contract:**
- `optimizer_budget_allocated` records what the optimizer allocated.
- `post_policy_net_authorized` = `optimizer_budget_allocated - total_policy_stop_overrides`.
- HUMAN_REVIEW capacity is NOT retroactively freed when a HUMAN_REVIEW recommendation is overridden to STOP.
- Both `optimizer_budget_allocated` and `post_policy_net_authorized` appear in the portfolio summary.

**Rationale:** Retroactive adjustment creates a feedback loop. The pre-screen already filters unconditional STOP cases. Recording divergence is sufficient and auditable.

**Divergence reporting:**
- `total_policy_overrides`: rows where `authorized_action != optimizer_recommendation` and `optimizer_recommendation != "NO_INTERVENTION"`
- `total_policy_stop_overrides`: subset where `authorized_action == "STOP"`

### 16.4 Safety Invariant

No STOP case ever becomes an authorized intervention. The policy engine always has the final word. `decide_action` is mandated for every row without exception.

---

## 17. Failure Handling

| Condition | Behavior | `optimizer_status` |
|-----------|----------|--------------------|
| Empty portfolio (`candidate_frame` has 0 rows) | Valid empty `PortfolioAllocation`, all counts zero | `"empty_portfolio"` |
| `budget_limit = 0` | All rows NO_INTERVENTION | `"budget_exhausted_before_allocation"` |
| `human_review_capacity = 0` | HUMAN_REVIEW excluded; rows get second-best arm or NO_INTERVENTION | `"human_review_capacity_zero"` |
| No (row, arm) pair has positive incremental value | All rows NO_INTERVENTION | `"no_positive_value_candidates"` |
| NaN/non-finite predictions for a row | Row excluded from allocation, `no_intervention_reason = "invalid_prediction"` | Count in `invalid_prediction_count` |
| Missing required columns in `candidate_frame` | Raises `ValueError` naming missing columns before any computation | N/A |
| `budget_limit < 0` or `human_review_capacity < 0` | Raises `ValueError` | N/A |
| Optimizer algorithm failure | Cannot occur; greedy terminates in O(n log n) with guaranteed feasibility | N/A |

---

## 18. Leakage Boundaries

### 18.1 Allowed Optimizer Inputs

| Input | Justification |
|-------|---------------|
| Decision-time context features (Day 2 whitelist) | Pre-decision; enforced by `build_feature_matrix` |
| `amount_inr`, `failure_category` | Decision-time; required for incremental value formula |
| `attempt_id`, `payment_id`, `customer_id` | Identifiers; audit trace only |
| `event_timestamp` | Audit trace timestamp only |
| `customer_opted_out`, `fraud_risk`, `attempt_number` | Decision-time policy-gate columns |
| Per-arm model probabilities from `ActionModelBundle` | Decision-time predictions |
| `RETRY_INTERVENTION_COST_INR`, `UNKNOWN_CATEGORY_RISK_FRACTION` | Constants from `recovery.scoring` |

### 18.2 Forbidden Optimizer Inputs

| Forbidden Field | Category |
|-----------------|----------|
| `recovered` | Realized label |
| `recovery_time_hours` | Realized timing |
| `simulated_recovered` | Simulated outcome |
| `simulated_recovered_amount_inr` | Simulated revenue |
| `treatment_timestamp`, `outcome_timestamp` | Post-intervention timing |
| `base_recovery_propensity`, `action_effect_logit`, `propensity_under_assignment` | Simulator ground truth; evaluation only |
| `assignment_probability`, `arm_source`, `assigned_action` | Post-randomization assignment metadata |

Synthetic ground truth is forbidden as an optimizer input. It may be used ONLY in offline evaluation (Section 21), after allocation is fixed.

### 18.3 Leakage Guard Strategy

1. Optimizer input interface enforces a column whitelist before any computation. Columns in the forbidden set raise `ValueError`.
2. A dedicated `OPTIMIZER_FORBIDDEN_COLUMNS` frozenset is defined in the optimizer module, extending `ml.features.FORBIDDEN_FEATURES` with Day 4/5 outcome and assignment columns.
3. `predict_all_actions(bundle, candidate_frame)` calls `build_feature_matrix` internally, mechanically enforcing the Day 2 whitelist.
4. Tests verify: hostile frame with forbidden columns raises `ValueError`; `build_feature_matrix` whitelist is not circumvented.

---

## 19. Audit and Observability Design

### 19.1 Per-Row Portfolio Entry (`PortfolioEntry`)

| Field | Type | Description |
|-------|------|-------------|
| `attempt_id` | `str` | Payment attempt identifier |
| `payment_id` | `str` | Payment identifier |
| `optimizer_recommendation` | `str` | Arm recommended by optimizer, or `"NO_INTERVENTION"` |
| `no_intervention_reason` | `str or None` | Reason if NO_INTERVENTION: `"non_positive_value"`, `"policy_pre_screen"`, `"budget_exhausted"`, `"human_review_capacity_exhausted"`, `"invalid_prediction"` |
| `incremental_value_by_arm` | `dict[str, float]` | Incremental revenue estimate per treated arm (positive-value arms only) |
| `best_arm_incremental_value_inr` | `float or None` | Incremental value of recommended arm; `None` if NO_INTERVENTION |
| `optimizer_sort_rank` | `int or None` | Position in sorted candidate list (1 = highest value); `None` if not a candidate |
| `authorized_action` | `str` | Canonical action authorized by `decide_action` |
| `authorization_reason` | `str` | Reason string from `PolicyDecision` |
| `matched_rule_id` | `str or None` | Rule that fired in `decide_action` |
| `policy_overrode_recommendation` | `bool` | True if `authorized_action != optimizer_recommendation` and `optimizer_recommendation != "NO_INTERVENTION"` |

### 19.2 Portfolio Summary (`PortfolioSummary`)

| Field | Type | Description |
|-------|------|-------------|
| `total_rows` | `int` | Total rows submitted |
| `pre_screen_stopped` | `int` | Rows removed by policy pre-screen |
| `eligible_count` | `int` | Rows submitted to optimizer |
| `invalid_prediction_count` | `int` | Rows excluded due to NaN predictions |
| `optimizer_budget_allocated` | `int` | Treated arms allocated by optimizer |
| `no_intervention_count` | `int` | Eligible rows assigned NO_INTERVENTION |
| `human_review_allocated` | `int` | HUMAN_REVIEW arms allocated |
| `budget_limit` | `int or None` | Configured limit |
| `human_review_capacity_limit` | `int or None` | Configured limit |
| `post_policy_net_authorized` | `int` | Rows where `authorized_action != "STOP"` |
| `total_policy_overrides` | `int` | Rows where policy diverged from recommendation |
| `total_policy_stop_overrides` | `int` | Subset where `authorized_action == "STOP"` |
| `optimizer_objective_value_inr` | `float` | Sum of `best_arm_incremental_value_inr` over allocated rows (MODEL ESTIMATE) |
| `optimizer_status` | `str` | Status code |
| `action_recommendation_counts` | `dict[str, int]` | Count per `optimizer_recommendation` value |
| `action_authorized_counts` | `dict[str, int]` | Count per `authorized_action` value |

**Row-count invariant:** `total_rows == pre_screen_stopped + invalid_prediction_count + optimizer_budget_allocated + no_intervention_count`. Verified by tests.

### 19.3 JSON Serialization

`PortfolioAllocation` serializes to JSON deterministically. Identical inputs produce byte-identical JSON. Implementation follows `recovery/audit.py::traces_to_json`: fixed field order, no `sort_keys` dependency.

### 19.4 Label Discipline

`optimizer_objective_value_inr` and all monetary quantities in audit output are labeled `"MODEL ESTIMATE"` in reports. No label implies realized revenue.

---

## 20. Greedy Baseline

The greedy baseline enforces the same constraints but uses a row-first (rather than global-pair-first) selection strategy.

**Greedy algorithm:**

```
1. Sort eligible rows by best positive-value arm's IncrementalValue descending
   (ties: attempt_id ascending)
2. For each row i in sorted row order:
   a. Find arm a* = argmax_a IncrementalValue(i, a) where value > 0
      (ties: ARM_ORDER index ascending)
   b. If a* exists and budget not exhausted:
      - If a* != HUMAN_REVIEW or hr_capacity not exhausted: allocate a*
      - Else (a* == HUMAN_REVIEW, hr exhausted):
          find next-best arm b != HUMAN_REVIEW with value > 0
          if found and budget not exhausted: allocate b
          else: NO_INTERVENTION (reason: "human_review_capacity_exhausted" or "budget_exhausted")
   c. Else: NO_INTERVENTION (reason: "non_positive_value" or "budget_exhausted")
```

**Key difference from portfolio optimizer:** Greedy considers each row's best arm based only on that row's values. Portfolio optimizer globally sorts all (row, arm) pairs and may skip a row's locally-best arm to allocate a globally-higher-value arm to a different row.

**Fair comparison requires:** Identical `candidate_frame`, identical per-arm probability predictions (same bundle call), identical `optimizer_config`, identical policy configuration.

**Comparison metrics:**
- `optimizer_objective_value_inr` vs `greedy_objective_value_inr`
- `post_policy_net_authorized` counts
- Action distribution (HUMAN_REVIEW utilization)
- Realized synthetic outcome comparison (Section 21)

If the portfolio optimizer does not outperform the greedy, that is documented honestly.

---

## 21. Offline Synthetic Evaluation Methodology

### 21.1 Evaluation Isolation

Synthetic ground truth is never fed into the optimizer or greedy. Allocation is fixed first, then outcomes are joined:

```
1. Run optimizer over candidate_frame (no outcome columns present)
2. Seal PortfolioAllocation (fixed)
3. Load Day 4 outcome frame separately
4. Join sealed allocation to outcome frame on attempt_id
5. Compute realized metrics from joined frame
```

### 21.2 Evaluation Metrics

| Metric | Label | Description |
|--------|-------|-------------|
| `estimated_portfolio_value_inr` | MODEL ESTIMATE | Sum of `best_arm_incremental_value_inr` over authorized rows |
| `realized_recovered_count` | OBSERVED SIMULATED OUTCOME | Count of `simulated_recovered == 1` over authorized rows |
| `realized_recovered_amount_inr` | OBSERVED SIMULATED OUTCOME | Sum of `simulated_recovered_amount_inr` over authorized rows |
| `no_intervention_realized_recovered_amount_inr` | OBSERVED SIMULATED OUTCOME (CONFOUNDED) | Sum over NO_INTERVENTION rows |
| `randomized_control_realized_recovered_amount_inr` | OBSERVED SIMULATED OUTCOME (UNCONFOUNDED) | Sum over Day 4 CONTROL-arm rows in test split |
| `optimizer_vs_greedy_objective_delta_inr` | MODEL ESTIMATE | Optimizer minus greedy estimated objective |
| `optimizer_vs_greedy_realized_delta_inr` | OBSERVED SIMULATED OUTCOME | Optimizer minus greedy realized recovery |

### 21.3 CONTROL Comparison

Two comparisons, explicitly labeled:

1. **Confounded**: NO_INTERVENTION rows' `simulated_recovered` amounts. Confounded because optimizer selected which rows to intervene on. Label: `"CONFOUNDED: optimizer-selected NO_INTERVENTION"`.

2. **Unconfounded**: Rows assigned to the Day 4 CONTROL arm in the randomized stratum of the test split. Label: `"UNCONFOUNDED: randomized CONTROL arm"`.

### 21.4 Label Discipline

All evaluation outputs carry MODEL ESTIMATE, OBSERVED SIMULATED OUTCOME, or SIMULATED GROUND TRUTH. "Causal" appears only in disclaimers.

---

## 22. Testing Strategy

### 22.1 Test Categories

| Category | Description |
|----------|-------------|
| Feasibility | All allocations satisfy every active constraint; verified programmatically |
| Determinism | Two identical calls produce byte-identical sorted-JSON output |
| Policy safety | Crafted STOP contexts produce `authorized_action == "STOP"` regardless of recommendation |
| Leakage guard | Hostile frames with forbidden columns raise `ValueError`; whitelist enforced |
| Zero budget | All rows NO_INTERVENTION; `optimizer_budget_allocated == 0` |
| Zero HUMAN_REVIEW capacity | `human_review_allocated == 0`; affected rows get second-best arm or NO_INTERVENTION |
| Empty portfolio | Valid empty allocation; no error |
| All non-positive values | `optimizer_status == "no_positive_value_candidates"` |
| At-most-one per row | No `attempt_id` appears twice in allocated set |
| Row-count invariant | `total_rows == pre_screen_stopped + invalid_prediction_count + optimizer_budget_allocated + no_intervention_count` |
| Audit completeness | Every row has all required fields; JSON serializes |
| NaN prediction guard | Rows with NaN receive NO_INTERVENTION, reason `"invalid_prediction"` |
| Policy override accounting | `total_policy_overrides` accurately counts divergences |
| Budget exhaustion | Rows beyond budget receive `no_intervention_reason = "budget_exhausted"` |
| HUMAN_REVIEW fallback | Rows whose best arm is HUMAN_REVIEW receive second-best when capacity exhausted |
| Greedy vs optimizer | Identical inputs, different objective values in fixtures where optimizer is provably better |
| Evaluation isolation | Outcome frame not present in candidate frame; join occurs post-allocation |

### 22.2 Test Infrastructure

Tests use synthetic constant-probability fixtures (not the full 5,000-row dataset) to make expected outcomes hand-calculable. The canonical full-dataset run is reserved for `DAY7_RESULTS.md`.

### 22.3 Estimated Test Count

Day 7 adds approximately 80-120 new tests. Cumulative total: approximately 855-895 (from 775 at Day 6).

---

## 23. GO/NO-GO Gates

### G1: Feasibility

**Criterion:** Every allocation satisfies all active constraints: no row allocated more than once; `optimizer_budget_allocated <= budget_limit`; `human_review_allocated <= human_review_capacity`; no non-positive-value arm allocated.

**Evidence:** Programmatic constraint check on canonical run; passing test suite.

**Failure condition:** Any constraint violation.

### G2: Determinism

**Criterion:** Two fresh runs on identical canonical inputs produce byte-identical sorted-JSON output.

**Evidence:** SHA-256 digest comparison of two runs in `DAY7_RESULTS.md`.

**Failure condition:** Any digest mismatch.

### G3: Policy Safety

**Criterion:** Three crafted STOP contexts (customer opted-out, fraud risk, hard decline) produce `authorized_action == "STOP"` in all cases.

**Evidence:** Policy safety probe in `DAY7_RESULTS.md`; unit test.

**Failure condition:** Any STOP context receives non-STOP authorized action.

### G4: Leakage Safety

**Criterion:** A hostile candidate frame containing forbidden columns raises `ValueError` before any computation, OR demonstrably does not cause those columns to enter probability computation.

**Evidence:** Unit test verifying `ValueError` before `predict_all_actions` call.

**Failure condition:** Any forbidden column contributes to optimizer output.

### G5: Baseline Comparison

**Criterion:** Portfolio optimizer and greedy baseline run with identical inputs. Results reported in `DAY7_RESULTS.md` regardless of which performs better.

**Evidence:** Comparison table in `DAY7_RESULTS.md`.

**Failure condition:** Comparison not run; or run with different inputs.

### G6: Synthetic Realized Evaluation

**Criterion:** Optimizer allocation evaluated against held-out Day 4 synthetic outcomes without those outcomes entering the allocation decision.

**Evidence:** Evaluation table in `DAY7_RESULTS.md` with MODEL ESTIMATE vs OBSERVED SIMULATED OUTCOME quantities.

**Failure condition:** Evaluation not run; or forbidden outcome column present in frame passed to optimizer.

### G7: Audit Completeness

**Criterion:** Every row has a `PortfolioEntry` with all required fields. Portfolio serializes to valid JSON. Row-count invariant holds: `total_rows == pre_screen_stopped + invalid_prediction_count + optimizer_budget_allocated + no_intervention_count`.

**Evidence:** Canonical run JSON serializes; row-count invariant verified.

**Failure condition:** Missing fields, serialization error, or row-count mismatch.

---

## 24. Risks and Limitations

**R1. Greedy sub-optimality with two binding constraints.**
When both global budget and HUMAN_REVIEW capacity simultaneously bind, the greedy may be sub-optimal. Gap is not analytically quantified. Exact DP is documented as future extension.

**R2. Model overestimation from Day 5/6.**
Day 5/6 found model estimates overshoot truth on every treated arm (+INR 87-231 per case). Optimizer estimated objective will exceed realized synthetic outcome. Documented with MODEL ESTIMATE vs OBSERVED SIMULATED OUTCOME labeling.

**R3. Probability injection choice for policy.**
Top arm probability injection into `decide_action` may differ from Day 4 simulator probabilities for the same row. Policy safety (G3) holds regardless.

**R4. HUMAN_REVIEW capacity asymmetry.**
High-value HUMAN_REVIEW opportunities fall back to second-best arms when capacity is exhausted. Correct behavior; recorded in audit.

**R5. Single canonical run.**
The GO/NO-GO verdict relies on dataset seed 42. Seed sensitivity analysis is deferred.

**R6. Thin regret margin from Day 6.**
Day 6 relative regret margin: 0.1383 vs threshold 0.15; per-seed sd approximately 0.010. Day 7 inherits this limitation.

**R7. Uniform cost simplification.**
The uniform retry cost applied to all arms is an acknowledged simplification. Per-arm cost differentiation is deferred.

**R8. Synthetic-world scope.**
All results are valid only within the synthetic world. No production claim is supported.

---

## 25. Future Extensions Explicitly Deferred

- Exact 2D DP solver for provable optimality at small constraint values
- OR-Tools integration for large-scale portfolios
- Per-arm differentiated action costs
- Per-segment fairness constraints
- Sequential or online allocation
- Multi-period portfolio
- Seed-sensitivity sweep across dataset seeds
- Out-of-sample stage-1 probability refits
- Pooled model as optimizer input
- Production deployment of any kind

---

## Appendix: Contract Summary

| Contract | Value | Source |
|----------|-------|--------|
| Action vocabulary | `RETRY_NOW, RETRY_LATER, REQUEST_UPDATE, HUMAN_REVIEW, STOP` | `config/business_rules.yaml` |
| TREATED_ARMS | `RETRY_NOW, RETRY_LATER, REQUEST_UPDATE, HUMAN_REVIEW` | `ml/action_model.py` |
| Intervention cost | `RETRY_INTERVENTION_COST_INR = 10.0` INR (illustrative) | `recovery/scoring.py` |
| Unknown risk fraction | `UNKNOWN_CATEGORY_RISK_FRACTION = 0.05` (illustrative) | `recovery/scoring.py` |
| Model contract | `P(recovered | context, action=a)` per arm | Day 5/6 |
| Incremental value formula | `(P_hat_a - P_hat_CONTROL) * amount - cost - risk_penalty` | `ml/decision_policy.py`, `ml/incremental.py` |
| Optimization objective | Maximize sum of positive incremental values subject to constraints | This document |
| Solver | Deterministic ranked greedy with HUMAN_REVIEW quota enforcement | This document |
| Tie-breaking | IncrementalValue desc, attempt_id asc, ARM_ORDER index asc | This document |
| Policy authorization boundary | `decide_action()` in `recovery/policy.py` | Days 1-6 |
| STOP dominance | Preserved via `decide_action()` | Days 1-6 |
| Resource accounting on override | Budget NOT retroactively freed | This document |
| Label vocabulary | MODEL ESTIMATE / OBSERVED SIMULATED OUTCOME / SIMULATED GROUND TRUTH | Days 4-6 |
| Forbidden optimizer inputs | Post-decision outcomes, ground truth, assignment metadata | This document + `ml/features.py` |
| Evaluation isolation | Allocation fixed before outcome join | This document |
| Synthetic-world scope | All claims within Day 4 simulator only | Days 4-6 |
