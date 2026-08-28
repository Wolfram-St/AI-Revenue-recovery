# Implementation Plan — Day 7: Bounded Recovery Portfolio Optimizer

**Date:** 2026-08-27  
**Status:** Approved Design Implementation Plan  
**Target Branch:** `feature/day7-optimizer`  
**Base Commit:** `06e6b4d`  
**Design Spec Commit:** `c85df66` (`docs/superpowers/specs/2026-08-27-day7-bounded-recovery-portfolio-optimizer-design.md`)  

---

## 1. Overview & Architectural Boundaries

This implementation plan breaks down Day 7 (**Bounded Recovery Portfolio Optimizer**) into eight executable, test-driven phases. Day 7 implements a deterministic, constraint-aware recovery portfolio optimizer operating over decision-time context features and calibrated per-arm probability models.

### Preserved Core Pipeline Architecture

```
Action-aware models (Day 5 calibrated per-arm XGBoost bundle)
   │
   ▼
Incremental value estimates (Day 5/6 formula relative to CONTROL)
   │
   ▼
Leakage-safe candidate builder (Decision-time whitelist + forbidden column filter)
   │
   ▼
Portfolio optimizer (Ranked greedy solver over global (row, arm) pairs)
   │
   ▼
Optimizer recommendation (optimizer_recommendation per row)
   │
   ▼
Deterministic policy authorization (decide_action() in recovery/policy.py)
   │
   ▼
Authorized portfolio (authorized_action per row; STOP dominance preserved)
   │
   ▼
Synthetic outcome evaluation (Offline join to Day 4 simulated ground truth post-allocation)
```

### Non-Negotiable Invariants

- **I-1: AI/model recommends; deterministic policy authorizes.** The optimizer is a recommendation layer. Every allocated action MUST pass through `decide_action()` before authorization.
- **I-2: STOP remains dominant.** When `stop_precedence: true` (default in `config/business_rules.yaml`), any matching STOP rule overrides any optimizer recommendation.
- **I-3: Optimizer recommendation and authorized_action must never be collapsed.** They are distinct semantic fields in every audit record.
- **I-4: Realized outcomes must never enter candidate construction or allocation.** Forbidden features and post-decision outcome fields are blocked before optimization.
- **I-5: Allocation must be fully fixed before synthetic outcomes are joined.** Synthetic outcomes enter ONLY during offline evaluation post-allocation.
- **I-6: Budget consumed by an optimizer allocation is NOT retroactively freed when policy overrides the recommendation.** Accounting tracks `optimizer_budget_allocated` and `post_policy_net_authorized` separately.
- **I-7: All allocation must be deterministic and reproducible.** Identical inputs and configuration produce byte-identical JSON traces across environments.

---

## 2. Task Breakdown

---

### Task 1: Freeze portfolio interfaces and contracts

#### 1. Goal
Define all dataclasses, frozen structures, domain exceptions, status codes, and constants for Day 7 portfolio optimization and audit without implementing solver logic.

#### 2. Files to create or modify
- `ml/portfolio_optimizer.py` (new module)
- `ml/portfolio_audit.py` (new module)
- `tests/test_portfolio_optimizer.py` (new test file)
- `tests/test_portfolio_audit.py` (new test file)

#### 3. Existing contracts consumed
- `ActionModelBundle` (`ml/action_model.py`)
- `PolicyConfig` (`recovery/policy.py`)
- `ARM_ORDER` (`ml/action_model.py`)
- `FORBIDDEN_FEATURES` (`ml/features.py`)
- `RETRY_INTERVENTION_COST_INR`, `UNKNOWN_CATEGORY_RISK_FRACTION` (`recovery/scoring.py`)

#### 4. New public interfaces
```python
# ml/portfolio_optimizer.py
@dataclass(frozen=True)
class OptimizerConfig:
    budget_limit: int | None = None
    human_review_capacity: int | None = None

class PortfolioOptimizationError(Exception):
    """Raised when portfolio optimization fails invalid domain inputs."""
    pass

OPTIMIZER_FORBIDDEN_COLUMNS: frozenset[str]

# ml/portfolio_audit.py
@dataclass(frozen=True)
class PortfolioEntry:
    attempt_id: str
    payment_id: str
    optimizer_recommendation: str
    no_intervention_reason: str | None
    incremental_value_by_arm: dict[str, float]
    best_arm_incremental_value_inr: float | None
    optimizer_sort_rank: int | None
    authorized_action: str
    authorization_reason: str
    matched_rule_id: str | None
    policy_overrode_recommendation: bool

@dataclass(frozen=True)
class PortfolioSummary:
    total_rows: int
    pre_screen_stopped: int
    eligible_count: int
    invalid_prediction_count: int
    optimizer_budget_allocated: int
    no_intervention_count: int
    human_review_allocated: int
    budget_limit: int | None
    human_review_capacity_limit: int | None
    post_policy_net_authorized: int
    total_policy_overrides: int
    total_policy_stop_overrides: int
    optimizer_objective_value_inr: float
    optimizer_status: str
    action_recommendation_counts: dict[str, int]
    action_authorized_counts: dict[str, int]

@dataclass(frozen=True)
class PortfolioAllocation:
    entries: tuple[PortfolioEntry, ...]
    summary: PortfolioSummary
    metadata: dict

    def to_json(self) -> str: ...
```

#### 5. Exact data-flow boundaries
Pure type definitions and validation methods. No DataFrame transforms or model inferences.

#### 6. Leakage risks and forbidden fields
Define `OPTIMIZER_FORBIDDEN_COLUMNS` extending `FORBIDDEN_FEATURES` with post-decision fields:
`recovered`, `recovery_time_hours`, `simulated_recovered`, `simulated_recovered_amount_inr`, `treatment_timestamp`, `outcome_timestamp`, `base_recovery_propensity`, `action_effect_logit`, `propensity_under_assignment`, `assignment_probability`, `arm_source`, `assigned_action`.

#### 7. Detailed TDD test cases
- `test_optimizer_config_validation`: verify negative budget or capacity raises `ValueError`.
- `test_portfolio_entry_immutability`: verify mutating attributes raises `FrozenInstanceError`.
- `test_portfolio_summary_immutability`: verify dataclass immutability.
- `test_optimizer_forbidden_columns_completeness`: verify all Day 1 label and Day 4/5 outcome/ground-truth fields are present in `OPTIMIZER_FORBIDDEN_COLUMNS`.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -v
```

#### 9. Minimal implementation sequence
1. Create `ml/portfolio_optimizer.py` with `OptimizerConfig`, `PortfolioOptimizationError`, and `OPTIMIZER_FORBIDDEN_COLUMNS`.
2. Create `ml/portfolio_audit.py` with `PortfolioEntry`, `PortfolioSummary`, and `PortfolioAllocation`.
3. Implement `to_json()` helper producing formatted, ordered JSON.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py
```

#### 12. Commit message
`feat(day7): freeze portfolio interfaces and audit contracts`

#### 13. Per-task implementation review checklist
- [ ] All dataclasses marked `frozen=True`
- [ ] `OPTIMIZER_FORBIDDEN_COLUMNS` strictly set
- [ ] JSON serialization method defined
- [ ] Domain exceptions defined

#### 14. Exit criteria
Dataclasses import cleanly, enforce immutability, and pass all contract structure unit tests.

---

### Task 2: Build leakage-safe candidate construction

#### 1. Goal
Implement decision-time feature frame validation, forbidden column checking, per-arm probability inference, incremental revenue calculation, and policy pre-screening (unconditional STOP filtering).

#### 2. Files to create or modify
- `ml/portfolio_optimizer.py`
- `tests/test_portfolio_optimizer.py`

#### 3. Existing contracts consumed
- `predict_all_actions` (`ml/action_model.py`)
- `build_feature_matrix` (`ml/features.py`)
- `decide_action` (`recovery/policy.py`)
- `RETRY_INTERVENTION_COST_INR`, `UNKNOWN_CATEGORY_RISK_FRACTION` (`recovery/scoring.py`)
- `GATE_CONTEXT_COLUMNS` (`ml/decision_policy.py`)

#### 4. New public interfaces
```python
@dataclass(frozen=True)
class CandidatePair:
    attempt_id: str
    payment_id: str
    row_index: int
    arm: str
    incremental_value_inr: float
    p_hat_arm: float
    p_hat_control: float

def build_candidate_universe(
    candidate_frame: pd.DataFrame,
    bundle: ActionModelBundle,
    policy: PolicyConfig,
) -> tuple[tuple[CandidatePair, ...], tuple[PortfolioEntry, ...], dict]:
    """Validate frame, evaluate policy pre-screen, predict arm probabilities, and yield positive candidate pairs."""
```

#### 5. Exact data-flow boundaries
- **Inputs:** Decision-time context `pd.DataFrame`, `ActionModelBundle`, `PolicyConfig`.
- **Outputs:** Tuple of eligible positive `CandidatePair` instances, pre-screened `PortfolioEntry` records (unconditional STOPs), and candidate metadata dict.

#### 6. Leakage risks and forbidden fields
- Check `candidate_frame.columns` against `OPTIMIZER_FORBIDDEN_COLUMNS`. Raise `ValueError` if any forbidden column is present.
- Delegation to `predict_all_actions` internally invokes `build_feature_matrix`, enforcing the Day 2 decision-time feature whitelist.

#### 7. Detailed TDD test cases
- `test_forbidden_column_rejection`: input frame with `simulated_recovered` raises `ValueError`.
- `test_missing_required_context`: missing `amount_inr` or `failure_category` raises `ValueError`.
- `test_nan_policy_gate_column_rejection`: NaN in `customer_opted_out` raises `ValueError`.
- `test_policy_prescreen_filters_unconditional_stop`: rows matching R001-R004 (opted-out, fraud risk) pre-screened to `NO_INTERVENTION` with `authorized_action == "STOP"`.
- `test_incremental_revenue_formula_match`: verify incremental value equals `(P_hat_a - P_hat_CONTROL) * amount - cost - risk_penalty`.
- `test_non_positive_value_exclusion`: arms with `incremental_value_inr <= 0` are excluded from candidate pairs.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_candidate_construction -v
```

#### 9. Minimal implementation sequence
1. Add `_validate_candidate_frame(frame)` checking missing required columns and `OPTIMIZER_FORBIDDEN_COLUMNS`.
2. Add policy pre-screening loop executing `decide_action(row_dict, policy)` without `recovery_probability` injection.
3. Compute per-arm probabilities using `predict_all_actions(bundle, candidate_frame)`.
4. Build `CandidatePair` list for `arm in TREATED_ARMS` where incremental revenue > 0.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_candidate_construction -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py -k test_candidate_construction
```

#### 12. Commit message
`feat(day7): build leakage-safe candidate construction and policy pre-screen`

#### 13. Per-task implementation review checklist
- [ ] Leakage validation runs before any prediction
- [ ] Pre-screen calls `decide_action` without `recovery_probability`
- [ ] Incremental revenue formula reuses scoring constants exactly
- [ ] Non-positive candidates gated out

#### 14. Exit criteria
Candidate pairs constructed cleanly; forbidden columns rejected fast; policy pre-screen correctly excludes unconditional STOP cases.

---

### Task 3: Implement deterministic global-pair ranking

#### 1. Goal
Implement deterministic ranking over candidate (row, arm) pairs using a strict three-level sort key guaranteeing 100% reproducible ordering.

#### 2. Files to create or modify
- `ml/portfolio_optimizer.py`
- `tests/test_portfolio_optimizer.py`

#### 3. Existing contracts consumed
- `ARM_ORDER` (`ml/action_model.py`)
- `CandidatePair` (Task 2)

#### 4. New public interfaces
```python
def sort_key_candidate_pair(candidate: CandidatePair) -> tuple[float, str, int]:
    """Return 3-level key: (-incremental_value_inr, attempt_id, ARM_ORDER_index)."""

def rank_candidate_pairs(
    candidates: Sequence[CandidatePair]
) -> tuple[CandidatePair, ...]:
    """Sort candidates deterministically by incremental value desc, attempt_id asc, ARM_ORDER asc."""
```

#### 5. Exact data-flow boundaries
Pure function. Input: sequence of `CandidatePair`. Output: sorted tuple of `CandidatePair`. Zero random or clock reads.

#### 6. Leakage risks and forbidden fields
Operates only on `CandidatePair` dataclasses. No ground-truth features present.

#### 7. Detailed TDD test cases
- `test_primary_sort_value_descending`: higher incremental value candidates ranked earlier.
- `test_secondary_sort_attempt_id_ascending`: equal incremental values broken deterministically by `attempt_id` ascending.
- `test_tertiary_sort_arm_order_ascending`: same row with equal incremental values broken by `ARM_ORDER` index (RETRY_NOW < RETRY_LATER < REQUEST_UPDATE < HUMAN_REVIEW).
- `test_ranking_permutation_invariance`: shuffling input candidate list produces byte-identical sorted output.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_candidate_ranking -v
```

#### 9. Minimal implementation sequence
1. Implement `sort_key_candidate_pair`.
2. Implement `rank_candidate_pairs` returning `tuple(sorted(candidates, key=sort_key_candidate_pair))`.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_candidate_ranking -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py -k test_candidate_ranking
```

#### 12. Commit message
`feat(day7): implement deterministic global-pair candidate ranking`

#### 13. Per-task implementation review checklist
- [ ] 3-level sort key implemented: value desc, attempt_id asc, ARM_ORDER asc
- [ ] Float values sorted descending using negation
- [ ] Permutation invariance verified by test

#### 14. Exit criteria
Candidate pairs ranked deterministically under all input permutations.

---

### Task 4: Implement deterministic two-constraint portfolio allocation

#### 1. Goal
Implement the constrained portfolio allocation solver enforcing global budget and HUMAN_REVIEW capacity bounds, feasibility rules, deterministic tie-breaking, accounting invariants, and infeasible candidate handling.

#### 2. Files to create or modify
- `ml/portfolio_optimizer.py`
- `tests/test_portfolio_optimizer.py`

#### 3. Existing contracts consumed
- `OptimizerConfig`, `CandidatePair`, `rank_candidate_pairs`, `ARM_ORDER`.

#### 4. New public interfaces
```python
def solve_portfolio_allocation(
    ranked_candidates: tuple[CandidatePair, ...],
    eligible_attempt_ids: set[str],
    config: OptimizerConfig,
) -> tuple[dict[str, CandidatePair], dict[str, str], dict]:
    """Solve constrained portfolio allocation over ranked global candidate pairs.
    
    Returns:
      allocated: dict[attempt_id, CandidatePair]
      unallocated_reasons: dict[attempt_id, str]
      solver_metadata: dict
    """
```

#### 5. Explicit Constraint, Feasibility, and Accounting Specification
- **Constraint C3 (Global Budget Limit):** `SUM_i SUM_a x(i, a) <= budget_limit`. If `budget_limit` is integer >= 0, total allocated items cannot exceed it. `None` = unconstrained.
- **Constraint C4 (HUMAN_REVIEW Capacity Limit):** `SUM_i x(i, HUMAN_REVIEW) <= human_review_capacity`. If `human_review_capacity` is integer >= 0, total allocated HUMAN_REVIEW arms cannot exceed it. `None` = unconstrained.
- **Constraint C1 & Feasibility (At-most-one per row):** `SUM_a x(i, a) <= 1`. Each `attempt_id` can be allocated at most once. Non-allocated rows default to NO_INTERVENTION (zero budget cost, zero value, always feasible).
- **Tie-Breaking:** Handled deterministically by Task 3 ranking (`IncrementalValue` desc, `attempt_id` asc, `ARM_ORDER` index asc).
- **Accounting Invariants:** `total_rows == pre_screen_stopped + invalid_prediction_count + optimizer_budget_allocated + no_intervention_count`.
- **Infeasible / Capacity-Exhausted Behavior:**
  - When `hr_used >= human_review_capacity`, the scan **DOES NOT BREAK**. It skips HUMAN_REVIEW items and continues evaluating non-HUMAN_REVIEW items.
  - A row whose highest-value arm was HUMAN_REVIEW may be allocated its second-best positive arm (if present in lower-ranked candidate pairs) when the loop reaches that pair.
  - When `budget_used >= budget_limit`, all remaining candidate pairs are skipped, and unallocated rows receive `no_intervention_reason = "budget_exhausted"`.

#### 6. Leakage risks and forbidden fields
No ground-truth features consumed. Allocation decisions depend strictly on decision-time incremental values and constraint limits.

#### 7. Detailed TDD test cases
- `test_unconstrained_allocation`: all positive candidates allocated best arm per row.
- `test_global_budget_constraint`: `budget_limit = 2` allocates exactly 2 highest-value global pairs.
- `test_human_review_capacity_constraint`: `human_review_capacity = 1` limits HUMAN_REVIEW to 1; excess HUMAN_REVIEW candidates fall back to second-best arm or NO_INTERVENTION.
- `test_zero_budget`: `budget_limit = 0` results in 0 allocations, status `"budget_exhausted_before_allocation"`.
- `test_zero_human_review_capacity`: `human_review_capacity = 0` excludes HUMAN_REVIEW; other arms allocate normally.
- `test_hand_crafted_portfolio_fixture`: multi-row fixture with hand-calculated expected allocation matching solver output exactly.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_allocation_solver -v
```

#### 9. Minimal implementation sequence
1. Implement `solve_portfolio_allocation` tracking `allocated_rows`, `budget_used`, `hr_used`, and candidate ranks.
2. Ensure skipped HUMAN_REVIEW pairs do not block subsequent non-HUMAN_REVIEW candidate pairs.
3. Record detailed `unallocated_reasons` (`"budget_exhausted"`, `"human_review_capacity_exhausted"`).

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_allocation_solver -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py -k test_allocation_solver
```

#### 12. Commit message
`feat(day7): implement deterministic two-constraint portfolio allocation solver`

#### 13. Per-task implementation review checklist
- [ ] Global budget C3 strictly enforced
- [ ] HUMAN_REVIEW capacity C4 strictly enforced
- [ ] Skipping HUMAN_REVIEW when capacity full continues loop for non-HR arms
- [ ] At-most-one action per row C1 structurally guaranteed
- [ ] Accounting invariants satisfied

#### 14. Exit criteria
Solver passes all unit tests and hand-calculated multi-constraint portfolio fixtures.

---

### Task 5: Apply deterministic policy authorization without corrupting allocation accounting

#### 1. Goal
Implement post-allocation policy authorization for every row via `decide_action()`, recording `optimizer_recommendation` and `authorized_action` as distinct fields, enforcing STOP dominance, and maintaining budget accounting invariants without retroactive budget freeing.

#### 2. Files to create or modify
- `ml/portfolio_optimizer.py`
- `ml/portfolio_audit.py`
- `tests/test_portfolio_optimizer.py`
- `tests/test_portfolio_audit.py`

#### 3. Existing contracts consumed
- `decide_action`, `PolicyConfig`, `PolicyDecision` (`recovery/policy.py`)
- `CandidateRecommendation` injection pattern (`ml/decision_policy.py`)
- Dataclasses from Task 1

#### 4. New public interfaces
```python
def optimize_portfolio(
    candidate_frame: pd.DataFrame,
    bundle: ActionModelBundle,
    policy: PolicyConfig | None = None,
    config: OptimizerConfig | None = None,
) -> PortfolioAllocation:
    """Run full bounded portfolio optimization and policy authorization pipeline."""
```

#### 5. Data-flow boundaries & Invariants Enforced
- **I-1:** AI recommends; policy authorizes via `decide_action()`.
- **I-2:** STOP dominance (context rules R001-R004 and probability rules R006-R008).
- **I-3:** `optimizer_recommendation` and `authorized_action` stored in separate fields; never collapsed.
- **I-6:** Budget consumed by optimizer allocation (`optimizer_budget_allocated`) is NOT retroactively freed when policy overrides recommendation to STOP. `post_policy_net_authorized` tracks net authorized interventions.

Probability injection conventions:
- For allocated rows: inject `P_hat_top_arm(i)` as `recovery_probability`.
- For NO_INTERVENTION rows: inject `P_hat_CONTROL(i)` as `recovery_probability`.

#### 6. Leakage risks and forbidden fields
No post-decision outcome fields allowed in `candidate_frame` or passed to `decide_action`.

#### 7. Detailed TDD test cases
- `test_policy_override_stop`: crafted context where optimizer recommends `RETRY_NOW` but policy condition fires STOP -> `optimizer_recommendation == "RETRY_NOW"`, `authorized_action == "STOP"`, `policy_overrode_recommendation == True`.
- `test_budget_accounting_non_retroactive`: verify `optimizer_budget_allocated` is NOT decremented when policy overrides a recommendation to STOP.
- `test_summary_row_count_invariant`: `total_rows == pre_screen_stopped + invalid_prediction_count + optimizer_budget_allocated + no_intervention_count`.
- `test_portfolio_allocation_json_reproducibility`: two identical pipeline calls yield byte-identical `to_json()` output.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -k test_optimize_portfolio -v
```

#### 9. Minimal implementation sequence
1. Implement post-allocation loop calling `decide_action` per row with injected probability.
2. Build `PortfolioEntry` per row.
3. Compute `PortfolioSummary` metrics (`post_policy_net_authorized`, `total_policy_overrides`, `total_policy_stop_overrides`).
4. Assemble and return `PortfolioAllocation`.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -k test_optimize_portfolio -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -k test_optimize_portfolio
```

#### 12. Commit message
`feat(day7): implement policy authorization wrapper and portfolio audit pipeline`

#### 13. Per-task implementation review checklist
- [ ] Separate semantic fields preserved (I-3)
- [ ] STOP dominance verified (I-2)
- [ ] Non-retroactive budget accounting verified (I-6)
- [ ] Row count invariant verified
- [ ] Byte-identical JSON output verified

#### 14. Exit criteria
Full optimization pipeline runs end-to-end; passes policy safety, audit completeness, and determinism tests.

---

### Task 6: Implement fair greedy baselines

#### 1. Goal
Implement a row-first greedy baseline allocation module to compare against the global portfolio optimizer under identical candidate universes, value inputs, constraints, and policy authorization.

#### 2. Files to create or modify
- `ml/portfolio_greedy.py` (new module)
- `tests/test_portfolio_greedy.py` (new test file)

#### 3. Existing contracts consumed
- `ActionModelBundle`, `PolicyConfig`, `OptimizerConfig`, `PortfolioAllocation`, `build_candidate_universe`, `decide_action`.

#### 4. New public interfaces
```python
def optimize_portfolio_greedy(
    candidate_frame: pd.DataFrame,
    bundle: ActionModelBundle,
    policy: PolicyConfig | None = None,
    config: OptimizerConfig | None = None,
) -> PortfolioAllocation:
    """Run row-first greedy baseline allocation under identical candidate universe and constraints."""
```

#### 5. Baseline Fairness Guarantees
- SAME candidate universe (same eligible rows after pre-screen)
- SAME value inputs (same incremental revenue formula and model probabilities)
- SAME active constraints (`budget_limit`, `human_review_capacity`)
- SAME evaluation split and candidate context frame
- SAME post-allocation policy authorization (`decide_action`)

#### 6. Leakage risks and forbidden fields
Input `candidate_frame` checked against `OPTIMIZER_FORBIDDEN_COLUMNS`.

#### 7. Detailed TDD test cases
- `test_greedy_baseline_validity`: produces valid `PortfolioAllocation` satisfying all constraints.
- `test_greedy_vs_optimizer_equivalence_unconstrained`: greedy and global optimizer produce identical objective value when unconstrained.
- `test_optimizer_outperforms_greedy_constrained_fixture`: crafted fixture with competing multi-arm capacity constraints where global optimizer achieves strictly higher objective value than row-first greedy.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_greedy.py -v
```

#### 9. Minimal implementation sequence
1. Create `ml/portfolio_greedy.py`.
2. Implement row-first allocation logic: sort rows by best incremental value, allocate top available arm per row subject to constraints.
3. Apply post-allocation policy authorization identically to Task 5.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_greedy.py -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_greedy.py
```

#### 12. Commit message
`feat(day7): implement fair row-first greedy baseline solver`

#### 13. Per-task implementation review checklist
- [ ] Baseline uses identical candidate universe, values, and constraints
- [ ] Policy authorization applied identically
- [ ] Fixture test proves global optimizer >= greedy baseline objective value

#### 14. Exit criteria
Greedy baseline implemented cleanly; fair comparison tests green.

---

### Task 7: Implement leakage-safe synthetic outcome evaluation

#### 1. Goal
Implement offline evaluation module joining sealed portfolio allocations to Day 4 synthetic outcomes post-allocation, computing realized recovery and revenue metrics while strictly separating unconfounded randomized-control comparisons from confounded NO_INTERVENTION comparisons.

#### 2. Files to create or modify
- `ml/portfolio_evaluation.py` (new module)
- `tests/test_portfolio_evaluation.py` (new test file)

#### 3. Existing contracts consumed
- `PortfolioAllocation`, `PortfolioSummary`
- Day 4 treatment outcome frame schema (`data/schema.json`)

#### 4. New public interfaces
```python
def evaluate_portfolio_allocation(
    allocation: PortfolioAllocation,
    outcome_frame: pd.DataFrame,
) -> dict:
    """Evaluate sealed portfolio allocation against held-out Day 4 synthetic outcomes."""

def compare_portfolio_to_baseline(
    optimizer_eval: dict,
    greedy_eval: dict,
) -> dict:
    """Compare optimizer evaluation metrics against greedy baseline metrics."""
```

#### 5. Data-flow boundaries & Explicit Comparison Separation
- **Evaluation Isolation:** Allocation structure is sealed BEFORE outcome frame is joined on `attempt_id`.
- **Confounded Comparison:** Comparing authorized intervention rows against NO_INTERVENTION rows within the portfolio allocation. (Labeled: `"CONFOUNDED: optimizer-selected NO_INTERVENTION"`).
- **Unconfounded Comparison:** Comparing authorized intervention rows against randomized CONTROL arm rows from Day 4 test split. (Labeled: `"UNCONFOUNDED: randomized CONTROL arm"`).
- **Label Discipline:** All output fields labeled `"MODEL ESTIMATE"`, `"OBSERVED SIMULATED OUTCOME"`, or `"SIMULATED GROUND TRUTH"`.

#### 6. Leakage risks and forbidden fields
Outcome frame MUST NOT be passed to `optimize_portfolio`. Outcome join occurs ONLY inside `evaluate_portfolio_allocation`.

#### 7. Detailed TDD test cases
- `test_evaluation_isolation`: outcome join requires sealed allocation object.
- `test_missing_attempt_id_alignment_rejection`: mismatched attempt_ids raise `ValueError`.
- `test_confounded_vs_unconfounded_labeling`: verify both comparison blocks exist and carry correct explicit labels.
- `test_optimizer_vs_greedy_delta_calculation`: verify objective and realized outcome deltas calculated accurately.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_evaluation.py -v
```

#### 9. Minimal implementation sequence
1. Create `ml/portfolio_evaluation.py`.
2. Implement `evaluate_portfolio_allocation` joining `allocation.entries` to `outcome_frame` on `attempt_id`.
3. Compute authorized intervention metrics, confounded NO_INTERVENTION metrics, and unconfounded randomized CONTROL metrics.
4. Implement `compare_portfolio_to_baseline`.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_evaluation.py -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_evaluation.py
```

#### 12. Commit message
`feat(day7): implement leakage-safe synthetic outcome evaluation`

#### 13. Per-task implementation review checklist
- [ ] Allocation sealed prior to outcome join (I-5)
- [ ] Confounded and unconfounded comparisons explicitly separated
- [ ] All metrics carry required label discipline

#### 14. Exit criteria
Evaluation module correctly joins outcomes post-allocation and generates complete evaluation dictionary.

---

### Task 8: Run Day 7 GO/NO-GO gates and documentation

#### 1. Goal
Run canonical Day 7 portfolio optimization and evaluation on the canonical test dataset, evaluate all seven pre-registered GO/NO-GO gates (G1-G7), update test suite, and document evidence in `docs/DAY7.md` and `docs/DAY7_RESULTS.md`.

#### 2. Files to create or modify
- `docs/DAY7.md` (new documentation)
- `docs/DAY7_RESULTS.md` (new evidence document)

#### 3. Existing contracts consumed
- `optimize_portfolio`, `optimize_portfolio_greedy`, `evaluate_portfolio_allocation`, `compare_portfolio_to_baseline`, canonical dataset (seed 42).

#### 4. Pre-Registered GO/NO-GO Gates (G1-G7)
- **G1: Feasibility** — Every allocation satisfies all active constraints: no row allocated more than once; `optimizer_budget_allocated <= budget_limit`; `human_review_allocated <= human_review_capacity`; no non-positive-value arm allocated. (Measurable criterion: 0 constraint violations).
- **G2: Determinism** — Two fresh runs on identical canonical inputs produce byte-identical sorted-JSON output. (Measurable criterion: SHA-256 digest match).
- **G3: Policy Safety** — Three crafted STOP contexts (opted-out, fraud risk, hard decline) produce `authorized_action == "STOP"` in all cases. (Measurable criterion: 100% STOP authorization).
- **G4: Leakage Safety** — A hostile candidate frame containing forbidden columns raises `ValueError` before any computation. (Measurable criterion: ValueError raised; 0 outcome leakage).
- **G5: Baseline Comparison** — Portfolio optimizer and greedy baseline run with identical inputs and metrics reported in `DAY7_RESULTS.md`. (Measurable criterion: Both runs complete with identical candidate frames).
- **G6: Synthetic Realized Evaluation** — Optimizer allocation evaluated against held-out Day 4 synthetic outcomes without outcome features entering allocation. (Measurable criterion: Sealed evaluation executed, unconfounded/confounded split reported).
- **G7: Audit Completeness** — Every row has a `PortfolioEntry` with all required fields. Row-count invariant holds: `total_rows == pre_screen_stopped + invalid_prediction_count + optimizer_budget_allocated + no_intervention_count`. (Measurable criterion: 100% row trace completeness and invariant verified).

#### 5. Exact Data-Flow Boundaries
Full pipeline execution over canonical dataset seed 42.

#### 6. Leakage risks and forbidden fields
Strict adherence to `OPTIMIZER_FORBIDDEN_COLUMNS` throughout execution.

#### 7. Detailed TDD test cases
- Integration test running all 7 gates programmatically on canonical data.

#### 8. Exact RED verification command
```powershell
python -m pytest -v
```

#### 9. Minimal implementation sequence
1. Run canonical benchmark script over `data/treatment_outcomes.csv` (dataset seed 42, policy master seed 20260826).
2. Generate G1-G7 gate results.
3. Write `docs/DAY7.md` specifying system architecture and operational guidelines.
4. Write `docs/DAY7_RESULTS.md` documenting canonical gate evidence and metrics.

#### 10. Exact GREEN verification command
```powershell
python -m pytest -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest
```

#### 12. Commit message
`docs: add Day 7 bounded recovery portfolio optimizer results and documentation`

#### 13. Per-task implementation review checklist
- [ ] All 7 GO/NO-GO gates evaluated with empirical evidence
- [ ] Full test suite green
- [ ] Documentation carries explicit MODEL ESTIMATE / OBSERVED SIMULATED OUTCOME labeling

#### 14. Exit criteria
All 7 gates green; full test suite green; documentation complete.

---

## 3. Mapping Invariants to Tasks

| Invariant | Description | Primary Task Location | Test Location |
|-----------|-------------|-----------------------|---------------|
| **I-1** | AI recommends; policy authorizes | Task 2, Task 4, Task 5, Task 8 | `test_portfolio_optimizer.py` |
| **I-2** | STOP remains dominant | Task 2, Task 5, Task 8 (Gate G3) | `test_portfolio_optimizer.py` |
| **I-3** | Separate recommendation & authorization fields | Task 1, Task 5, Task 8 (Gate G7) | `test_portfolio_audit.py` |
| **I-4** | Decision-time inputs only; no outcome leakage | Task 1, Task 2, Task 7, Task 8 (Gate G4) | `test_portfolio_optimizer.py` |
| **I-5** | Allocation fixed before outcome join | Task 7, Task 8 (Gate G6) | `test_portfolio_evaluation.py` |
| **I-6** | Budget NOT retroactively freed on policy override | Task 4, Task 5 | `test_portfolio_optimizer.py` |
| **I-7** | Deterministic & reproducible allocation | Task 3, Task 4, Task 8 (Gate G2) | `test_portfolio_optimizer.py` |

---

## 4. Dependencies & Verification Commands Summary

- **Dependencies:** Pure Python stdlib (`dataclasses`, `ast`, `json`), `numpy`, `pandas`, `pytest`. No external MILP/SAT solver added.
- **Full Test Suite Verification:** `python -m pytest -v`
- **Docker Verification:** `docker compose run --rm app pytest`
