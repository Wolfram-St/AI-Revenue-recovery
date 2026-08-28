# Implementation Plan — Day 7: Hardened Bounded Recovery Portfolio Optimizer

**Date:** 2026-08-27 (Revised 2026-08-28)  
**Status:** Approved Hardened Implementation Plan (Plan Revision Loop)  
**Target Branch:** `feature/day7-optimizer`  
**Base Commit:** `06e6b4d`  
**Design Spec Commit:** `c85df66` (`docs/superpowers/specs/2026-08-27-day7-bounded-recovery-portfolio-optimizer-design.md`)  
**Previous Plan Commit:** `d24526a`  

---

## 1. Overview & Architectural Boundaries

This implementation plan defines the hardened, test-driven, 8-task implementation for Day 7 (**Bounded Recovery Portfolio Optimizer**). Day 7 implements a deterministic, constraint-aware recovery portfolio optimizer operating over decision-time context features and calibrated per-arm probability models.

This revision hardens the plan across nine specific architectural findings:
1. **Real Monetary Budget (`budget_limit_inr`):** Enforces a true INR monetary budget limit alongside human review capacity.
2. **Pre-Allocation vs Post-Allocation Policy:** Separates pre-screening (`PRE_ALLOCATION_POLICY` for context-only terminal STOP rules) from post-allocation authorization (`POST_ALLOCATION_POLICY` for full deterministic policy evaluation).
3. **Cost and Value Accounting:** Formally defines `gross_incremental_value_inr`, `action_cost_inr`, and `net_incremental_value_inr`, preventing double-subtraction.
4. **Invalid Predictions Handling:** Defines explicit handling for NaN/Inf/out-of-bounds probabilities via the `INVALID_PREDICTION` state.
5. **Row Accounting Partition:** Establishes a 4-bucket mutually exclusive and collectively exhaustive row partition (`PRE_SCREEN_STOPPED`, `INVALID_PREDICTION`, `OPTIMIZER_ALLOCATED`, `NO_INTERVENTION`).
6. **Greedy vs Optimizer Equivalence:** Formally documents unconstrained equivalence conditions and separates objective equality tests from tie-broken portfolio identity tests.
7. **Hardened G5 Gate:** Defines G5 with explicit programmatic checks for candidate universe digest matching, constraint equality, and objective superiority (`optimizer_objective >= greedy_objective`).
8. **Held-Out Evaluation Boundary:** Enforces test-split isolation and disjoint attempt ID assertions against training/validation splits.
9. **Deterministic JSON Serialization:** Specifies exact `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False` serialization and paise rounding.

### Preserved Core Pipeline Architecture

```
Action-aware models (Day 5 calibrated per-arm XGBoost bundle)
   │
   ▼
Incremental value estimates (Gross & Net incremental INR relative to CONTROL)
   │
   ▼
Leakage-safe candidate builder (Decision-time whitelist + forbidden column filter + Pre-Screening)
   │
   ▼
Portfolio optimizer (Ranked greedy solver over global (row, arm) pairs with INR monetary budget & HR capacity)
   │
   ▼
Optimizer recommendation (optimizer_recommendation per row)
   │
   ▼
Deterministic policy authorization (decide_action() in recovery/policy.py; POST_ALLOCATION_POLICY)
   │
   ▼
Authorized portfolio (authorized_action per row; STOP dominance preserved)
   │
   ▼
Synthetic outcome evaluation (Offline join to Day 4 simulated ground truth post-allocation on held-out test split)
```

### Non-Negotiable Invariants

- **I-1: AI/model recommends; deterministic policy authorizes.** The optimizer is a recommendation layer. Every allocated action MUST pass through `decide_action()` post-allocation before authorization.
- **I-2: STOP remains dominant.** When `stop_precedence: true` (default in `config/business_rules.yaml`), any matching STOP rule overrides any optimizer recommendation.
- **I-3: Optimizer recommendation and authorized_action must never be collapsed.** They are distinct semantic fields in every audit record.
- **I-4: Realized outcomes must never enter candidate construction or allocation.** Forbidden features and post-decision outcome fields are blocked before optimization.
- **I-5: Allocation must be fully fixed before synthetic outcomes are joined.** Synthetic outcomes enter ONLY during offline evaluation post-allocation.
- **I-6: Budget consumed by an optimizer allocation is NOT retroactively freed when policy overrides the recommendation.** Accounting tracks `budget_allocated_inr` and `post_policy_net_authorized_count` separately.
- **I-7: All allocation must be deterministic and reproducible.** Identical inputs and configuration produce byte-identical JSON traces across environments.

---

## 2. Formal Mathematical Formulation (Option B: Real Monetary Budget)

### 2.1 Variables
For each eligible row $i$ and treated arm $a \in \text{TREATED\_ARMS} = \{\text{RETRY\_NOW}, \text{RETRY\_LATER}, \text{REQUEST\_UPDATE}, \text{HUMAN\_REVIEW}\}$:
$$x(i, a) \in \{0, 1\}$$
where $x(i, a) = 1$ indicates that action $a$ is recommended for row $i$.

### 2.2 Value & Cost Accounting Formulas
For row $i$ with payment amount $\text{amount\_inr}(i)$ and failure category $\text{failure\_category}(i)$:

1. **Risk Penalty:**
   $$\text{risk\_penalty\_inr}(i) = \begin{cases} \text{UNKNOWN\_CATEGORY\_RISK\_FRACTION} \times \text{amount\_inr}(i) & \text{if } \text{failure\_category}(i) = \text{"unknown"} \\ 0.0 & \text{otherwise} \end{cases}$$
   *(where $\text{UNKNOWN\_CATEGORY\_RISK\_FRACTION} = 0.05$, imported from `recovery.scoring`)*

2. **Action Cost:**
   $$\text{action\_cost\_inr}(i, a) = \text{RETRY\_INTERVENTION\_COST\_INR} = 10.0 \text{ INR}$$
   *(Source: `RETRY_INTERVENTION_COST_INR` imported from `recovery.scoring`. Constant per treated arm per Day 6 evidence)*

3. **Gross Incremental Value:**
   $$\text{gross\_incremental\_value\_inr}(i, a) = (\hat{P}_a(i) - \hat{P}_{\text{CONTROL}}(i)) \times \text{amount\_inr}(i) - \text{risk\_penalty\_inr}(i)$$

4. **Net Incremental Value:**
   $$\text{net\_incremental\_value\_inr}(i, a) = \text{gross\_incremental\_value\_inr}(i, a) - \text{action\_cost\_inr}(i, a)$$

### 2.3 Ranking Objective
$$\text{maximize } \sum_{i} \sum_{a} x(i, a) \times \text{net\_incremental\_value\_inr}(i, a)$$

### 2.4 Constraints
1. **At-most-one action per row:**
   $$\sum_{a} x(i, a) \le 1 \quad \forall i$$
2. **Real Monetary Budget Limit (INR):**
   $$\sum_{i} \sum_{a} x(i, a) \times \text{action\_cost\_inr}(i, a) \le \text{budget\_limit\_inr}$$
   *(where $\text{budget\_limit\_inr}$ is a non-negative float/int or `None` if unconstrained)*
3. **HUMAN_REVIEW Capacity Limit:**
   $$\sum_{i} x(i, \text{HUMAN\_REVIEW}) \le \text{human\_review\_capacity}$$
   *(where $\text{human\_review\_capacity}$ is a non-negative integer or `None` if unconstrained)*
4. **Positive Net Value Gate:**
   $$x(i, a) = 0 \quad \text{if } \text{net\_incremental\_value\_inr}(i, a) \le 0.0$$
5. **Pre-Allocation Policy Pre-Screening:**
   $$x(i, a) = 0 \quad \text{if row } i \text{ triggers a terminal context-only STOP rule in } \text{PRE\_ALLOCATION\_POLICY}$$

---

## 3. Explicit Row Accounting Partition

Every input row $i$ submitted to the optimizer pipeline belongs to **EXACTLY ONE** of the following four mutually exclusive and collectively exhaustive buckets in the optimizer state partition:

1. **`PRE_SCREEN_STOPPED`**: Row triggered a context-only terminal STOP rule during `PRE_ALLOCATION_POLICY` pre-screening. (`optimizer_recommendation = "NO_INTERVENTION"`, `no_intervention_reason = "policy_pre_screen: STOP"`, `authorized_action = "STOP"`).
2. **`INVALID_PREDICTION`**: Row produced an invalid model probability prediction (NaN, $\pm\infty$, or $<0.0$ or $>1.0$). (`optimizer_recommendation = "NO_INTERVENTION"`, `no_intervention_reason = "invalid_prediction"`).
3. **`OPTIMIZER_ALLOCATED`**: Row was assigned a positive net-value treated action by the optimizer solver. (`optimizer_recommendation = arm`).
4. **`NO_INTERVENTION`**: Row was eligible for optimization but received no allocation because all candidate arms had net value $\le 0.0$, global monetary budget was exhausted, or HUMAN_REVIEW capacity was exhausted without a positive net fallback arm. (`optimizer_recommendation = "NO_INTERVENTION"`).

### Mandatory Partition Invariants
$$\text{total\_rows} = \text{pre\_screen\_stopped} + \text{invalid\_prediction\_count} + \text{optimizer\_allocated\_count} + \text{no\_intervention\_count}$$

$$\text{PRE\_SCREEN\_STOPPED} \cap \text{INVALID\_PREDICTION} \cap \text{OPTIMIZER\_ALLOCATED} \cap \text{NO\_INTERVENTION} = \emptyset$$

*Note: Policy authorization (`authorized_action`) is evaluated post-allocation across ALL rows and is a separate post-allocation authorization dimension, NOT an optimizer-state bucket.*

---

## 4. Pre-Allocation vs Post-Allocation Policy Architecture

### 4.1 Stage 1: `PRE_ALLOCATION_POLICY` (Optimization Pre-Screening)
- **Scope:** Evaluated BEFORE optimization as an eligibility gate.
- **Rules Evaluated:** ONLY context-only terminal STOP rules whose conditions depend exclusively on decision-time context features (`customer_opted_out`, `fraud_risk`, `failure_category`, `attempt_number`, `amount_inr`, `failure_code`, `issuer_response`).
- **Forbidden in Pre-Screening:** Conditions referencing `recovery_probability`, `optimizer_recommendation`, predicted action probabilities, post-decision outcomes, or treatment assignments MUST NOT be evaluated during pre-screening. (Rules R006, R007, R008 are probability-dependent and are skipped in Stage 1).
- **Outcome:** Pre-screened rows are placed in `PRE_SCREEN_STOPPED`, receive `optimizer_recommendation = "NO_INTERVENTION"`, `no_intervention_reason = "policy_pre_screen: STOP"`, and `authorized_action = "STOP"`. They consume $0.0$ INR budget and 0 HUMAN_REVIEW capacity, and do NOT enter candidate construction.

### 4.2 Stage 2: `POST_ALLOCATION_POLICY` (Final Deterministic Authorization)
- **Scope:** Evaluated AFTER optimizer allocation for EVERY row in the dataset (allocated and unallocated) via `decide_action(policy_context, policy)`.
- **Probability Injection:**
  - Allocated rows (`OPTIMIZER_ALLOCATED`): inject $\hat{P}_{\text{top\_arm}}(i)$ as `recovery_probability`.
  - Unallocated rows (`NO_INTERVENTION`, `PRE_SCREEN_STOPPED`, `INVALID_PREDICTION`): inject $\hat{P}_{\text{CONTROL}}(i)$ as `recovery_probability`.
- **Authorization Boundary:** Returns `authorized_action` and `authorization_reason`. If policy overrides the recommendation (e.g. probability-dependent STOP rule R006 fires), `authorized_action = "STOP"` and `policy_overrode_recommendation = True`.
- **Non-Retroactive Budget Invariant (I-6):** Monetary budget consumed by the optimizer allocation (`budget_allocated_inr`) and HUMAN_REVIEW capacity consumed (`human_review_allocated_count`) are **NOT retroactively freed** or reassigned when policy overrides an allocation to STOP.

---

## 5. Task Breakdown

---

### Task 1: Freeze portfolio interfaces and contracts

#### 1. Goal
Define all dataclasses, frozen structures, domain exceptions, status codes, and constants for Day 7 portfolio optimization, monetary budget accounting, audit traces, and deterministic JSON serialization without implementing solver logic.

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
    budget_limit_inr: float | None = None
    human_review_capacity: int | None = None

class PortfolioOptimizationError(Exception):
    """Raised when portfolio optimization encounters invalid domain inputs or configuration."""
    pass

OPTIMIZER_FORBIDDEN_COLUMNS: frozenset[str]

# ml/portfolio_audit.py
@dataclass(frozen=True)
class PortfolioEntry:
    attempt_id: str
    payment_id: str
    row_index: int
    optimizer_recommendation: str
    no_intervention_reason: str | None
    gross_incremental_value_by_arm: dict[str, float]
    action_cost_by_arm: dict[str, float]
    net_incremental_value_by_arm: dict[str, float]
    selected_gross_incremental_value_inr: float | None
    selected_action_cost_inr: float | None
    selected_net_incremental_value_inr: float | None
    optimizer_sort_rank: int | None
    authorized_action: str
    authorization_reason: str
    matched_rule_id: str | None
    policy_overrode_recommendation: bool

@dataclass(frozen=True)
class PortfolioSummary:
    total_rows: int
    pre_screen_stopped_count: int
    invalid_prediction_count: int
    optimizer_allocated_count: int
    no_intervention_count: int
    eligible_candidate_count: int
    budget_limit_inr: float | None
    budget_allocated_inr: float
    budget_remaining_inr: float | None
    human_review_capacity_limit: int | None
    human_review_allocated_count: int
    post_policy_net_authorized_count: int
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

    def to_json(self) -> str:
        """Return byte-identical sorted JSON using sort_keys=True, separators=(',', ':'), allow_nan=False."""
```

#### 5. Exact data-flow boundaries
Pure type definitions, domain exception definitions, and JSON formatting specification. No DataFrame transforms or model inferences.

#### 6. Leakage risks and forbidden fields
Define `OPTIMIZER_FORBIDDEN_COLUMNS` extending `FORBIDDEN_FEATURES` with post-decision outcome fields:
`recovered`, `recovery_time_hours`, `simulated_recovered`, `simulated_recovered_amount_inr`, `treatment_timestamp`, `outcome_timestamp`, `base_recovery_propensity`, `action_effect_logit`, `propensity_under_assignment`, `assignment_probability`, `arm_source`, `assigned_action`.

#### 7. Detailed TDD test cases
- `test_optimizer_config_validation`: verify negative `budget_limit_inr` or negative `human_review_capacity` raises `ValueError`.
- `test_portfolio_entry_immutability`: verify mutating attributes on `PortfolioEntry` raises `FrozenInstanceError`.
- `test_portfolio_summary_immutability`: verify dataclass immutability on `PortfolioSummary`.
- `test_optimizer_forbidden_columns_completeness`: verify all Day 1 label and Day 4/5 outcome/ground-truth fields are present in `OPTIMIZER_FORBIDDEN_COLUMNS`.
- `test_deterministic_json_serialization_contract`: verify `to_json()` uses `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False`, paise rounding (`round(val, 2)`), and produces byte-identical output across repeated calls.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -v
```

#### 9. Minimal implementation sequence
1. Create `ml/portfolio_optimizer.py` with `OptimizerConfig`, `PortfolioOptimizationError`, and `OPTIMIZER_FORBIDDEN_COLUMNS`.
2. Create `ml/portfolio_audit.py` with `PortfolioEntry`, `PortfolioSummary`, and `PortfolioAllocation`.
3. Implement `to_json()` helper using `json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)` with paise rounding on monetary fields.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py
```

#### 12. Commit message
`feat(day7): freeze portfolio interfaces, monetary budget contracts, and audit schemas`

#### 13. Per-task implementation review checklist
- [ ] All dataclasses marked `frozen=True`
- [ ] `budget_limit_inr` and `budget_allocated_inr` present in `PortfolioSummary`
- [ ] `OPTIMIZER_FORBIDDEN_COLUMNS` strictly defined
- [ ] JSON serialization method uses `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False`
- [ ] Domain exception `PortfolioOptimizationError` defined

#### 14. Exit criteria
Dataclasses import cleanly, enforce immutability, support monetary budget fields, and pass all contract structure unit tests.

---

### Task 2: Build leakage-safe candidate construction

#### 1. Goal
Implement decision-time feature frame validation, forbidden column checking, invalid prediction handling, per-arm probability inference, gross/net incremental revenue calculation, and pre-allocation policy pre-screening (`PRE_ALLOCATION_POLICY`).

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
    gross_incremental_value_inr: float
    action_cost_inr: float
    net_incremental_value_inr: float
    p_hat_arm: float
    p_hat_control: float

def build_candidate_universe(
    candidate_frame: pd.DataFrame,
    bundle: ActionModelBundle,
    policy: PolicyConfig,
) -> tuple[tuple[CandidatePair, ...], dict[str, PortfolioEntry], dict]:
    """Validate frame, handle invalid predictions, run pre-allocation policy pre-screening, and yield positive net-value candidate pairs."""
```

#### 5. Exact data-flow boundaries
- **Inputs:** Decision-time context `pd.DataFrame`, `ActionModelBundle`, `PolicyConfig`.
- **Outputs:** Tuple of eligible positive net-value `CandidatePair` instances, dictionary of pre-screened & invalid `PortfolioEntry` records, and metadata dict.

#### 6. Pre-Allocation Pre-Screening & Leakage Rules
- **Leakage Guard:** Reject any frame containing columns in `OPTIMIZER_FORBIDDEN_COLUMNS` with `ValueError`.
- **Pre-Allocation Policy Pre-Screening:** Evaluates `decide_action` using ONLY context features (no probability injected). Rules R001-R004 (opt-out, fraud, hard decline) fire and place rows into `PRE_SCREEN_STOPPED`. Probability-dependent rules (R006-R008) are NOT evaluated during pre-screening.
- **Invalid Prediction Handling:** Rows with NaN, $\pm\infty$, or probabilities $< 0.0$ or $> 1.0$ are placed into `INVALID_PREDICTION`, produce 0 candidate pairs, consume $0.0$ INR budget, and increment `invalid_prediction_count`.
- **Net Value Gate:** Only candidate pairs with `net_incremental_value_inr > 0.0` enter the candidate universe.

#### 7. Detailed TDD test cases
- `test_forbidden_column_rejection`: input frame with `simulated_recovered` raises `ValueError`.
- `test_missing_required_context`: missing `amount_inr` or `failure_category` raises `ValueError`.
- `test_invalid_prediction_nan_handling`: NaN probability places row in `INVALID_PREDICTION` bucket, 0 budget consumed, 0 candidate pairs created.
- `test_invalid_prediction_out_of_bounds_handling`: probability $1.2$ or $-0.1$ places row in `INVALID_PREDICTION` bucket without silent clipping.
- `test_pre_allocation_context_only_stop_prescreen`: rows matching R001-R004 placed in `PRE_SCREEN_STOPPED` with `authorized_action == "STOP"`.
- `test_pre_allocation_ignores_probability_rules`: probability-dependent rules (R006-R008) do NOT pre-screen rows when probabilities are un-injected.
- `test_gross_vs_net_value_calculation`: verify `gross_incremental_value_inr = (P_hat_a - P_hat_CONTROL) * amount - risk_penalty`, `action_cost_inr = 10.0`, `net_incremental_value_inr = gross - cost`.
- `test_positive_net_value_gate`: candidates with positive gross value but negative net value (`net_incremental_value_inr <= 0.0`) are excluded from candidate pairs.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_candidate_construction -v
```

#### 9. Minimal implementation sequence
1. Implement `_validate_candidate_frame(frame)`.
2. Implement Stage 1 pre-screening calling `decide_action` on context alone.
3. Compute per-arm probabilities via `predict_all_actions` and check finite [0, 1] bounds.
4. Construct `CandidatePair` instances for `arm in TREATED_ARMS` where `net_incremental_value_inr > 0.0`.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_candidate_construction -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py -k test_candidate_construction
```

#### 12. Commit message
`feat(day7): build leakage-safe candidate builder with pre-allocation policy screening and invalid prediction guards`

#### 13. Per-task implementation review checklist
- [ ] Leakage validation runs before any prediction
- [ ] Pre-allocation pre-screening evaluates ONLY context-only STOP rules
- [ ] Invalid predictions (NaN, $\pm\infty$, out-of-bounds) route to `INVALID_PREDICTION` bucket
- [ ] `gross_incremental_value_inr`, `action_cost_inr`, and `net_incremental_value_inr` explicitly separated
- [ ] Candidate gate requires `net_incremental_value_inr > 0.0`

#### 14. Exit criteria
Candidate universe constructed safely; invalid predictions guarded; pre-allocation screening filters context STOPs; net value formula verified.

---

### Task 3: Implement deterministic global-pair ranking

#### 1. Goal
Implement deterministic ranking over candidate (row, arm) pairs using a strict three-level sort key based on net incremental value, guaranteeing 100% reproducible ordering.

#### 2. Files to create or modify
- `ml/portfolio_optimizer.py`
- `tests/test_portfolio_optimizer.py`

#### 3. Existing contracts consumed
- `ARM_ORDER` (`ml/action_model.py`)
- `CandidatePair` (Task 2)

#### 4. New public interfaces
```python
def sort_key_candidate_pair(candidate: CandidatePair) -> tuple[float, str, int]:
    """Return 3-level key: (-net_incremental_value_inr, attempt_id, ARM_ORDER_index)."""

def rank_candidate_pairs(
    candidates: Sequence[CandidatePair]
) -> tuple[CandidatePair, ...]:
    """Sort candidates deterministically by net_incremental_value_inr desc, attempt_id asc, ARM_ORDER asc."""
```

#### 5. Exact data-flow boundaries
Pure function. Input: sequence of `CandidatePair`. Output: sorted tuple of `CandidatePair`. Zero random or clock reads.

#### 6. Leakage risks and forbidden fields
Operates only on `CandidatePair` dataclasses. No ground-truth features present.

#### 7. Detailed TDD test cases
- `test_primary_sort_net_value_descending`: higher `net_incremental_value_inr` ranked earlier.
- `test_secondary_sort_attempt_id_ascending`: equal net values broken deterministically by `attempt_id` ascending.
- `test_tertiary_sort_arm_order_ascending`: same row with equal net values across multiple arms broken by `ARM_ORDER` index (RETRY_NOW < RETRY_LATER < REQUEST_UPDATE < HUMAN_REVIEW).
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
`feat(day7): implement deterministic global-pair ranking based on net incremental value`

#### 13. Per-task implementation review checklist
- [ ] 3-level sort key implemented: net_value desc, attempt_id asc, ARM_ORDER asc
- [ ] Float values sorted descending using negation
- [ ] Permutation invariance verified by test

#### 14. Exit criteria
Candidate pairs ranked deterministically under all input permutations based on net incremental value.

---

### Task 4: Implement deterministic two-constraint portfolio allocation

#### 1. Goal
Implement the constrained portfolio allocation solver enforcing real monetary budget limits (`budget_limit_inr`) and HUMAN_REVIEW capacity bounds, feasibility rules, deterministic tie-breaking, accounting invariants, and infeasible candidate handling.

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
    """Solve constrained portfolio allocation over ranked global candidate pairs under INR monetary budget and HR capacity bounds.
    
    Returns:
      allocated: dict[attempt_id, CandidatePair]
      unallocated_reasons: dict[attempt_id, str]
      solver_metadata: dict (budget_allocated_inr, budget_remaining_inr, hr_allocated_count)
    """
```

#### 5. Explicit Constraint, Feasibility, and Accounting Specification
- **Monetary Budget Constraint (C3):**
  $$\sum_{i} \sum_{a} x(i, a) \times \text{action\_cost\_inr}(i, a) \le \text{budget\_limit\_inr}$$
  When considering candidate pair $(i, a)$, if $\text{budget\_allocated\_inr} + \text{action\_cost\_inr}(i, a) > \text{budget\_limit\_inr}$, candidate pair $(i, a)$ cannot be allocated.
- **HUMAN_REVIEW Capacity Constraint (C4):**
  $$\sum_{i} x(i, \text{HUMAN\_REVIEW}) \le \text{human\_review\_capacity}$$
  When considering candidate pair $(i, \text{HUMAN\_REVIEW})$, if $\text{hr\_allocated\_count} \ge \text{human\_review\_capacity}$, candidate pair $(i, \text{HUMAN\_REVIEW})$ cannot be allocated.
- **Capacity-Exhausted Behavior:**
  - When HUMAN_REVIEW capacity is reached, the scan **DOES NOT STOP**. It skips HUMAN_REVIEW candidate pairs and continues evaluating non-HUMAN_REVIEW candidate pairs for remaining rows.
  - If a row's best arm was HUMAN_REVIEW, it can be allocated its second-best positive net arm if evaluated later in the ranked list.
- **Budget-Exhausted Behavior:**
  - When remaining monetary budget is insufficient for a candidate pair's `action_cost_inr`, that candidate pair is skipped. Scan continues to evaluate other candidates whose `action_cost_inr` fits within remaining budget.
- **At-Most-One Action Per Row (C1):** If row $i$ is already in `allocated`, subsequent candidate pairs for row $i$ are skipped.
- **Objective Maximization:** Sum of `net_incremental_value_inr` over allocated candidate pairs.

#### 6. Leakage risks and forbidden fields
No ground-truth features consumed. Allocation decisions depend strictly on decision-time net incremental values, action costs, and constraint limits.

#### 7. Detailed TDD test cases
- `test_unconstrained_monetary_allocation`: all positive net candidates allocated best arm per row.
- `test_monetary_budget_limit_inr`: `budget_limit_inr = 25.0` INR with `action_cost_inr = 10.0` INR allocates exactly 2 candidates ($20.0$ INR allocated, $5.0$ INR remaining); 3rd candidate ($30.0$ INR required) skipped with `no_intervention_reason = "budget_exhausted"`.
- `test_human_review_capacity_constraint`: `human_review_capacity = 1` limits HUMAN_REVIEW to 1; excess HUMAN_REVIEW candidates fall back to second-best arm or NO_INTERVENTION.
- `test_zero_monetary_budget`: `budget_limit_inr = 0.0` results in 0 allocations, `budget_allocated_inr = 0.0`, status `"budget_exhausted_before_allocation"`.
- `test_zero_human_review_capacity`: `human_review_capacity = 0` excludes HUMAN_REVIEW; other arms allocate normally.
- `test_hand_crafted_portfolio_fixture`: multi-row fixture with hand-calculated expected allocation matching solver output exactly.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_allocation_solver -v
```

#### 9. Minimal implementation sequence
1. Implement `solve_portfolio_allocation` tracking `allocated_rows`, `budget_allocated_inr`, `hr_allocated_count`, and candidate ranks.
2. Ensure skipped HUMAN_REVIEW or budget-exceeding pairs do not block subsequent valid candidate pairs.
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
`feat(day7): implement deterministic two-constraint portfolio allocation solver with real monetary budget`

#### 13. Per-task implementation review checklist
- [ ] Monetary budget constraint C3 strictly enforced in INR
- [ ] HUMAN_REVIEW capacity C4 strictly enforced
- [ ] Skipping HUMAN_REVIEW when capacity full continues loop for non-HR arms
- [ ] At-most-one action per row C1 structurally guaranteed
- [ ] Budget accounting tracks `budget_allocated_inr` and `budget_remaining_inr`

#### 14. Exit criteria
Solver passes all unit tests and hand-calculated multi-constraint portfolio fixtures under real monetary budget limits.

---

### Task 5: Apply deterministic policy authorization without corrupting allocation accounting

#### 1. Goal
Implement post-allocation policy authorization (`POST_ALLOCATION_POLICY`) for every row via `decide_action()`, recording `optimizer_recommendation` and `authorized_action` as distinct fields, enforcing STOP dominance, building the 4-bucket row partition, and maintaining budget accounting invariants without retroactive budget freeing.

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

#### 5. Data-Flow Boundaries, Invariants & Partition Verification
- **I-1:** AI recommends; policy authorizes via `decide_action()`.
- **I-2:** STOP dominance enforced post-allocation across all rows.
- **I-3:** `optimizer_recommendation` and `authorized_action` stored in separate fields; never collapsed.
- **I-6:** Budget consumed by optimizer allocation (`budget_allocated_inr`) and HUMAN_REVIEW capacity (`human_review_allocated_count`) are **NOT retroactively freed** or reassigned when policy overrides a recommendation to STOP. `post_policy_net_authorized_count` tracks net authorized non-STOP interventions.
- **4-Bucket Row Partition Assertion:** Code explicitly asserts:
  `total_rows == pre_screen_stopped_count + invalid_prediction_count + optimizer_allocated_count + no_intervention_count` and verifies that all four bucket sets are pairwise disjoint.

Probability injection conventions:
- For allocated rows (`OPTIMIZER_ALLOCATED`): inject $\hat{P}_{\text{top\_arm}}(i)$ as `recovery_probability`.
- For unallocated rows (`NO_INTERVENTION`, `PRE_SCREEN_STOPPED`, `INVALID_PREDICTION`): inject $\hat{P}_{\text{CONTROL}}(i)$ as `recovery_probability`.

#### 6. Leakage risks and forbidden fields
No post-decision outcome fields allowed in `candidate_frame` or passed to `decide_action`.

#### 7. Detailed TDD test cases
- `test_policy_override_stop`: crafted context where optimizer recommends `RETRY_NOW` but policy condition fires STOP -> `optimizer_recommendation == "RETRY_NOW"`, `authorized_action == "STOP"`, `policy_overrode_recommendation == True`.
- `test_budget_accounting_non_retroactive`: verify `budget_allocated_inr` is NOT decremented when policy overrides a recommendation to STOP.
- `test_summary_row_partition_invariant`: verify 4 buckets are mutually exclusive and collectively exhaustive (`total_rows == pre_screen + invalid + allocated + no_intervention`).
- `test_portfolio_allocation_json_reproducibility`: two identical pipeline calls yield byte-identical `to_json()` output.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -k test_optimize_portfolio -v
```

#### 9. Minimal implementation sequence
1. Implement post-allocation loop calling `decide_action` per row with injected probability.
2. Build `PortfolioEntry` per row including gross, cost, and net values.
3. Compute `PortfolioSummary` metrics (`budget_allocated_inr`, `budget_remaining_inr`, `post_policy_net_authorized_count`, `total_policy_overrides`, `total_policy_stop_overrides`).
4. Validate 4-bucket row partition assertion.
5. Assemble and return `PortfolioAllocation`.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -k test_optimize_portfolio -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -k test_optimize_portfolio
```

#### 12. Commit message
`feat(day7): implement post-allocation policy authorization, audit trace, and row partition assertions`

#### 13. Per-task implementation review checklist
- [ ] Separate semantic fields preserved (I-3)
- [ ] STOP dominance verified (I-2)
- [ ] Non-retroactive monetary budget accounting verified (I-6)
- [ ] 4-bucket row partition invariant explicitly asserted
- [ ] Byte-identical JSON output verified (`sort_keys=True`, `allow_nan=False`)

#### 14. Exit criteria
Full optimization pipeline runs end-to-end; passes policy safety, audit completeness, row partition, and determinism tests.

---

### Task 6: Implement fair greedy baselines

#### 1. Goal
Implement a row-first greedy baseline allocation module to compare against the global portfolio optimizer under identical candidate universes, value inputs, monetary constraints, and policy authorization, proving unconstrained equivalence and constrained superiority.

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
    """Run row-first greedy baseline allocation under identical candidate universe, action costs, and constraints."""
```

#### 5. Baseline Fairness Guarantees
- SAME candidate universe (same pre-screened eligible rows)
- SAME value & cost inputs (same gross, action cost, and net incremental value formulas)
- SAME active constraints (`budget_limit_inr`, `human_review_capacity`)
- SAME evaluation split and candidate context frame
- SAME post-allocation policy authorization (`decide_action`)

#### 6. Proof & Test Specification for Unconstrained Equivalence
- **Unconstrained Equivalence Proof:** Under unconstrained monetary budget (`budget_limit_inr = None`) and unconstrained capacity (`human_review_capacity = None`), each row is independent and both global solver and row-first greedy select $\text{argmax}_a \text{net\_incremental\_value\_inr}(i, a)$ for every row where net value $> 0.0$.
- **Required Invariant:** Total objective value MUST be equal (`optimizer_objective_value_inr == greedy_objective_value_inr`).
- **Tie-Broken Identity:** When tie-breaking is controlled (`ARM_ORDER` index ascending), both solvers produce byte-identical selected portfolios.

#### 7. Detailed TDD test cases
- `test_greedy_baseline_validity`: produces valid `PortfolioAllocation` satisfying all monetary and capacity constraints.
- `test_unconstrained_objective_equivalence`: under unconstrained conditions, optimizer and greedy produce equal total objective value (`optimizer_objective == greedy_objective`).
- `test_unconstrained_deterministic_portfolio_identity`: under unconstrained conditions with controlled ties, optimizer and greedy produce identical selected portfolios.
- `test_optimizer_outperforms_greedy_constrained_fixture`: crafted multi-row fixture with competing monetary budget and HUMAN_REVIEW capacity constraints where global optimizer achieves strictly higher objective value than row-first greedy.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_greedy.py -v
```

#### 9. Minimal implementation sequence
1. Create `ml/portfolio_greedy.py`.
2. Implement row-first allocation logic: sort rows by best positive net incremental value, allocate top available arm per row subject to monetary budget and capacity bounds.
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
`feat(day7): implement fair row-first greedy baseline solver with unconstrained equivalence tests`

#### 13. Per-task implementation review checklist
- [ ] Baseline uses identical candidate universe, action costs, and constraints
- [ ] Policy authorization applied identically
- [ ] Unconstrained objective equality verified
- [ ] Constrained global superiority verified

#### 14. Exit criteria
Greedy baseline implemented cleanly; unconstrained equivalence and constrained optimization tests green.

---

### Task 7: Implement leakage-safe synthetic outcome evaluation

#### 1. Goal
Implement offline evaluation module joining sealed portfolio allocations to Day 4 synthetic outcomes post-allocation on the held-out test split, computing realized recovery and revenue metrics while strictly separating unconfounded randomized-control comparisons from confounded NO_INTERVENTION comparisons.

#### 2. Files to create or modify
- `ml/portfolio_evaluation.py` (new module)
- `tests/test_portfolio_evaluation.py` (new test file)

#### 3. Existing contracts consumed
- `PortfolioAllocation`, `PortfolioSummary`
- Day 4 treatment outcome frame schema (`data/schema.json`)
- Split definitions (`data/splits.py`)

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

#### 5. Held-Out Boundary, Disjoint IDs & Explicit Comparison Separation
- **Held-Out Test Boundary:** Day 7 candidate rows for benchmark evaluation are drawn EXCLUSIVELY from the 15% held-out `test` split (558 randomized test rows or full test split). Test attempt IDs MUST be verified disjoint from model training and validation attempt IDs.
- **Evaluation Isolation:** Allocation structure is sealed BEFORE outcome frame is joined on `attempt_id`.
- **Confounded Comparison:** Comparing authorized intervention rows against NO_INTERVENTION rows within the portfolio allocation. (Labeled: `"CONFOUNDED: optimizer-selected NO_INTERVENTION"`).
- **Unconfounded Comparison:** Comparing authorized intervention rows against randomized CONTROL arm rows from Day 4 test split. (Labeled: `"UNCONFOUNDED: randomized CONTROL arm"`).
- **Label Discipline:** All output fields labeled `"MODEL ESTIMATE"`, `"OBSERVED SIMULATED OUTCOME"`, or `"SIMULATED GROUND TRUTH"`.

#### 6. Leakage risks and forbidden fields
Outcome frame MUST NOT be passed to `optimize_portfolio`. Outcome join occurs ONLY inside `evaluate_portfolio_allocation`.

#### 7. Detailed TDD test cases
- `test_evaluation_isolation`: outcome join requires sealed allocation object.
- `test_held_out_test_split_disjoint_ids`: verify evaluation attempt IDs are disjoint from model training and validation attempt IDs.
- `test_missing_attempt_id_alignment_rejection`: mismatched attempt_ids raise `ValueError`.
- `test_confounded_vs_unconfounded_labeling`: verify both comparison blocks exist and carry correct explicit labels.
- `test_optimizer_vs_greedy_delta_calculation`: verify net objective and realized outcome deltas calculated accurately.

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
`feat(day7): implement leakage-safe synthetic outcome evaluation on held-out test split`

#### 13. Per-task implementation review checklist
- [ ] Allocation sealed prior to outcome join (I-5)
- [ ] Held-out test split attempt IDs verified disjoint from train/val IDs
- [ ] Confounded and unconfounded comparisons explicitly separated
- [ ] All metrics carry required label discipline

#### 14. Exit criteria
Evaluation module correctly joins outcomes post-allocation, enforces test split isolation, and generates complete evaluation dictionary.

---

### Task 8: Run Day 7 GO/NO-GO gates and documentation

#### 1. Goal
Run canonical Day 7 portfolio optimization and evaluation on the canonical held-out test dataset, evaluate all seven pre-registered GO/NO-GO gates (G1-G7) including the hardened G5 gate, update test suite, and document evidence in `docs/DAY7.md` and `docs/DAY7_RESULTS.md`.

#### 2. Files to create or modify
- `docs/DAY7.md` (new documentation)
- `docs/DAY7_RESULTS.md` (new evidence document)

#### 3. Existing contracts consumed
- `optimize_portfolio`, `optimize_portfolio_greedy`, `evaluate_portfolio_allocation`, `compare_portfolio_to_baseline`, canonical dataset (seed 42).

#### 4. Pre-Registered GO/NO-GO Gates (G1-G7 Specification)
- **G1: Feasibility** — Every allocation satisfies all active constraints: no row allocated more than once; `budget_allocated_inr <= budget_limit_inr`; `human_review_allocated_count <= human_review_capacity`; no non-positive-net arm allocated. (Measurable criterion: 0 constraint violations).
- **G2: Determinism** — Two fresh runs on identical canonical inputs produce byte-identical sorted-JSON output via `to_json()`. (Measurable criterion: SHA-256 digest match).
- **G3: Policy Safety** — Three crafted STOP contexts (opted-out, fraud risk, hard decline) produce `authorized_action == "STOP"` in all cases. (Measurable criterion: 100% STOP authorization).
- **G4: Leakage Safety** — A hostile candidate frame containing forbidden columns raises `ValueError` before any computation. (Measurable criterion: ValueError raised; 0 outcome leakage).
- **G5: Meaningful Baseline Comparison** — Programmatically verifies:
  1. Identical candidate-universe digest (same pre-screened eligible rows)
  2. Identical constraint configuration (`budget_limit_inr`, `human_review_capacity`)
  3. Identical action costs (`action_cost_inr = 10.0` INR)
  4. Identical net incremental-value inputs
  5. `optimizer_objective_value_inr >= greedy_objective_value_inr`
  *(Measurable criterion: Fairness checks 1-4 pass 100% AND objective superiority check 5 passes).*
- **G6: Synthetic Realized Evaluation** — Optimizer allocation evaluated against held-out Day 4 synthetic outcomes on the test split without outcome features entering allocation. (Measurable criterion: Sealed evaluation executed on held-out test split, unconfounded/confounded split reported).
- **G7: Audit Completeness & Partition Invariant** — Every row has a `PortfolioEntry` with all required fields. Partition invariant holds: `total_rows == pre_screen_stopped_count + invalid_prediction_count + optimizer_allocated_count + no_intervention_count`. (Measurable criterion: 100% row trace completeness and partition invariant verified).

#### 5. Exact Data-Flow Boundaries
Full pipeline execution over canonical dataset seed 42 (held-out test split).

#### 6. Leakage risks and forbidden fields
Strict adherence to `OPTIMIZER_FORBIDDEN_COLUMNS` throughout execution.

#### 7. Detailed TDD test cases
- Integration test running all 7 gates programmatically on canonical data.

#### 8. Exact RED verification command
```powershell
python -m pytest -v
```

#### 9. Minimal implementation sequence
1. Run canonical benchmark script over `data/treatment_outcomes.csv` (dataset seed 42, policy master seed 20260826, held-out test split).
2. Generate G1-G7 gate results.
3. Write `docs/DAY7.md` specifying system architecture, real monetary budget semantics, and operational guidelines.
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
- [ ] Hardened G5 gate verifies candidate universe digest equality and objective superiority
- [ ] Full test suite green
- [ ] Documentation carries explicit MODEL ESTIMATE / OBSERVED SIMULATED OUTCOME labeling

#### 14. Exit criteria
All 7 gates green; full test suite green; documentation complete.

---

## 6. Mapping Invariants to Tasks

| Invariant | Description | Primary Task Location | Test Location |
|-----------|-------------|-----------------------|---------------|
| **I-1** | AI recommends; policy authorizes | Task 2, Task 4, Task 5, Task 8 | `test_portfolio_optimizer.py` |
| **I-2** | STOP remains dominant | Task 2, Task 5, Task 8 (Gate G3) | `test_portfolio_optimizer.py` |
| **I-3** | Separate recommendation & authorization fields | Task 1, Task 5, Task 8 (Gate G7) | `test_portfolio_audit.py` |
| **I-4** | Decision-time inputs only; no outcome leakage | Task 1, Task 2, Task 7, Task 8 (Gate G4) | `test_portfolio_optimizer.py` |
| **I-5** | Allocation fixed before outcome join | Task 7, Task 8 (Gate G6) | `test_portfolio_evaluation.py` |
| **I-6** | Budget NOT retroactively freed on policy override | Task 4, Task 5 | `test_portfolio_optimizer.py` |
| **I-7** | Deterministic & reproducible allocation | Task 1, Task 3, Task 4, Task 5, Task 8 (Gate G2) | `test_portfolio_optimizer.py`, `test_portfolio_audit.py` |

---

## 7. Dependencies & Verification Commands Summary

- **Dependencies:** Pure Python stdlib (`dataclasses`, `ast`, `json`), `numpy`, `pandas`, `pytest`. No external MILP/SAT solver added.
- **Full Test Suite Verification:** `python -m pytest -v`
- **Docker Verification:** `docker compose run --rm app pytest`
