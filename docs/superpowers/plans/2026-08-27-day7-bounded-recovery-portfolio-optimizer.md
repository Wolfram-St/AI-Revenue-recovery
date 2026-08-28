# Implementation Plan — Day 7: Exact Bounded Recovery Portfolio Optimizer & Hardened Evaluation Gates

**Date:** 2026-08-27 (Revised 2026-08-28 — Plan Hardening Loop Round 2)
**Status:** Approved Hardened Implementation Plan (Exact Integer-Paise Solver & Performance Gate Specification)
**Target Branch:** `feature/day7-optimizer`
**Base Commit:** `06e6b4d`
**Design Spec Commit:** `c85df66` (`docs/superpowers/specs/2026-08-27-day7-bounded-recovery-portfolio-optimizer-design.md`)
**Previous Plan Commit:** `701eac2`

---

## 1. Overview & Architectural Boundaries

This implementation plan defines the test-driven, 8-task implementation for Day 7 (**Bounded Recovery Portfolio Optimizer**). Day 7 implements an **EXACT, deterministic, 2D Dynamic Programming (DP) portfolio optimizer** operating over decision-time context features and calibrated per-arm probability models under real monetary budget limits and human review capacity constraints.

### Key Hardening & Solver Decisions

1. **Exact 2D Dynamic Programming Solver (Option A):** The production Day 7 optimizer implements an exact 2D DP algorithm over rows with state dimensions `(budget_paise, human_review_capacity)`. It guarantees exact objective maximization without heuristic approximation or greedy fallback.
2. **Single Canonical Money Unit (Integer Paise):**
   - **Integer Paise** (`int`) is the single canonical internal representation for solver budget accounting, action costs, and feasibility checks.
   - Public/external interfaces accept monetary inputs in INR floats. Before entering DP state logic, monetary values MUST be validated and converted deterministically to integer paise: $\text{paise} = \text{int}(\text{round}(\text{inr\_val} \times 100))$.
   - Input validation rule: Monetary inputs must be representable to exactly 2 decimal places ($|\text{inr\_val} \times 100 - \text{round}(\text{inr\_val} \times 100)| < 10^{-4}$); malformed fractional paise raise `PortfolioOptimizationError`.
   - Binary floating-point values MUST NEVER be used as DP state indices or budget feasibility checks.
   - Budget feasibility checks are exact integer comparisons over paise (`paise_spent <= budget_limit_paise`). No floating-point epsilon is used for money or budget accounting.
3. **Precise DP Complexity & State vs Transition Distinction:**
   - **State-Space Complexity:** $\mathcal{O}(N \times U_{\text{paise}} \times H_{\text{max}})$ states.
   - **Practical Action-Transition Work:** $\mathcal{O}(N \times K \times U_{\text{paise}} \times H_{\text{max}})$ action evaluations, where $K = 4$ candidate arms.
   - **Canonical Dimensions ($N = 558, U = 50\text{ cost units} = 50,000\text{ paise}, H = 50, K = 4$):**
     - State count (under 10 INR unit discretization): $558 \times 51 \times 51 = 1,451,358$ states.
     - Action-transition evaluations: $558 \times 4 \times 51 \times 51 = 5,805,432$ evaluations.
4. **Proposed Guard Limits & Preflight Memory/Performance Gate:**
   - Proposed hard guard limits: $N_{\text{max}} = 1000$ rows, $U_{\text{max}} = 50,000\text{ paise}$ (500 INR budget at 10 INR step / 50 units), $H_{\text{max}} = 200$ HR slots.
   - Exceeding guard limits raises `PortfolioProblemTooLargeError` fail-closed. No silent greedy fallback exists.
   - Specific runtime claims (e.g. "15-30 ms") are removed; performance and memory usage are implementation-verification concerns. Task 4 and Task 8 mandate an explicit solver preflight benchmark in Docker recording dimensions, state count, transition count, elapsed time, peak memory, solver type, and exactness status.
5. **Redesigned G5 Gates (G5A, G5B, G5C, G5D):** Replaces vague comparison with four explicit sub-gates:
   - **G5A (Fair Comparison Integrity):** Programmatic verification of identical candidate universe digest, constraints, action costs, net values, and eligibility rules between optimizer and greedy baseline.
   - **G5B (Exact Solver Correctness):** Verification against independently known mathematical optima and test-only brute-force recursive enumeration on small fixtures.
   - **G5C (Baseline Comparison):** Verification that `portfolio_objective_delta_inr >= 0.0` (exact DP optimizer never performs worse than greedy).
   - **G5D (Canonical Advantage Reporting):** Deterministic result labeling (`PORTFOLIO_ADVANTAGE_OBSERVED` vs `NO_PORTFOLIO_ADVANTAGE_OBSERVED`) for scientific honesty on canonical runs.
6. **Preserved Invariants & Boundaries:** Real monetary budget (`budget_limit_inr`), HR capacity (`human_review_capacity`), gross/cost/net value separation, 4-bucket row partition (`PRE_SCREEN_STOPPED`, `INVALID_PREDICTION`, `OPTIMIZER_ALLOCATED`, `NO_INTERVENTION`), Stage 1 pre-screening vs Stage 2 authorization, non-retroactive budget accounting, held-out test split isolation, and deterministic JSON serialization (`sort_keys=True`, `allow_nan=False`).

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
Exact Portfolio Optimizer (Exact 2D DP solver over candidate rows with integer paise budget & HR capacity)
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
- **I-6: Budget consumed by an optimizer allocation is NOT retroactively freed when policy overrides the recommendation.** Accounting tracks `budget_allocated_inr` (and `budget_allocated_paise`) and `post_policy_net_authorized_count` separately.
- **I-7: All allocation must be deterministic and reproducible.** Identical inputs and configuration produce byte-identical JSON traces across environments.

---

## 2. Formal Mathematical Formulation & Exact Solver Design

### 2.1 Optimization Variables
For each eligible row $i \in \{1, \dots, N\}$ and treated arm $a \in \text{TREATED\_ARMS} = \{\text{RETRY\_NOW}, \text{RETRY\_LATER}, \text{REQUEST\_UPDATE}, \text{HUMAN\_REVIEW}\}$:
$$x(i, a) \in \{0, 1\}$$
where $x(i, a) = 1$ indicates that action $a$ is recommended for row $i$, and $x(i, a) = 0$ otherwise.

### 2.2 Value & Cost Accounting Formulas (Integer Paise Canonical Contract)
For row $i$ with payment amount $\text{amount\_inr}(i)$ and failure category $\text{failure\_category}(i)$:

1. **Monetary Input Conversion & Validation:**
   Public interfaces receive $\text{amount\_inr}(i)$ and $\text{budget\_limit\_inr}$ as floats.
   - Validation check: $|\text{val} \times 100 - \text{round}(\text{val} \times 100)| < 10^{-4}$. Non-2-decimal floats raise `PortfolioOptimizationError`.
   - Integer paise conversion:
     $$\text{amount\_paise}(i) = \text{int}(\text{round}(\text{amount\_inr}(i) \times 100))$$
     $$\text{budget\_limit\_paise} = \text{int}(\text{round}(\text{budget\_limit\_inr} \times 100))$$

2. **Risk Penalty:**
   $$\text{risk\_penalty\_inr}(i) = \begin{cases} \text{UNKNOWN\_CATEGORY\_RISK\_FRACTION} \times \text{amount\_inr}(i) & \text{if } \text{failure\_category}(i) = \text{"unknown"} \\ 0.0 & \text{otherwise} \end{cases}$$
   *(where $\text{UNKNOWN\_CATEGORY\_RISK\_FRACTION} = 0.05$, imported from `recovery.scoring`)*

3. **Action Cost (Integer Paise Canonical Form):**
   $$\text{action\_cost\_inr}(i, a) = \text{RETRY\_INTERVENTION\_COST\_INR} = 10.00\text{ INR}$$
   $$\text{action\_cost\_paise}(i, a) = \text{int}(\text{round}(\text{action\_cost\_inr}(i, a) \times 100)) = 1000\text{ paise}$$
   *(Source: `RETRY_INTERVENTION_COST_INR` imported from `recovery.scoring`)*

4. **Gross Incremental Value (INR float):**
   $$\text{gross\_incremental\_value\_inr}(i, a) = (\hat{P}_a(i) - \hat{P}_{\text{CONTROL}}(i)) \times \text{amount\_inr}(i) - \text{risk\_penalty\_inr}(i)$$

5. **Net Incremental Value (INR float):**
   $$\text{net\_incremental\_value\_inr}(i, a) = \text{gross\_incremental\_value\_inr}(i, a) - \text{action\_cost\_inr}(i, a)$$

### 2.3 Exact Mathematical Objective
$$\text{maximize } \sum_{i=1}^{N} \sum_{a \in \text{TREATED\_ARMS}} x(i, a) \times \text{net\_incremental\_value\_inr}(i, a)$$

### 2.4 Constraints (Integer-Exact Arithmetic)
1. **At-most-one action per row:**
   $$\sum_{a \in \text{TREATED\_ARMS}} x(i, a) \le 1 \quad \forall i \in \{1, \dots, N\}$$
2. **Real Monetary Budget Limit (Integer Paise Exact):**
   $$\sum_{i=1}^{N} \sum_{a \in \text{TREATED\_ARMS}} x(i, a) \times \text{action\_cost\_paise}(i, a) \le \text{budget\_limit\_paise}$$
   *(Exact integer arithmetic over paise; NO float epsilon used for budget feasibility)*
3. **HUMAN_REVIEW Capacity Limit:**
   $$\sum_{i=1}^{N} x(i, \text{HUMAN\_REVIEW}) \le \text{human\_review\_capacity}$$
4. **Positive Net Value Gate:**
   $$x(i, a) = 0 \quad \text{if } \text{net\_incremental\_value\_inr}(i, a) \le 0.0$$
5. **Pre-Allocation Policy Pre-Screening:**
   $$x(i, a) = 0 \quad \text{if row } i \text{ triggers a terminal context-only STOP rule in } \text{PRE\_ALLOCATION\_POLICY}$$

---

### 2.5 Exact 2D Dynamic Programming (DP) Solver Specification

To solve the 2-constraint 0-1 knapsack allocation problem with mutually exclusive row choices **exactly** and **deterministically**, Day 7 uses a 2D Dynamic Programming solver over candidate rows.

#### Integer Paise State Indexing
- **Monetary Step Discretization:** Since all action costs in the Day 6/7 model are uniform multiples of 10.00 INR (1000 paise), budget state indices can be indexed in integer cost units $u = \text{paise} // 1000$. (In the general case of non-uniform costs, integer paise indices $p \in \{0, \dots, \text{budget\_limit\_paise}\}$ are used).
- Floating-point values MUST NEVER be used as DP state indices or array dimensions.
- Integer budget capacity: $U_{\text{max}} = \text{budget\_limit\_paise} // 1000$.
- HR Capacity Dimension: Integer capacity $H_{\text{max}} = \text{human\_review\_capacity}$.

#### State Definition
For row index $i \in \{0, 1, \dots, N\}$, budget unit index $u \in \{0, 1, \dots, U_{\text{max}}\}$, and HR capacity index $h \in \{0, 1, \dots, H_{\text{max}}\}$:
$$\text{DP}[i, u, h] = \text{maximum total net incremental value (INR float) achievable using a subset of rows } \{1, \dots, i\}$$

#### DP Recurrence Relation
For row $i$ with candidate set $A_i = \{a \in \text{TREATED\_ARMS} \mid \text{net\_incremental\_value\_inr}(i, a) > 0.0\}$:
$$\text{DP}[i, u, h] = \max \Bigg( \text{DP}[i-1, u, h], \max_{a \in A_i : c_{\text{units}}(i,a) \le u, \text{hr}(a) \le h} \left\{ \text{net\_incremental\_value\_inr}(i, a) + \text{DP}[i-1, u - c_{\text{units}}(i,a), h - \text{hr}(a)] \right\} \Bigg)$$
where $\text{hr}(a) = 1$ if $a = \text{HUMAN\_REVIEW}$ else $0$, and $c_{\text{units}}(i,a) = \text{action\_cost\_paise}(i,a) // 1000$.

#### Backpointer Matrix & Deterministic Reconstruction
A compact backpointer array $\text{Backtrack}[i, u, h]$ records the chosen action $a^* \in A_i \cup \{\text{NO\_INTERVENTION}\}$ that achieved $\text{DP}[i, u, h]$.

**Model-Derived Objective Tie Policy:** Float epsilon $\epsilon = 10^{-6}\text{ INR}$ is used ONLY for comparing float model-derived net values during DP updates:
- If $v_{\text{candidate}} > \text{DP}[u, h] + 10^{-6}$: select candidate action $a$.
- If $|v_{\text{candidate}} - \text{DP}[u, h]| \le 10^{-6}$: break tie deterministically:
  1. Prefer choice with smaller total monetary paise spent.
  2. Prefer choice with smaller total HR capacity spent.
  3. Prefer choice with earlier `ARM_ORDER` precedence ($\text{RETRY\_NOW} < \text{RETRY\_LATER} < \text{REQUEST\_UPDATE} < \text{HUMAN\_REVIEW}$).

Traceback from state $(N, U_{\text{max}}, H_{\text{max}})$ deterministically reconstructs the exact optimal allocation vector $x^*(i, a)$. Reporting paise rounding (`round(val, 2)`) applies to presentation/audit trace only and does not alter optimization decision logic.

#### Precise Complexity & Work Disclosures
- **State-Space Complexity:** $\mathcal{O}(N \times U_{\text{max}} \times H_{\text{max}})$ states.
- **Practical Action-Transition Work:** $\mathcal{O}(N \times K \times U_{\text{max}} \times H_{\text{max}})$ action evaluations, where $K = 4$ candidate arms per row.
- **Canonical Run Work ($N = 558, U = 50\text{ units} = 50,000\text{ paise}, H = 50, K = 4$):**
  - Total DP state cells: $558 \times 51 \times 51 = 1,451,358$ states.
  - Action-transition evaluations: $558 \times 4 \times 51 \times 51 = 5,805,432$ evaluations.
- **Memory Implementation Requirement:** To prevent excessive memory allocation in pure Python, implementation must use a 2-layer rolling DP array for value updates and compact integer array storage (`int8` or `uint8` for arm indices) for backtrack metadata.
- **Proposed Guard Limits & Fail-Closed Contract:**
  - Proposed maximum rows: $N_{\text{max}} = 1000$
  - Proposed maximum budget units: $U_{\text{max}} = 500$ ($50,000\text{ paise} = 500.00\text{ INR}$)
  - Proposed maximum HR slots: $H_{\text{max}} = 200$
  *These are proposed hard guard limits pending implementation-time preflight benchmark validation.* If $N > N_{\text{max}}$ or $U > U_{\text{max}}$ or $H > H_{\text{max}}$, the solver raises `PortfolioProblemTooLargeError` naming the exceeded limit. **It DOES NOT silently fall back to greedy or produce an approximate solution.**

---

## 3. Explicit Row Accounting Partition

Every input row $i$ submitted to the optimizer pipeline belongs to **EXACTLY ONE** of the following four mutually exclusive and collectively exhaustive buckets in the optimizer state partition:

1. **`PRE_SCREEN_STOPPED`**: Row triggered a context-only terminal STOP rule during `PRE_ALLOCATION_POLICY` pre-screening. (`optimizer_recommendation = "NO_INTERVENTION"`, `no_intervention_reason = "policy_pre_screen: STOP"`, `authorized_action = "STOP"`).
2. **`INVALID_PREDICTION`**: Row produced an invalid model probability prediction (NaN, $\pm\infty$, or $<0.0$ or $>1.0$). (`optimizer_recommendation = "NO_INTERVENTION"`, `no_intervention_reason = "invalid_prediction"`).
3. **`OPTIMIZER_ALLOCATED`**: Row was assigned a positive net-value treated action by the exact DP solver. (`optimizer_recommendation = arm`).
4. **`NO_INTERVENTION`**: Row was eligible for optimization but received no allocation because all candidate arms had net value $\le 0.0$, or exact DP solver determined NO_INTERVENTION maximized the total portfolio objective under constraints. (`optimizer_recommendation = "NO_INTERVENTION"`).

### Mandatory Partition Invariants
$$\text{total\_rows} = \text{pre\_screen\_stopped\_count} + \text{invalid\_prediction\_count} + \text{optimizer\_allocated\_count} + \text{no\_intervention\_count}$$

$$\text{PRE\_SCREEN\_STOPPED} \cap \text{INVALID\_PREDICTION} \cap \text{OPTIMIZER\_ALLOCATED} \cap \text{NO\_INTERVENTION} = \emptyset$$

---

## 4. Pre-Allocation vs Post-Allocation Policy Architecture

### 4.1 Stage 1: `PRE_ALLOCATION_POLICY` (Optimization Pre-Screening)
- **Scope:** Evaluated BEFORE optimization as an eligibility gate.
- **Rules Evaluated:** ONLY context-only terminal STOP rules whose conditions depend exclusively on decision-time context features (`customer_opted_out`, `fraud_risk`, `failure_category`, `attempt_number`, `amount_inr`, `failure_code`, `issuer_response`).
- **Forbidden in Pre-Screening:** Conditions referencing `recovery_probability`, `optimizer_recommendation`, predicted action probabilities, post-decision outcomes, or treatment assignments MUST NOT be evaluated during pre-screening. (Rules R006, R007, R008 are probability-dependent and are skipped in Stage 1).
- **Outcome:** Pre-screened rows are placed in `PRE_SCREEN_STOPPED`, receive `optimizer_recommendation = "NO_INTERVENTION"`, `no_intervention_reason = "policy_pre_screen: STOP"`, and `authorized_action = "STOP"`. They consume $0\text{ paise}$ budget and 0 HUMAN_REVIEW capacity, and do NOT enter DP candidate construction.

### 4.2 Stage 2: `POST_ALLOCATION_POLICY` (Final Deterministic Authorization)
- **Scope:** Evaluated AFTER exact DP allocation for EVERY row in the dataset (allocated and unallocated) via `decide_action(policy_context, policy)`.
- **Probability Injection:**
  - Allocated rows (`OPTIMIZER_ALLOCATED`): inject $\hat{P}_{\text{top\_arm}}(i)$ as `recovery_probability`.
  - Unallocated rows (`NO_INTERVENTION`, `PRE_SCREEN_STOPPED`, `INVALID_PREDICTION`): inject $\hat{P}_{\text{CONTROL}}(i)$ as `recovery_probability`.
- **Authorization Boundary:** Returns `authorized_action` and `authorization_reason`. If policy overrides recommendation (e.g. probability-dependent STOP rule R006 fires), `authorized_action = "STOP"` and `policy_overrode_recommendation = True`.
- **Non-Retroactive Budget Invariant (I-6):** Monetary budget consumed by the exact DP allocation (`budget_allocated_inr`, `budget_allocated_paise`) and HUMAN_REVIEW capacity consumed (`human_review_allocated_count`) are **NOT retroactively freed** or reassigned when policy overrides an allocation to STOP.

---

## 5. Task Breakdown

---

### Task 1: Freeze portfolio interfaces and contracts

#### 1. Goal
Define all dataclasses, frozen structures, domain exceptions (including `PortfolioProblemTooLargeError`), status codes, and constants for Day 7 portfolio optimization, integer paise budget contracts, exact DP solver parameters, audit traces, and deterministic JSON serialization without implementing solver logic.

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
    max_supported_rows: int = 1000
    max_supported_budget_units: int = 500
    max_supported_hr_capacity: int = 200

class PortfolioOptimizationError(Exception):
    """Raised when portfolio optimization encounters invalid domain inputs or configuration."""
    pass

class PortfolioProblemTooLargeError(PortfolioOptimizationError):
    """Raised when portfolio problem dimensions exceed exact DP solver supported limits."""
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
    selected_action_cost_paise: int | None
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
    budget_limit_paise: int | None
    budget_allocated_inr: float
    budget_allocated_paise: int
    budget_remaining_inr: float | None
    budget_remaining_paise: int | None
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
Pure type definitions, exception declarations, and JSON formatting specification. No DataFrame transforms or model inferences.

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
1. Create `ml/portfolio_optimizer.py` with `OptimizerConfig`, `PortfolioOptimizationError`, `PortfolioProblemTooLargeError`, and `OPTIMIZER_FORBIDDEN_COLUMNS`.
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
`feat(day7): freeze portfolio interfaces, exact DP exceptions, and audit schemas`

#### 13. Per-task implementation review checklist
- [ ] All dataclasses marked `frozen=True`
- [ ] `PortfolioProblemTooLargeError` defined inheriting from `PortfolioOptimizationError`
- [ ] `budget_limit_paise` and `budget_allocated_paise` present in `PortfolioSummary`
- [ ] `OPTIMIZER_FORBIDDEN_COLUMNS` strictly defined
- [ ] JSON serialization method uses `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False`

#### 14. Exit criteria
Dataclasses import cleanly, enforce immutability, support monetary budget fields in integer paise, and pass all contract structure unit tests.

---

### Task 2: Build leakage-safe candidate construction

#### 1. Goal
Implement decision-time feature frame validation, forbidden column checking, invalid prediction handling, per-arm probability inference, gross/net incremental revenue calculation, integer paise conversion validation, and pre-allocation policy pre-screening (`PRE_ALLOCATION_POLICY`).

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
    action_cost_paise: int
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

#### 5. Exact data-flow boundaries & Integer Paise Validation Contract
- **Inputs:** Decision-time context `pd.DataFrame`, `ActionModelBundle`, `PolicyConfig`.
- **Outputs:** Tuple of eligible positive net-value `CandidatePair` instances, dictionary of pre-screened & invalid `PortfolioEntry` records, and metadata dict.
- **Monetary Conversion Validation Contract:** Every monetary input (`amount_inr`, `budget_limit_inr`) is checked:
  $$|\text{val} \times 100 - \text{round}(\text{val} \times 100)| < 10^{-4}$$
  If validation fails (non-2-decimal malformed float), raise `PortfolioOptimizationError`. Do NOT silently round arbitrary malformed precision.
  Valid floats are converted to integer paise: $\text{paise} = \text{int}(\text{round}(\text{val} \times 100))$.

#### 6. Pre-Allocation Pre-Screening & Leakage Rules
- **Leakage Guard:** Reject any frame containing columns in `OPTIMIZER_FORBIDDEN_COLUMNS` with `ValueError`.
- **Pre-Allocation Policy Pre-Screening:** Evaluates `decide_action` using ONLY context features (no probability injected). Rules R001-R004 (opt-out, fraud, hard decline) fire and place rows into `PRE_SCREEN_STOPPED`. Probability-dependent rules (R006-R008) are NOT evaluated during pre-screening.
- **Invalid Prediction Handling:** Rows with NaN, $\pm\infty$, or probabilities $< 0.0$ or $> 1.0$ are placed into `INVALID_PREDICTION`, produce 0 candidate pairs, consume $0\text{ paise}$ budget, and increment `invalid_prediction_count`.
- **Net Value Gate:** Only candidate pairs with `net_incremental_value_inr > 0.0` enter the candidate universe.

#### 7. Detailed TDD test cases
- `test_forbidden_column_rejection`: input frame with `simulated_recovered` raises `ValueError`.
- `test_missing_required_context`: missing `amount_inr` or `failure_category` raises `ValueError`.
- `test_malformed_monetary_float_rejection`: `amount_inr = 10.12345` raises `PortfolioOptimizationError` without silent rounding.
- `test_invalid_prediction_nan_handling`: NaN probability places row in `INVALID_PREDICTION` bucket, 0 budget consumed, 0 candidate pairs created.
- `test_invalid_prediction_out_of_bounds_handling`: probability $1.2$ or $-0.1$ places row in `INVALID_PREDICTION` bucket without silent clipping.
- `test_pre_allocation_context_only_stop_prescreen`: rows matching R001-R004 placed in `PRE_SCREEN_STOPPED` with `authorized_action == "STOP"`.
- `test_pre_allocation_ignores_probability_rules`: probability-dependent rules (R006-R008) do NOT pre-screen rows when probabilities are un-injected.
- `test_gross_vs_net_value_calculation`: verify `gross_incremental_value_inr = (P_hat_a - P_hat_CONTROL) * amount - risk_penalty`, `action_cost_inr = 10.0`, `action_cost_paise = 1000`, `net_incremental_value_inr = gross - cost`.
- `test_positive_net_value_gate`: candidates with positive gross value but negative net value (`net_incremental_value_inr <= 0.0`) are excluded from candidate pairs.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_candidate_construction -v
```

#### 9. Minimal implementation sequence
1. Implement `_validate_candidate_frame(frame)` including 2-decimal paise validation.
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
`feat(day7): build leakage-safe candidate builder with integer paise validation and policy pre-screening`

#### 13. Per-task implementation review checklist
- [ ] Leakage validation runs before any prediction
- [ ] Non-2-decimal float amounts raise `PortfolioOptimizationError`
- [ ] Pre-allocation pre-screening evaluates ONLY context-only STOP rules
- [ ] Invalid predictions (NaN, $\pm\infty$, out-of-bounds) route to `INVALID_PREDICTION` bucket
- [ ] `action_cost_paise` (1000 paise) explicitly stored alongside float INR

#### 14. Exit criteria
Candidate universe constructed safely; malformed floats rejected; pre-allocation screening filters context STOPs; net value formula verified.

---

### Task 3: Implement deterministic global-pair ranking

#### 1. Goal
Implement deterministic ranking over candidate (row, arm) pairs using a strict three-level sort key based on net incremental value, guaranteeing 100% reproducible ordering for diagnostic and baseline comparison functions.

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

### Task 4: Implement exact 2D DP portfolio allocation solver & preflight benchmark gate

#### 1. Goal
Implement the exact 2D Dynamic Programming portfolio allocation solver enforcing real monetary budget limits (`budget_limit_paise`) and HUMAN_REVIEW capacity bounds, integer paise cost discretization, 2-layer rolling DP memory optimization, deterministic tie-breaking, state backpointers, explicit fail-closed guard limits (`PortfolioProblemTooLargeError`), and an explicit solver preflight benchmark requirement.

#### 2. Files to create or modify
- `ml/portfolio_optimizer.py`
- `tests/test_portfolio_optimizer.py`

#### 3. Existing contracts consumed
- `OptimizerConfig`, `CandidatePair`, `PortfolioProblemTooLargeError`, `ARM_ORDER`.

#### 4. New public interfaces
```python
def solve_portfolio_allocation(
    candidates: tuple[CandidatePair, ...],
    eligible_attempt_ids: set[str],
    config: OptimizerConfig,
) -> tuple[dict[str, CandidatePair], dict[str, str], dict]:
    """Solve constrained portfolio allocation exactly via 2D Dynamic Programming over candidate rows using integer paise budget indices.
    
    Returns:
      allocated: dict[attempt_id, CandidatePair]
      unallocated_reasons: dict[attempt_id, str]
      solver_metadata: dict (budget_allocated_inr, budget_allocated_paise, budget_remaining_paise, hr_allocated_count, solver_type="exact_dp_2d", preflight_stats)
    """

def run_solver_preflight_benchmark(
    candidates: tuple[CandidatePair, ...],
    config: OptimizerConfig,
) -> dict:
    """Execute preflight benchmark recording dimensions, DP state count, transition count, elapsed time, peak memory, and exactness status."""
```

#### 5. Exact DP Algorithm & Constraint Specification
- **Oversized Problem Guard:** Check $N \le \text{max\_supported\_rows}$ (1000), $U \le \text{max\_supported\_budget\_units}$ (500), $H \le \text{max\_supported\_hr\_capacity}$ (200). If exceeded, raise `PortfolioProblemTooLargeError`. No silent greedy fallback!
- **Discretization & Integer Arithmetic:** Budget limit $B_{\text{paise}} = \text{int}(\text{round}(\text{budget\_limit\_inr} \times 100))$. Index units $U = B_{\text{paise}} // 1000$. Integer arithmetic is used for all budget feasibility checks (`cost_paise <= budget_remaining_paise`). NO float epsilon is used for budget feasibility!
- **State Table & Memory Optimization:**
  - Value table: 2-row rolling array `dp_prev[u, h]` and `dp_curr[u, h]` of float net incremental values to minimize memory overhead.
  - Backpointer table: Compact 3D NumPy array `backtrack[i, u, h]` storing selected arm index ($0 \dots 4$).
- **Row Recurrence:** Iterate candidate rows $i = 1 \dots N$:
  - Options for row $i$: NO_INTERVENTION (value 0.0, cost 0, HR 0) OR any positive candidate arm $a \in A_i$ (net value $v_a$, cost units $c_a$, HR $h_a$).
  - Update `dp_curr` backwards over $u \in [U, c_a]$, $h \in [H, h_a]$:
    $$v_{\text{candidate}} = v_a + \text{dp\_prev}[u - c_a, h - h_a]$$
    If $v_{\text{candidate}} > \text{dp\_curr}[u, h] + 10^{-6}$: select action $a$.
    If $|v_{\text{candidate}} - \text{dp\_curr}[u, h]| \le 10^{-6}$: break tie deterministically (prefer lower $u$, then lower $h$, then earlier `ARM_ORDER`).
- **Traceback:** Trace from `backtrack[N, U_final, H_final]` to reconstruct exact optimal allocation vector $x^*(i, a)$.
- **Preflight Benchmark Recording:** Collects dimensions $N, U, H, K$, state count ($N \times (U+1) \times (H+1)$), transition count ($N \times K \times (U+1) \times (H+1)$), elapsed seconds, peak MB memory, solver type `exact_dp_2d`, and exactness status `EXACT_DP_OPTIMAL`.

#### 6. Leakage risks and forbidden fields
No ground-truth features consumed. Allocation decisions depend strictly on decision-time net incremental values, action costs, and constraint limits.

#### 7. Detailed TDD Test Cases (16 Exact Correctness Tests)
1. `test_exact_optimum_tiny_enumerable_fixture`: verify exact DP objective equals test-only brute-force recursive enumerator optimum on a 3-row, 2-arm fixture.
2. `test_greedy_suboptimal_exact_dp_superior_fixture`: crafted fixture where global highest-value greedy picks a high-value HR item that exhausts HR capacity, missing two medium-value HR items with higher combined net sum -> exact DP achieves strictly higher objective than greedy.
3. `test_monetary_and_hr_constraint_interaction`: fixture testing combined binding monetary budget and HR capacity limits.
4. `test_at_most_one_action_per_row`: structural guarantee verified; no attempt_id allocated twice.
5. `test_paise_boundary_monetary_exactness`: monetary budget enforced exactly at integer paise boundaries using integer comparisons (no float epsilon).
6. `test_hr_capacity_exactness`: HR capacity limit enforced exactly.
7. `test_exact_dp_deterministic_tie_breaking`: identical input frame produces byte-identical DP allocation across 100 repeated runs.
8. `test_deterministic_portfolio_reconstruction`: traceback produces identical selected candidate pairs regardless of candidate array insertion order.
9. `test_all_selected_pairs_have_positive_net_value`: no candidate with `net_incremental_value_inr <= 0.0` is allocated.
10. `test_non_positive_value_exclusion_reasons`: zero and negative net candidates excluded with reason `"non_positive_net_value"`.
11. `test_unconstrained_mathematical_optimum`: unconstrained configuration selects `argmax_a net_value` per row.
12. `test_oversized_problem_raises_portfolio_problem_too_large_error`: input exceeding $N=1000$ or $U=500$ raises `PortfolioProblemTooLargeError`.
13. `test_no_silent_approximation`: verify DP table evaluates exact values without float rounding drift.
14. `test_no_silent_greedy_fallback`: verify solver metadata explicitly records `solver_type: "exact_dp_2d"`.
15. `test_brute_force_enumerator_validation`: test-only recursive brute-force enumerator validates exact DP output across 50 random small candidate frames ($N \le 12$).
16. `test_preflight_benchmark_recording`: verify `run_solver_preflight_benchmark` records $N, U, H, K$, state count ($1,451,358$), transition count ($5,805,432$), elapsed time, and memory metrics cleanly.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_exact_dp -v
```

#### 9. Minimal implementation sequence
1. Implement `_test_brute_force_enumerate` helper in `tests/test_portfolio_optimizer.py`.
2. Implement `solve_portfolio_allocation` using 2-layer rolling DP array with integer paise/unit budget indices in `ml/portfolio_optimizer.py`.
3. Implement `run_solver_preflight_benchmark` collecting performance and memory trace data.
4. Implement deterministic tie-breaking and backpointer traceback reconstruction.
5. Add size guard raising `PortfolioProblemTooLargeError`.

#### 10. Exact GREEN verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py -k test_exact_dp -v
```

#### 11. Docker verification command
```powershell
docker compose run --rm app pytest tests/test_portfolio_optimizer.py -k test_exact_dp
```

#### 12. Commit message
`feat(day7): implement exact 2D dynamic programming portfolio allocation solver with integer paise budget indices and preflight benchmark`

#### 13. Per-task implementation review checklist
- [ ] Exact 2D DP algorithm implemented (Option A)
- [ ] Integer paise used for all monetary budget indices and comparisons (no float epsilon for money)
- [ ] HR capacity dimension enforced exactly
- [ ] 2-layer rolling DP array used to optimize memory
- [ ] Preflight benchmark helper records dimensions, state count, transition count, time, and memory
- [ ] `PortfolioProblemTooLargeError` raised on oversized inputs (no silent fallback)
- [ ] Test-only brute-force recursive enumerator confirms exact optimum

#### 14. Exit criteria
Exact DP solver passes all 16 correctness unit tests, outperforms greedy on sub-optimal greedy fixtures, matches brute-force enumerator on small fixtures, and executes preflight benchmark cleanly.

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
    """Run full bounded portfolio optimization (exact 2D DP) and policy authorization pipeline."""
```

#### 5. Data-Flow Boundaries, Invariants & Partition Verification
- **I-1:** AI recommends; policy authorizes via `decide_action()`.
- **I-2:** STOP dominance enforced post-allocation across all rows.
- **I-3:** `optimizer_recommendation` and `authorized_action` stored in separate fields; never collapsed.
- **I-6:** Budget consumed by optimizer allocation (`budget_allocated_inr`, `budget_allocated_paise`) and HUMAN_REVIEW capacity (`human_review_allocated_count`) are **NOT retroactively freed** or reassigned when policy overrides a recommendation to STOP. `post_policy_net_authorized_count` tracks net authorized non-STOP interventions.
- **4-Bucket Row Partition Assertion:** Code explicitly asserts:
  `total_rows == pre_screen_stopped_count + invalid_prediction_count + optimizer_allocated_count + no_intervention_count` and verifies that all four bucket sets are pairwise disjoint.

Probability injection conventions:
- For allocated rows (`OPTIMIZER_ALLOCATED`): inject $\hat{P}_{\text{top\_arm}}(i)$ as `recovery_probability`.
- For unallocated rows (`NO_INTERVENTION`, `PRE_SCREEN_STOPPED`, `INVALID_PREDICTION`): inject $\hat{P}_{\text{CONTROL}}(i)$ as `recovery_probability`.

#### 6. Leakage risks and forbidden fields
No post-decision outcome fields allowed in `candidate_frame` or passed to `decide_action`.

#### 7. Detailed TDD test cases
- `test_policy_override_stop`: crafted context where optimizer recommends `RETRY_NOW` but policy condition fires STOP -> `optimizer_recommendation == "RETRY_NOW"`, `authorized_action == "STOP"`, `policy_overrode_recommendation == True`.
- `test_budget_accounting_non_retroactive`: verify `budget_allocated_inr` and `budget_allocated_paise` are NOT decremented when policy overrides a recommendation to STOP.
- `test_summary_row_partition_invariant`: verify 4 buckets are mutually exclusive and collectively exhaustive (`total_rows == pre_screen + invalid + allocated + no_intervention`).
- `test_portfolio_allocation_json_reproducibility`: two identical pipeline calls yield byte-identical `to_json()` output.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_optimizer.py tests/test_portfolio_audit.py -k test_optimize_portfolio -v
```

#### 9. Minimal implementation sequence
1. Implement post-allocation loop calling `decide_action` per row with injected probability.
2. Build `PortfolioEntry` per row including gross, cost (INR and paise), and net values.
3. Compute `PortfolioSummary` metrics (`budget_allocated_inr`, `budget_allocated_paise`, `budget_remaining_inr`, `budget_remaining_paise`, `post_policy_net_authorized_count`, `total_policy_overrides`, `total_policy_stop_overrides`).
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
- [ ] Non-retroactive monetary budget accounting verified in INR and paise (I-6)
- [ ] 4-bucket row partition invariant explicitly asserted
- [ ] Byte-identical JSON output verified (`sort_keys=True`, `allow_nan=False`)

#### 14. Exit criteria
Full optimization pipeline runs end-to-end; passes policy safety, audit completeness, row partition, and determinism tests.

---

### Task 6: Implement fair greedy baselines

#### 1. Goal
Implement a row-first greedy baseline allocation module to compare against the exact global portfolio optimizer under identical candidate universes, value inputs, monetary constraints (in integer paise), and policy authorization, proving unconstrained equivalence and constrained optimizer superiority.

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
- SAME value & cost inputs (same gross, action cost in paise, and net incremental value formulas)
- SAME active constraints (`budget_limit_inr` / `budget_limit_paise`, `human_review_capacity`)
- SAME evaluation split and candidate context frame
- SAME post-allocation policy authorization (`decide_action`)

#### 6. Proof & Test Specification for Unconstrained Equivalence
- **Unconstrained Equivalence Proof:** Under unconstrained monetary budget (`budget_limit_inr = None`) and unconstrained capacity (`human_review_capacity = None`), each row is independent and both exact DP solver and row-first greedy select $\text{argmax}_a \text{net\_incremental\_value\_inr}(i, a)$ for every row where net value $> 0.0$.
- **Required Invariant:** Total objective value MUST be equal (`optimizer_objective_value_inr == greedy_objective_value_inr`).
- **Tie-Broken Identity:** When tie-breaking is controlled (`ARM_ORDER` index ascending), both solvers produce byte-identical selected portfolios.

#### 7. Detailed TDD test cases
- `test_greedy_baseline_validity`: produces valid `PortfolioAllocation` satisfying all monetary (paise) and capacity constraints.
- `test_unconstrained_objective_equivalence`: under unconstrained conditions, exact DP optimizer and greedy produce equal total objective value (`optimizer_objective == greedy_objective`).
- `test_unconstrained_deterministic_portfolio_identity`: under unconstrained conditions with controlled ties, exact DP optimizer and greedy produce identical selected portfolios.
- `test_exact_dp_outperforms_greedy_constrained_fixture`: crafted multi-row fixture with competing monetary budget and HUMAN_REVIEW capacity constraints where exact DP global optimizer achieves strictly higher objective value than row-first greedy.

#### 8. Exact RED verification command
```powershell
python -m pytest tests/test_portfolio_greedy.py -v
```

#### 9. Minimal implementation sequence
1. Create `ml/portfolio_greedy.py`.
2. Implement row-first allocation logic: sort rows by best positive net incremental value, allocate top available arm per row subject to monetary budget (paise) and capacity bounds.
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
- [ ] Baseline uses identical candidate universe, action costs (paise), and constraints
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

### Task 8: Run Day 7 GO/NO-GO gates, solver preflight benchmark, and documentation

#### 1. Goal
Run canonical Day 7 portfolio optimization and evaluation on the canonical held-out test dataset, execute solver preflight performance/memory benchmark, evaluate all seven pre-registered GO/NO-GO gates (G1-G7) including the redesigned G5 sub-gates (G5A, G5B, G5C, G5D), update test suite, and document evidence in `docs/DAY7.md` and `docs/DAY7_RESULTS.md`.

#### 2. Files to create or modify
- `docs/DAY7.md` (new documentation)
- `docs/DAY7_RESULTS.md` (new evidence document)

#### 3. Existing contracts consumed
- `optimize_portfolio`, `optimize_portfolio_greedy`, `evaluate_portfolio_allocation`, `compare_portfolio_to_baseline`, `run_solver_preflight_benchmark`, canonical dataset (seed 42).

#### 4. Pre-Registered GO/NO-GO Gates (G1-G7 Specification with G5 Redesign)
- **G1: Feasibility** — Every allocation satisfies all active constraints: no row allocated more than once; `budget_allocated_paise <= budget_limit_paise`; `human_review_allocated_count <= human_review_capacity`; no non-positive-net arm allocated. (Measurable criterion: 0 constraint violations).
- **G2: Determinism** — Two fresh runs on identical canonical inputs produce byte-identical sorted-JSON output via `to_json()`. (Measurable criterion: SHA-256 digest match).
- **G3: Policy Safety** — Three crafted STOP contexts (opted-out, fraud risk, hard decline) produce `authorized_action == "STOP"` in all cases. (Measurable criterion: 100% STOP authorization).
- **G4: Leakage Safety** — A hostile candidate frame containing forbidden columns raises `ValueError` before any computation. (Measurable criterion: ValueError raised; 0 outcome leakage).
- **G5: Baseline Comparison & Solver Correctness (Redesigned Sub-Gates):**
  - **G5A (Fair Comparison Integrity):** Programmatically verifies identical candidate-universe digest, constraint config, action costs, net value inputs, positive-value eligibility, invalid prediction handling, and pre-allocation filtering between optimizer and greedy baseline. (Measurable criterion: Fairness checks pass 100%).
  - **G5B (Exact Solver Correctness):** Programmatically verifies exact DP solver objective equals known brute-force optimum on small test fixtures. (Measurable criterion: 0 deviation from mathematical optimum).
  - **G5C (Baseline Comparison):** Programmatically verifies `portfolio_objective_delta_inr = optimizer_objective - greedy_objective >= 0.0`. (Measurable criterion: Exact DP never performs worse than greedy).
  - **G5D (Canonical Advantage Reporting):** Programmatically classifies canonical test run delta into deterministic labels: `PORTFOLIO_ADVANTAGE_OBSERVED` ($\text{delta\_inr} > 0$) or `NO_PORTFOLIO_ADVANTAGE_OBSERVED` ($\text{delta\_inr} == 0$). Zero canonical advantage is documented scientifically without calling equality an improvement.
- **G6: Synthetic Realized Evaluation** — Optimizer allocation evaluated against held-out Day 4 synthetic outcomes on the test split without outcome features entering allocation. (Measurable criterion: Sealed evaluation executed on held-out test split, unconfounded/confounded split reported).
- **G7: Audit Completeness & Partition Invariant** — Every row has a `PortfolioEntry` with all required fields. Partition invariant holds: `total_rows == pre_screen_stopped_count + invalid_prediction_count + optimizer_allocated_count + no_intervention_count`. (Measurable criterion: 100% row trace completeness and partition invariant verified).

#### 5. Exact Data-Flow Boundaries & Preflight Benchmark Requirement
Full pipeline execution over canonical dataset seed 42 (held-out test split). Solver preflight benchmark MUST be executed and recorded in `DAY7_RESULTS.md` documenting:
- Input dimensions $N=558, U=50, H=50, K=4$
- State count: $1,451,358$ states
- Action-transition evaluations: $5,805,432$ evaluations
- Solver elapsed time & peak memory in Docker
- Exactness status `EXACT_DP_OPTIMAL`

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
2. Run `run_solver_preflight_benchmark` recording performance and memory evidence.
3. Generate G1-G7 gate results (including G5A-G5D sub-gates).
4. Write `docs/DAY7.md` specifying system architecture, exact integer paise 2D DP solver semantics, real monetary budget rules, and operational guidelines.
5. Write `docs/DAY7_RESULTS.md` documenting canonical gate evidence, preflight performance/memory metrics, G5D advantage classification, and evaluation tables.

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
- [ ] Solver preflight performance and memory benchmark executed and recorded
- [ ] Redesigned G5 gate evaluates G5A, G5B, G5C, G5D explicitly
- [ ] Scientific honesty maintained for G5D (zero delta labeled `NO_PORTFOLIO_ADVANTAGE_OBSERVED`)
- [ ] Full test suite green
- [ ] Documentation carries explicit MODEL ESTIMATE / OBSERVED SIMULATED OUTCOME labeling

#### 14. Exit criteria
All 7 gates green; preflight benchmark documented; full test suite green; documentation complete.

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
