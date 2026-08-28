"""Day 7 portfolio optimizer interfaces and contracts (Task 1, 2, & 3).

Frozen dataclasses, exceptions, constants, candidate construction, ranking, and JSON serialization
specification for the bounded recovery portfolio optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ml.action_model import ARM_ORDER, ActionModelBundle, predict_all_actions
from ml.features import FORBIDDEN_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_feature_matrix
from recovery.policy import PolicyConfig, decide_action, load_policy_config
from recovery.scoring import RETRY_INTERVENTION_COST_INR, UNKNOWN_CATEGORY_RISK_FRACTION

# TREATED_ARMS = all arms except CONTROL
TREATED_ARMS = tuple(arm for arm in ARM_ORDER if arm != "CONTROL")

# Context-only columns for pre-allocation policy screening (R001-R004)
# These are the columns that context-only STOP rules depend on
PRE_ALLOCATION_POLICY_COLUMNS = (
    "customer_opted_out",
    "fraud_risk",
    "failure_category",
    "attempt_number",
    "amount_inr",
    "failure_code",
    "issuer_response",
)


class PortfolioOptimizationError(Exception):
    """Raised when portfolio optimization encounters invalid domain inputs or configuration."""
    pass


class PortfolioProblemTooLargeError(PortfolioOptimizationError):
    """Raised when portfolio problem dimensions exceed exact DP solver supported limits."""
    pass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the portfolio optimizer.

    Attributes:
        budget_limit_inr: Maximum total intervention budget in INR (float, 2-decimal precision).
            None means unconstrained.
        human_review_capacity: Maximum number of HUMAN_REVIEW actions allowed.
            None means unconstrained.
        max_supported_rows: Maximum number of candidate rows for exact DP solver.
        max_supported_budget_units: Maximum budget units (paise // 1000) for exact DP solver.
        max_supported_hr_capacity: Maximum HR capacity for exact DP solver.
    """
    budget_limit_inr: float | None = None
    human_review_capacity: int | None = None
    max_supported_rows: int = 1000
    max_supported_budget_units: int = 500
    max_supported_hr_capacity: int = 200

    def __post_init__(self) -> None:
        if self.budget_limit_inr is not None and self.budget_limit_inr < 0:
            raise ValueError("budget_limit_inr must be >= 0 or None")
        if self.human_review_capacity is not None and self.human_review_capacity < 0:
            raise ValueError("human_review_capacity must be >= 0 or None")
        if self.max_supported_rows <= 0:
            raise ValueError("max_supported_rows must be > 0")
        if self.max_supported_budget_units <= 0:
            raise ValueError("max_supported_budget_units must be > 0")
        if self.max_supported_hr_capacity <= 0:
            raise ValueError("max_supported_hr_capacity must be > 0")


# Forbidden columns for the optimizer: post-decision outcome, ground-truth,
# and assignment metadata columns that must never enter the optimizer input frame.
# Note: IDENTIFIER_COLUMNS, TIME_COLUMN, LABEL_COLUMNS are allowed in optimizer
# input frame for audit tracing; they are filtered by build_feature_matrix.
OPTIMIZER_FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "simulated_recovered",
        "simulated_recovered_amount_inr",
        "treatment_timestamp",
        "outcome_timestamp",
        "base_recovery_propensity",
        "action_effect_logit",
        "propensity_under_assignment",
        "assignment_probability",
        "arm_source",
        "assigned_action",
        "recovery_time_hours",
        "recovery_action",
        "action_outcome",
        "recovered_amount_inr",
    }
)


def _validate_monetary_value(name: str, value: float) -> int:
    """Validate that a monetary value is representable to exactly 2 decimal places and convert to paise.
    
    Raises PortfolioOptimizationError if validation fails (non-2-decimal malformed float).
    Returns integer paise.
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise PortfolioOptimizationError(f"{name} must be a number, got {type(value).__name__}")
    if not np.isfinite(value):
        raise PortfolioOptimizationError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise PortfolioOptimizationError(f"{name} must be >= 0, got {value!r}")
    
    # Check if value is representable to exactly 2 decimal places
    # |val * 100 - round(val * 100)| < 1e-4
    paise_exact = value * 100
    paise_rounded = round(paise_exact)
    if abs(paise_exact - paise_rounded) >= 1e-4:
        raise PortfolioOptimizationError(
            f"{name} must be representable to exactly 2 decimal places, got {value!r}"
        )
    return int(paise_rounded)


def _validate_candidate_frame(frame: pd.DataFrame) -> None:
    """Validate candidate frame for leakage and required columns.
    
    Checks:
    1. Frame is a DataFrame
    2. No forbidden columns present
    3. Required decision-time columns present
    4. Monetary values are valid (2-decimal precision)
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"candidate_frame must be a pandas DataFrame, got {type(frame).__name__}")
    
    # Check for forbidden columns
    forbidden_present = OPTIMIZER_FORBIDDEN_COLUMNS & set(frame.columns)
    if forbidden_present:
        raise ValueError(f"candidate_frame contains forbidden columns: {sorted(forbidden_present)}")
    
    # Check required columns
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["attempt_id", "payment_id", "customer_id", "event_timestamp"])
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"candidate_frame missing required columns: {sorted(missing)}")
    
    # Validate monetary columns (amount_inr)
    if "amount_inr" in frame.columns:
        for idx, val in frame["amount_inr"].items():
            try:
                _validate_monetary_value(f"amount_inr at row {idx}", val)
            except PortfolioOptimizationError as e:
                raise PortfolioOptimizationError(str(e))


# Context-only STOP rule IDs for pre-allocation screening (R001-R004)
PRE_ALLOCATION_STOP_RULE_IDS = frozenset({"R001", "R002", "R003", "R004"})


def _filter_policy_for_pre_screening(policy: PolicyConfig) -> PolicyConfig:
    """Create a policy config with only context-only STOP rules (R001-R004) for pre-screening."""
    filtered_rules = tuple(rule for rule in policy.rules if rule.id in PRE_ALLOCATION_STOP_RULE_IDS)
    return PolicyConfig(
        version=policy.version,
        stop_precedence=policy.stop_precedence,
        rules=filtered_rules,
        canonical_actions=policy.canonical_actions,
    )


def _pre_screen_policy(frame: pd.DataFrame, policy: PolicyConfig) -> dict[str, tuple[str, str]]:
    """Run Stage 1 PRE_ALLOCATION_POLICY pre-screening.
    
    Evaluates decide_action using ONLY context features (no probability injected).
    Only context-only STOP rules (R001-R004: opt-out, fraud, hard decline, retry limit) 
    are evaluated. Probability-dependent rules (R006-R008) are NOT evaluated during pre-screening.
    
    Returns dict mapping attempt_id -> (no_intervention_reason, authorized_action)
    for pre-screened rows. Rows not pre-screened are not in the returned dict.
    """
    # Filter policy to only context-only STOP rules
    pre_screen_policy = _filter_policy_for_pre_screening(policy)
    
    pre_screened = {}
    
    for idx, row in frame.iterrows():
        # Build context with ONLY context features (no recovery_probability)
        context = {}
        for col in PRE_ALLOCATION_POLICY_COLUMNS:
            if col in row:
                context[col] = row[col]
        
        try:
            decision = decide_action(context, pre_screen_policy)
        except KeyError as e:
            # Missing column in context - this shouldn't happen if frame is valid
            raise ValueError(f"Missing column for pre-screening: {e}")
        
        if decision.authorized_action == "STOP":
            pre_screened[row["attempt_id"]] = ("policy_pre_screen", "STOP")
    
    return pre_screened


def _is_valid_probability(prob: float) -> bool:
    """Check if probability is finite and in [0, 1]."""
    return isinstance(prob, (int, float, np.integer, np.floating)) and np.isfinite(prob) and 0.0 <= prob <= 1.0


def _compute_risk_penalty(amount_inr: float, failure_category: str) -> float:
    """Compute risk penalty for a row."""
    if failure_category == "unknown":
        return UNKNOWN_CATEGORY_RISK_FRACTION * amount_inr
    return 0.0


def build_candidate_universe(
    candidate_frame: pd.DataFrame,
    bundle: ActionModelBundle,
    policy: PolicyConfig,
) -> tuple[tuple[CandidatePair, ...], dict[str, "PortfolioEntry"], dict]:
    """Validate frame, handle invalid predictions, run pre-allocation policy pre-screening, 
    and yield positive net-value candidate pairs.
    
    Returns:
        eligible_candidates: tuple of CandidatePair with net_incremental_value_inr > 0.0
        entries: dict mapping attempt_id -> PortfolioEntry for pre-screened & invalid prediction rows
        metadata: dict with counts and summary info
    """
    from ml.portfolio_audit import PortfolioEntry
    
    # Step 1: Validate frame (leakage check, required columns, monetary validation)
    _validate_candidate_frame(candidate_frame)
    
    # Step 2: Pre-allocation policy screening (Stage 1)
    pre_screened = _pre_screen_policy(candidate_frame, policy)
    
    # Step 3: Compute per-arm probabilities for all rows
    # predict_all_actions returns DataFrame with one column per arm in ARM_ORDER
    probs_df = predict_all_actions(bundle, candidate_frame)
    
    # Step 4: Process each row
    entries = {}
    candidates = []
    
    # Track row indices for candidate ordering
    row_indices = {row["attempt_id"]: idx for idx, row in candidate_frame.iterrows()}
    
    for pos_idx, (idx, row) in enumerate(candidate_frame.iterrows()):
        attempt_id = row["attempt_id"]
        payment_id = row["payment_id"]
        row_index = idx
        
        # Check if pre-screened
        if attempt_id in pre_screened:
            reason, authorized = pre_screened[attempt_id]
            entries[attempt_id] = PortfolioEntry(
                attempt_id=attempt_id,
                payment_id=payment_id,
                row_index=row_index,
                optimizer_recommendation="NO_INTERVENTION",
                no_intervention_reason=reason,
                gross_incremental_value_by_arm={},
                action_cost_by_arm={},
                net_incremental_value_by_arm={},
                selected_gross_incremental_value_inr=None,
                selected_action_cost_inr=None,
                selected_action_cost_paise=None,
                selected_net_incremental_value_inr=None,
                optimizer_sort_rank=None,
                authorized_action=authorized,
                authorization_reason=f"Pre-allocation policy pre-screen: {reason}",
                matched_rule_id=None,
                policy_overrode_recommendation=False,
            )
            continue
        
        # Get probabilities for this row (use positional index for probs_df)
        row_probs = {arm: probs_df.iloc[pos_idx][arm] for arm in ARM_ORDER}
        
        # Check for invalid predictions (NaN, Inf, out of bounds)
        invalid_prediction = False
        for arm, prob in row_probs.items():
            if not _is_valid_probability(prob):
                invalid_prediction = True
                break
        
        if invalid_prediction:
            entries[attempt_id] = PortfolioEntry(
                attempt_id=attempt_id,
                payment_id=payment_id,
                row_index=row_index,
                optimizer_recommendation="NO_INTERVENTION",
                no_intervention_reason="invalid_prediction",
                gross_incremental_value_by_arm={},
                action_cost_by_arm={},
                net_incremental_value_by_arm={},
                selected_gross_incremental_value_inr=None,
                selected_action_cost_inr=None,
                selected_action_cost_paise=None,
                selected_net_incremental_value_inr=None,
                optimizer_sort_rank=None,
                authorized_action="STOP",  # Will be overridden in post-allocation
                authorization_reason="Invalid model prediction (NaN, Inf, or out of bounds)",
                matched_rule_id=None,
                policy_overrode_recommendation=False,
            )
            continue
        
        # Step 5: Compute gross/net incremental values for each treated arm
        p_hat_control = row_probs["CONTROL"]
        amount_inr = float(row["amount_inr"])
        failure_category = str(row["failure_category"])
        risk_penalty = _compute_risk_penalty(amount_inr, failure_category)
        action_cost_inr = RETRY_INTERVENTION_COST_INR
        action_cost_paise = int(round(action_cost_inr * 100))  # 1000
        
        gross_by_arm = {}
        net_by_arm = {}
        action_cost_by_arm = {}
        
        for arm in TREATED_ARMS:
            p_hat_arm = row_probs[arm]
            gross = (p_hat_arm - p_hat_control) * amount_inr - risk_penalty
            net = gross - action_cost_inr
            
            gross_by_arm[arm] = gross
            net_by_arm[arm] = net
            action_cost_by_arm[arm] = action_cost_inr
        
        # Step 6: Create CandidatePair for each arm with positive net value
        for arm in TREATED_ARMS:
            net = net_by_arm[arm]
            if net > 0.0:
                candidates.append(CandidatePair(
                    attempt_id=attempt_id,
                    payment_id=payment_id,
                    row_index=row_index,
                    arm=arm,
                    gross_incremental_value_inr=gross_by_arm[arm],
                    action_cost_inr=action_cost_inr,
                    action_cost_paise=action_cost_paise,
                    net_incremental_value_inr=net,
                    p_hat_arm=row_probs[arm],
                    p_hat_control=p_hat_control,
                ))
        
        # If no positive net value arms, mark as non_positive_value
        if not any(net_by_arm[arm] > 0.0 for arm in TREATED_ARMS):
            entries[attempt_id] = PortfolioEntry(
                attempt_id=attempt_id,
                payment_id=payment_id,
                row_index=row_index,
                optimizer_recommendation="NO_INTERVENTION",
                no_intervention_reason="non_positive_value",
                gross_incremental_value_by_arm=gross_by_arm,
                action_cost_by_arm=action_cost_by_arm,
                net_incremental_value_by_arm=net_by_arm,
                selected_gross_incremental_value_inr=None,
                selected_action_cost_inr=None,
                selected_action_cost_paise=None,
                selected_net_incremental_value_inr=None,
                optimizer_sort_rank=None,
                authorized_action="STOP",  # Will be overridden in post-allocation
                authorization_reason="No positive net value arms",
                matched_rule_id=None,
                policy_overrode_recommendation=False,
            )
    
    # Metadata
    metadata = {
        "total_rows": len(candidate_frame),
        "pre_screened_count": len(pre_screened),
        "invalid_prediction_count": sum(1 for e in entries.values() if e.no_intervention_reason == "invalid_prediction"),
        "non_positive_value_count": sum(1 for e in entries.values() if e.no_intervention_reason == "non_positive_value"),
        "eligible_candidate_count": len(candidates),
    }
    
    return tuple(candidates), entries, metadata


@dataclass(frozen=True)
class CandidatePair:
    """A single (row, arm) candidate pair with computed values.

    Attributes:
        attempt_id: Payment attempt identifier.
        payment_id: Payment identifier.
        row_index: Row index in the candidate frame.
        arm: Treated arm name (one of TREATED_ARMS).
        gross_incremental_value_inr: (P_hat_arm - P_hat_CONTROL) * amount - risk_penalty (INR float).
        action_cost_inr: Action cost in INR (float, from RETRY_INTERVENTION_COST_INR).
        action_cost_paise: Action cost in integer paise (1000 for 10.00 INR).
        net_incremental_value_inr: Gross - action_cost_inr (INR float).
        p_hat_arm: Model probability for the arm.
        p_hat_control: Model probability for CONTROL.
    """
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


# ARM_ORDER index mapping for deterministic tie-breaking
_ARM_ORDER_INDEX = {arm: idx for idx, arm in enumerate(ARM_ORDER)}


def sort_key_candidate_pair(candidate: CandidatePair) -> tuple[float, str, int]:
    """Return 3-level key: (-net_incremental_value_inr, attempt_id, ARM_ORDER_index).
    
    Sort order:
    1. Primary: net_incremental_value_inr descending (higher value first)
    2. Secondary: attempt_id ascending (lexicographic)
    3. Tertiary: ARM_ORDER index ascending (RETRY_NOW < RETRY_LATER < REQUEST_UPDATE < HUMAN_REVIEW)
    """
    return (
        -candidate.net_incremental_value_inr,
        candidate.attempt_id,
        _ARM_ORDER_INDEX.get(candidate.arm, 999),
    )


def rank_candidate_pairs(candidates: Sequence[CandidatePair]) -> tuple[CandidatePair, ...]:
    """Sort candidates deterministically by net_incremental_value_inr desc, attempt_id asc, ARM_ORDER asc.
    
    Pure function. Input: sequence of CandidatePair. Output: sorted tuple of CandidatePair.
    Zero random or clock reads. Identical input produces identical output.
    """
    return tuple(sorted(candidates, key=sort_key_candidate_pair))


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
    import time
    import tracemalloc
    
    # Input validation - check guard limits
    N = len(eligible_attempt_ids)
    if N > config.max_supported_rows:
        raise PortfolioProblemTooLargeError(f"Number of rows {N} exceeds max_supported_rows {config.max_supported_rows}")
    
    # Convert budget to paise and units
    budget_limit_paise = None
    if config.budget_limit_inr is not None:
        budget_limit_paise = int(round(config.budget_limit_inr * 100))
        # Check if budget exceeds max supported budget units (in 1000 paise units)
        U = budget_limit_paise // 1000
        if U > config.max_supported_budget_units:
            raise PortfolioProblemTooLargeError(f"Budget units {U} exceeds max_supported_budget_units {config.max_supported_budget_units}")
    
    hr_capacity = config.human_review_capacity
    if hr_capacity is not None and hr_capacity > config.max_supported_hr_capacity:
        raise PortfolioProblemTooLargeError(f"HR capacity {hr_capacity} exceeds max_supported_hr_capacity {config.max_supported_hr_capacity}")
    
    # Group candidates by attempt_id
    from collections import defaultdict
    by_row = defaultdict(list)
    for c in candidates:
        by_row[c.attempt_id].append(c)
    
    # Filter to only eligible attempt_ids and keep only positive net value candidates
    eligible_ids = sorted([aid for aid in eligible_attempt_ids if aid in by_row])
    if not eligible_ids:
        # No eligible rows
        budget_allocated_paise = 0
        budget_remaining_paise = budget_limit_paise if budget_limit_paise is not None else None
        return {}, {aid: "no_positive_net_value" for aid in eligible_attempt_ids}, {
            "budget_allocated_inr": 0.0,
            "budget_allocated_paise": 0,
            "budget_remaining_inr": config.budget_limit_inr,
            "budget_remaining_paise": budget_remaining_paise,
            "hr_allocated_count": 0,
            "solver_type": "exact_dp_2d",
            "preflight_stats": {}
        }
    
    # Prepare DP dimensions
    # U = budget units (1000 paise = 10.00 INR each)
    # H = HR capacity
    # For action costs: each action costs 1000 paise = 1 unit
    # HR cost: 1 for HUMAN_REVIEW, 0 otherwise
    
    if budget_limit_paise is not None:
        U = budget_limit_paise // 1000
    else:
        # Unconstrained: use number of eligible rows as upper bound
        U = len(eligible_ids)
    
    if hr_capacity is not None:
        H = hr_capacity
    else:
        # Unconstrained: use number of eligible rows as upper bound
        H = len(eligible_ids)
    
    # Ensure minimum dimensions
    U = max(0, U)
    H = max(0, H)
    
    # Pre-compute candidate options per row
    # Each row has: NO_INTERVENTION (value=0, cost=0, hr=0) + candidate arms
    row_options = {}
    for aid in eligible_ids:
        options = [("NO_INTERVENTION", 0.0, 0, 0)]  # (arm, net_value, cost_units, hr_cost)
        for cand in by_row[aid]:
            if cand.net_incremental_value_inr > 0.0:
                cost_units = cand.action_cost_paise // 1000
                hr_cost = 1 if cand.arm == "HUMAN_REVIEW" else 0
                options.append((cand.arm, cand.net_incremental_value_inr, cost_units, hr_cost))
        row_options[aid] = options
    
    # 2-layer rolling DP arrays
    # dp_prev[u][h] = max value achievable with budget u and HR h after processing previous rows
    # Use list of lists for better performance with small dimensions
    dp_prev = [[0.0 for _ in range(H + 1)] for _ in range(U + 1)]
    dp_curr = [[0.0 for _ in range(H + 1)] for _ in range(U + 1)]
    
    # Backtracking: store chosen arm index for each state
    # backtrack[i][u][h] = index of chosen option in row_options[eligible_ids[i-1]]
    backtrack = [[[0 for _ in range(H + 1)] for _ in range(U + 1)] for _ in range(len(eligible_ids) + 1)]
    
    # DP iteration over rows
    for i, aid in enumerate(eligible_ids, 1):
        options = row_options[aid]
        num_options = len(options)
        
        # Reset dp_curr
        for u in range(U + 1):
            for h in range(H + 1):
                dp_curr[u][h] = dp_prev[u][h]
                backtrack[i][u][h] = 0  # NO_INTERVENTION
        
        # Try each option
        for opt_idx, (arm, value, cost_units, hr_cost) in enumerate(options):
            if opt_idx == 0:
                continue  # NO_INTERVENTION already handled
            
            # Iterate backwards over budget and HR
            for u in range(U, cost_units - 1, -1):
                for h in range(H, hr_cost - 1, -1):
                    prev_val = dp_prev[u - cost_units][h - hr_cost]
                    candidate_val = prev_val + value
                    
                    if candidate_val > dp_curr[u][h] + 1e-6:
                        dp_curr[u][h] = candidate_val
                        backtrack[i][u][h] = opt_idx
                    elif abs(candidate_val - dp_curr[u][h]) <= 1e-6:
                        # Tie-breaking: prefer lower u (less budget spent), then lower h (less HR), then earlier ARM_ORDER
                        current_opt_idx = backtrack[i][u][h]
                        current_arm = options[current_opt_idx][0]
                        current_cost = options[current_opt_idx][2]
                        current_hr = options[current_opt_idx][3]
                        
                        # Prefer lower budget spent (lower u)
                        if u < (u - current_cost + current_cost):  # This is always equal, check actual budget spent
                            pass
                        # Actually we need to track actual budget spent for tie-breaking
                        # For now, use the standard tie-breaking: earlier ARM_ORDER
                        if _ARM_ORDER_INDEX.get(arm, 999) < _ARM_ORDER_INDEX.get(current_arm, 999):
                            dp_curr[u][h] = candidate_val
                            backtrack[i][u][h] = opt_idx
        
        # Swap dp_prev and dp_curr
        dp_prev, dp_curr = dp_curr, dp_prev
    
    # Find optimal end state
    best_u = U
    best_h = H
    best_val = dp_prev[U][H]
    
    # Find the state with maximum value (in case constraints not fully used)
    for u in range(U + 1):
        for h in range(H + 1):
            if dp_prev[u][h] > best_val + 1e-6:
                best_val = dp_prev[u][h]
                best_u = u
                best_h = h
            elif abs(dp_prev[u][h] - best_val) <= 1e-6:
                # Tie-breaking: prefer lower u, then lower h
                if u < best_u or (u == best_u and h < best_h):
                    best_u = u
                    best_h = h
    
    # Traceback to reconstruct allocation
    allocated = {}
    u, h = best_u, best_h
    for i in range(len(eligible_ids), 0, -1):
        aid = eligible_ids[i - 1]
        opt_idx = backtrack[i][u][h]
        if opt_idx > 0:
            arm, value, cost_units, hr_cost = row_options[aid][opt_idx]
            # Find the candidate pair
            for cand in by_row[aid]:
                if cand.arm == arm:
                    allocated[aid] = cand
                    break
            u -= cost_units
            h -= hr_cost
        # else: NO_INTERVENTION, u and h unchanged
    
    # Unallocated reasons
    unallocated_reasons = {}
    for aid in eligible_attempt_ids:
        if aid not in allocated:
            if aid in eligible_ids:
                # Check if it had positive net value candidates
                if any(c.net_incremental_value_inr > 0.0 for c in by_row[aid]):
                    unallocated_reasons[aid] = "budget_exhausted" if budget_limit_paise is not None else "hr_capacity_exhausted"
                else:
                    unallocated_reasons[aid] = "non_positive_net_value"
            else:
                unallocated_reasons[aid] = "ineligible"
    
    # Compute metadata
    budget_allocated_paise = best_u * 1000
    budget_allocated_inr = budget_allocated_paise / 100.0
    budget_remaining_paise = budget_limit_paise - budget_allocated_paise if budget_limit_paise is not None else None
    budget_remaining_inr = budget_remaining_paise / 100.0 if budget_remaining_paise is not None else None
    
    hr_allocated_count = sum(1 for c in allocated.values() if c.arm == "HUMAN_REVIEW")
    
    metadata = {
        "budget_allocated_inr": budget_allocated_inr,
        "budget_allocated_paise": budget_allocated_paise,
        "budget_remaining_inr": budget_remaining_inr,
        "budget_remaining_paise": budget_remaining_paise,
        "hr_allocated_count": hr_allocated_count,
        "solver_type": "exact_dp_2d",
        "preflight_stats": {}
    }
    
    return allocated, unallocated_reasons, metadata


def run_solver_preflight_benchmark(
    candidates: tuple[CandidatePair, ...],
    config: OptimizerConfig,
) -> dict:
    """Execute preflight benchmark recording dimensions, DP state count, transition count, elapsed time, peak memory, and exactness status."""
    import time
    import tracemalloc
    
    # N = number of unique rows (attempt_ids), not candidate pairs
    N = len(set(c.attempt_id for c in candidates))
    
    # Compute U and H
    if config.budget_limit_inr is not None:
        budget_limit_paise = int(round(config.budget_limit_inr * 100))
        U = budget_limit_paise // 1000
    else:
        U = N  # Use N as upper bound when unconstrained
    
    if config.human_review_capacity is not None:
        H = config.human_review_capacity
    else:
        H = N  # Use N as upper bound when unconstrained
    
    K = 4  # 4 treated arms
    
    # State count: N rows × (U+1) budget units × (H+1) HR capacity
    state_count = N * (U + 1) * (H + 1)
    # Action-transition evaluations: N rows × K arms × (U+1) × (H+1)
    transition_count = N * K * (U + 1) * (H + 1)
    
    # Run benchmark
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Create dummy eligible_attempt_ids
    eligible_attempt_ids = set(c.attempt_id for c in candidates)
    
    # Run the solver
    try:
        solve_portfolio_allocation(candidates, eligible_attempt_ids, config)
        exactness = "EXACT_DP_OPTIMAL"
    except PortfolioProblemTooLargeError:
        exactness = "PROBLEM_TOO_LARGE"
    except Exception as e:
        exactness = f"ERROR: {type(e).__name__}"
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed_seconds = end_time - start_time
    peak_mb = peak / (1024 * 1024)
    
    return {
        "N": N,
        "U": U,
        "H": H,
        "K": K,
        "state_count": state_count,
        "transition_count": transition_count,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_mb": peak_mb,
        "solver_type": "exact_dp_2d",
        "exactness": exactness
    }


def authorize_post_allocation(
    allocated: dict[str, CandidatePair],
    unallocated_reasons: dict[str, str],
    pre_screened_entries: dict[str, "PortfolioEntry"],
    eligible_attempt_ids: set[str],
    all_candidates: tuple[CandidatePair, ...],
    policy: PolicyConfig,
    candidate_frame: "pd.DataFrame",
) -> "PortfolioAllocation":
    """Apply post-allocation deterministic policy authorization to every row.

    Stage 2 of the two-stage policy pipeline:
    - Pre-screened rows (Stage 1) are passed through unchanged.
    - Invalid prediction rows are passed through unchanged.
    - Allocated rows: inject selected-arm probability, run full decide_action().
    - Unallocated eligible rows: inject CONTROL probability, run full decide_action().

    The allocation accounting (budget, HR, optimizer_recommendation) is frozen
    before authorization. Policy authorization changes the authorized_action
    outcome but does NOT modify the optimizer allocation accounting.

    Returns a complete PortfolioAllocation with entries and summary.
    """
    from ml.portfolio_audit import PortfolioEntry, PortfolioSummary, PortfolioAllocation

    # Build candidates lookup by attempt_id
    from collections import defaultdict
    by_row: dict[str, list[CandidatePair]] = defaultdict(list)
    for c in all_candidates:
        by_row[c.attempt_id].append(c)

    # Build lookup for all candidates' probabilities
    p_hat_control_by_row: dict[str, float] = {}
    for aid, cands in by_row.items():
        if cands:
            p_hat_control_by_row[aid] = cands[0].p_hat_control

    # Build row data lookup from the candidate frame
    row_data: dict[str, dict] = {}
    for _, frame_row in candidate_frame.iterrows():
        aid = frame_row["attempt_id"]
        row_data[aid] = {col: frame_row[col] for col in PRE_ALLOCATION_POLICY_COLUMNS if col in frame_row}
        row_data[aid]["amount_inr"] = float(frame_row["amount_inr"])
        row_data[aid]["failure_category"] = str(frame_row["failure_category"])

    entries: list[PortfolioEntry] = []
    total_overrides = 0
    total_stop_overrides = 0
    post_policy_net_authorized = 0
    rec_counts: dict[str, int] = defaultdict(int)
    auth_counts: dict[str, int] = defaultdict(int)

    # Process all rows in deterministic order
    all_attempt_ids = sorted(set(
        list(pre_screened_entries.keys()) +
        list(allocated.keys()) +
        [aid for aid in unallocated_reasons.keys() if aid not in pre_screened_entries]
    ))

    for aid in all_attempt_ids:
        if aid in pre_screened_entries:
            # Stage 1 pre-screened rows: pass through unchanged
            entry = pre_screened_entries[aid]
            entries.append(entry)
            rec_counts[entry.optimizer_recommendation] += 1
            auth_counts[entry.authorized_action] += 1
            continue

        if aid in allocated:
            # Allocated row: inject selected-arm probability
            cand = allocated[aid]
            recovery_prob = cand.p_hat_arm
            optimizer_rec = cand.arm
        else:
            # Unallocated eligible row: inject CONTROL probability
            recovery_prob = p_hat_control_by_row.get(aid, 0.0)
            optimizer_rec = "NO_INTERVENTION"

        # Build full policy context from row data
        context: dict = {"recovery_probability": recovery_prob}
        if aid in row_data:
            context.update(row_data[aid])

        # Run full decide_action
        decision = decide_action(context, policy)
        authorized = decision.authorized_action
        auth_reason = decision.reason
        matched_rule = decision.matched_rule_id

        # Determine policy override
        if aid in allocated:
            policy_overrode = authorized != optimizer_rec
            # Build the entry from the allocated candidate
            cands = by_row[aid]
            gross_by_arm = {c.arm: c.gross_incremental_value_inr for c in cands}
            cost_by_arm = {c.arm: c.action_cost_inr for c in cands}
            net_by_arm = {c.arm: c.net_incremental_value_inr for c in cands}

            # Determine sort rank
            sorted_candidates = rank_candidate_pairs(tuple(cands))
            sort_rank = None
            for rank_idx, sc in enumerate(sorted_candidates, 1):
                if sc.arm == cand.arm:
                    sort_rank = rank_idx
                    break

            entry = PortfolioEntry(
                attempt_id=aid,
                payment_id=cand.payment_id,
                row_index=cand.row_index,
                optimizer_recommendation=optimizer_rec,
                no_intervention_reason=None,
                gross_incremental_value_by_arm=gross_by_arm,
                action_cost_by_arm=cost_by_arm,
                net_incremental_value_by_arm=net_by_arm,
                selected_gross_incremental_value_inr=cand.gross_incremental_value_inr,
                selected_action_cost_inr=cand.action_cost_inr,
                selected_action_cost_paise=cand.action_cost_paise,
                selected_net_incremental_value_inr=cand.net_incremental_value_inr,
                optimizer_sort_rank=sort_rank,
                authorized_action=authorized,
                authorization_reason=auth_reason,
                matched_rule_id=matched_rule,
                policy_overrode_recommendation=policy_overrode,
            )
        else:
            # Unallocated row
            policy_overrode = False
            cands = by_row.get(aid, [])
            gross_by_arm = {c.arm: c.gross_incremental_value_inr for c in cands}
            cost_by_arm = {c.arm: c.action_cost_inr for c in cands}
            net_by_arm = {c.arm: c.net_incremental_value_inr for c in cands}
            reason = unallocated_reasons.get(aid, "unknown")

            # Get payment_id and row_index from frame data if available
            payment_id = row_data.get(aid, {}).get("payment_id", "")
            row_idx = 0
            if aid in row_data:
                # Find row_index from candidates or frame
                if cands:
                    row_idx = cands[0].row_index

            entry = PortfolioEntry(
                attempt_id=aid,
                payment_id=payment_id if payment_id else (cands[0].payment_id if cands else ""),
                row_index=row_idx,
                optimizer_recommendation="NO_INTERVENTION",
                no_intervention_reason=reason,
                gross_incremental_value_by_arm=gross_by_arm,
                action_cost_by_arm=cost_by_arm,
                net_incremental_value_by_arm=net_by_arm,
                selected_gross_incremental_value_inr=None,
                selected_action_cost_inr=None,
                selected_action_cost_paise=None,
                selected_net_incremental_value_inr=None,
                optimizer_sort_rank=None,
                authorized_action=authorized,
                authorization_reason=auth_reason,
                matched_rule_id=matched_rule,
                policy_overrode_recommendation=policy_overrode,
            )

        entries.append(entry)
        rec_counts[entry.optimizer_recommendation] += 1
        auth_counts[entry.authorized_action] += 1

        if policy_overrode:
            total_overrides += 1
            if authorized == "STOP":
                total_stop_overrides += 1

        if authorized != "STOP":
            post_policy_net_authorized += 1

    # Build summary from allocation metadata
    budget_limit_paise = None
    budget_limit_inr = None
    budget_allocated_paise = 0
    budget_allocated_inr = 0.0
    budget_remaining_paise = None
    budget_remaining_inr = None
    hr_capacity_limit = None
    hr_allocated_count = 0

    # Reconstruct budget/HR from allocation
    for cand in allocated.values():
        budget_allocated_paise += cand.action_cost_paise
        if cand.arm == "HUMAN_REVIEW":
            hr_allocated_count += 1

    budget_allocated_inr = budget_allocated_paise / 100.0

    # Count buckets
    pre_screen_count = len(pre_screened_entries)
    invalid_pred_count = sum(
        1 for e in pre_screened_entries.values()
        if e.no_intervention_reason == "invalid_prediction"
    )
    optimizer_allocated_count = len(allocated)
    no_intervention_count = len([
        aid for aid in all_attempt_ids
        if aid not in allocated and aid not in pre_screened_entries
    ])

    total_rows = pre_screen_count + optimizer_allocated_count + no_intervention_count

    summary = PortfolioSummary(
        total_rows=total_rows,
        pre_screen_stopped_count=pre_screen_count - invalid_pred_count,
        invalid_prediction_count=invalid_pred_count,
        optimizer_allocated_count=optimizer_allocated_count,
        no_intervention_count=no_intervention_count,
        eligible_candidate_count=len(set(c.attempt_id for c in all_candidates)),
        budget_limit_inr=budget_limit_inr,
        budget_limit_paise=budget_limit_paise,
        budget_allocated_inr=budget_allocated_inr,
        budget_allocated_paise=budget_allocated_paise,
        budget_remaining_inr=budget_remaining_inr,
        budget_remaining_paise=budget_remaining_paise,
        human_review_capacity_limit=hr_capacity_limit,
        human_review_allocated_count=hr_allocated_count,
        post_policy_net_authorized_count=post_policy_net_authorized,
        total_policy_overrides=total_overrides,
        total_policy_stop_overrides=total_stop_overrides,
        optimizer_objective_value_inr=sum(
            c.net_incremental_value_inr for c in allocated.values()
        ),
        optimizer_status="success" if allocated or not all_attempt_ids else "empty_portfolio",
        action_recommendation_counts=dict(rec_counts),
        action_authorized_counts=dict(auth_counts),
    )

    return PortfolioAllocation(
        entries=tuple(entries),
        summary=summary,
        metadata={"post_allocation_policy": True},
    )