"""Day 7 portfolio optimizer interfaces and contracts (Task 1 & 2).

Frozen dataclasses, exceptions, constants, candidate construction, and JSON serialization
specification for the bounded recovery portfolio optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    
    for idx, row in candidate_frame.iterrows():
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
        
        # Get probabilities for this row
        row_probs = {arm: probs_df.iloc[idx][arm] for arm in ARM_ORDER}
        
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