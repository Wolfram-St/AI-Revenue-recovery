"""Day 7 portfolio optimizer interfaces and contracts (Task 1).

Frozen dataclasses, exceptions, constants, and JSON serialization specification
for the bounded recovery portfolio optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.features import FORBIDDEN_FEATURES


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
        "recovered",
        "recovery_time_hours",
        "recovery_action",
        "action_outcome",
        "recovered_amount_inr",
    }
)


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