"""Day 7 portfolio audit interfaces and contracts (Task 1).

Frozen dataclasses for portfolio-level audit traces and deterministic JSON serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class PortfolioEntry:
    """Per-row portfolio audit entry.

    Attributes:
        attempt_id: Payment attempt identifier.
        payment_id: Payment identifier.
        row_index: Row index in the candidate frame.
        optimizer_recommendation: Arm recommended by optimizer, or "NO_INTERVENTION".
        no_intervention_reason: Reason if NO_INTERVENTION:
            "non_positive_value", "policy_pre_screen", "budget_exhausted",
            "human_review_capacity_exhausted", "invalid_prediction", or None.
        gross_incremental_value_by_arm: Gross incremental revenue estimate per treated arm.
        action_cost_by_arm: Action cost per treated arm (INR float).
        net_incremental_value_by_arm: Net incremental revenue estimate per treated arm.
        selected_gross_incremental_value_inr: Gross value of recommended arm; None if NO_INTERVENTION.
        selected_action_cost_inr: Action cost of recommended arm (INR float); None if NO_INTERVENTION.
        selected_action_cost_paise: Action cost of recommended arm (integer paise); None if NO_INTERVENTION.
        selected_net_incremental_value_inr: Net value of recommended arm; None if NO_INTERVENTION.
        optimizer_sort_rank: Position in sorted candidate list (1 = highest); None if not a candidate.
        authorized_action: Canonical action authorized by decide_action().
        authorization_reason: Reason string from PolicyDecision.
        matched_rule_id: Rule that fired in decide_action(); None if residual default.
        policy_overrode_recommendation: True if authorized_action != optimizer_recommendation
            and optimizer_recommendation != "NO_INTERVENTION".
    """
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
    """Portfolio-level summary metrics.

    Attributes:
        total_rows: Total rows submitted to the optimizer.
        pre_screen_stopped_count: Rows removed by policy pre-screen (PRE_SCREEN_STOPPED).
        invalid_prediction_count: Rows excluded due to NaN/out-of-bounds predictions (INVALID_PREDICTION).
        optimizer_allocated_count: Rows assigned a treated arm by the optimizer (OPTIMIZER_ALLOCATED).
        no_intervention_count: Eligible rows assigned NO_INTERVENTION by optimizer.
        eligible_candidate_count: Rows submitted to optimizer after pre-screening.
        budget_limit_inr: Configured budget limit in INR; None if unconstrained.
        budget_limit_paise: Configured budget limit in integer paise; None if unconstrained.
        budget_allocated_inr: Total budget allocated by optimizer (INR float).
        budget_allocated_paise: Total budget allocated by optimizer (integer paise).
        budget_remaining_inr: Remaining budget after allocation (INR float); None if unconstrained.
        budget_remaining_paise: Remaining budget after allocation (integer paise); None if unconstrained.
        human_review_capacity_limit: Configured HR capacity limit; None if unconstrained.
        human_review_allocated_count: HUMAN_REVIEW arms allocated by optimizer.
        post_policy_net_authorized_count: Rows where authorized_action != "STOP" after policy.
        total_policy_overrides: Rows where policy diverged from optimizer recommendation.
        total_policy_stop_overrides: Subset where authorized_action == "STOP".
        optimizer_objective_value_inr: Sum of selected_net_incremental_value_inr over allocated rows (MODEL ESTIMATE).
        optimizer_status: Status code (e.g., "success", "empty_portfolio", "budget_exhausted_before_allocation",
            "human_review_capacity_zero", "no_positive_value_candidates").
        action_recommendation_counts: Count per optimizer_recommendation value.
        action_authorized_counts: Count per authorized_action value.
    """
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
    """Complete portfolio allocation result with audit trace.

    Attributes:
        entries: Tuple of per-row PortfolioEntry records.
        summary: PortfolioSummary with aggregate metrics.
        metadata: Additional metadata (config, timestamps, solver info, etc.).
    """
    entries: tuple[PortfolioEntry, ...]
    summary: PortfolioSummary
    metadata: dict

    def to_json(self) -> str:
        """Return byte-identical sorted JSON using sort_keys=True, separators=(',', ':'), allow_nan=False.

        Monetary fields are serialized as-is (INR float with 2-decimal precision from rounding).
        Integer paise fields are serialized as integers.
        """
        def _jsonify(value: Any) -> Any:
            if isinstance(value, tuple):
                return [_jsonify(item) for item in value]
            if isinstance(value, PortfolioEntry):
                return _entry_to_dict(value)
            if isinstance(value, PortfolioSummary):
                return _summary_to_dict(value)
            if isinstance(value, dict):
                return {k: _jsonify(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_jsonify(item) for item in value]
            return value

        def _entry_to_dict(entry: PortfolioEntry) -> dict:
            payload = {}
            for field in fields(entry):
                payload[field.name] = _jsonify(getattr(entry, field.name))
            return payload

        def _summary_to_dict(summary: PortfolioSummary) -> dict:
            payload = {}
            for field in fields(summary):
                payload[field.name] = _jsonify(getattr(summary, field.name))
            return payload

        payload = {
            "entries": [_entry_to_dict(entry) for entry in self.entries],
            "summary": _summary_to_dict(self.summary),
            "metadata": _jsonify(self.metadata),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)