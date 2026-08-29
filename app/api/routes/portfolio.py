"""Portfolio optimization endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.portfolio import (
    PortfolioRequest,
    PortfolioResponse,
    PortfolioEntryResponse,
    PortfolioSummaryResponse,
)
from app.services.portfolio_service import optimize_portfolio

router = APIRouter(prefix="/api", tags=["portfolio"])


def _serialize_entry(entry) -> PortfolioEntryResponse:
    return PortfolioEntryResponse(
        attempt_id=entry.attempt_id,
        payment_id=entry.payment_id,
        optimizer_recommendation=entry.optimizer_recommendation,
        authorized_action=entry.authorized_action,
        authorization_reason=entry.authorization_reason,
        matched_rule_id=entry.matched_rule_id,
        policy_overrode_recommendation=entry.policy_overrode_recommendation,
        selected_net_incremental_value_inr=entry.selected_net_incremental_value_inr,
        selected_action_cost_inr=entry.selected_action_cost_inr,
        no_intervention_reason=entry.no_intervention_reason,
    )


def _serialize_summary(summary) -> PortfolioSummaryResponse:
    return PortfolioSummaryResponse(
        total_rows=summary.total_rows,
        optimizer_allocated_count=summary.optimizer_allocated_count,
        no_intervention_count=summary.no_intervention_count,
        budget_limit_inr=summary.budget_limit_inr,
        budget_allocated_inr=summary.budget_allocated_inr,
        budget_remaining_inr=summary.budget_remaining_inr,
        human_review_capacity_limit=summary.human_review_capacity_limit,
        human_review_allocated_count=summary.human_review_allocated_count,
        post_policy_net_authorized_count=summary.post_policy_net_authorized_count,
        total_policy_overrides=summary.total_policy_overrides,
        optimizer_objective_value_inr=summary.optimizer_objective_value_inr,
        optimizer_status=summary.optimizer_status,
        action_recommendation_counts=summary.action_recommendation_counts,
        action_authorized_counts=summary.action_authorized_counts,
    )


@router.post("/portfolio/optimize", response_model=PortfolioResponse)
def portfolio_optimize(request: PortfolioRequest) -> PortfolioResponse:
    allocation = optimize_portfolio(
        budget_inr=request.budget_inr,
        human_review_capacity=request.human_review_capacity,
    )

    return PortfolioResponse(
        solver=allocation.metadata.get("solver_type", "unknown"),
        summary=_serialize_summary(allocation.summary),
        entries=[_serialize_entry(e) for e in allocation.entries],
        metadata=allocation.metadata,
    )
