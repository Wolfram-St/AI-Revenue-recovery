"""Portfolio API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PortfolioRequest(BaseModel):
    budget_inr: float = Field(ge=0, description="Maximum total intervention budget in INR")
    human_review_capacity: int = Field(ge=0, description="Maximum number of HUMAN_REVIEW actions")


class PortfolioEntryResponse(BaseModel):
    attempt_id: str
    payment_id: str
    optimizer_recommendation: str
    authorized_action: str
    authorization_reason: str
    matched_rule_id: str | None
    policy_overrode_recommendation: bool
    selected_net_incremental_value_inr: float | None
    selected_action_cost_inr: float | None
    no_intervention_reason: str | None


class PortfolioSummaryResponse(BaseModel):
    total_rows: int
    optimizer_allocated_count: int
    no_intervention_count: int
    budget_limit_inr: float | None
    budget_allocated_inr: float
    budget_remaining_inr: float | None
    human_review_capacity_limit: int | None
    human_review_allocated_count: int
    post_policy_net_authorized_count: int
    total_policy_overrides: int
    optimizer_objective_value_inr: float
    optimizer_status: str
    action_recommendation_counts: dict[str, int]
    action_authorized_counts: dict[str, int]


class PortfolioResponse(BaseModel):
    solver: str
    summary: PortfolioSummaryResponse
    entries: list[PortfolioEntryResponse]
    metadata: dict[str, Any]
