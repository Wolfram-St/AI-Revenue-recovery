"""Dashboard API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ActionDistribution(BaseModel):
    action: str
    count: int


class DashboardResponse(BaseModel):
    total_cases: int
    revenue_at_risk_inr: float
    estimated_recoverable_value_inr: float
    candidate_count: int
    stop_count: int
    noop_count: int
    action_distribution: list[ActionDistribution]
    demo_mode: bool
