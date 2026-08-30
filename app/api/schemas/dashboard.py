"""Dashboard API schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


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
    # Counterfactual Uplift & Causal Attribution Metrics
    control_holdout_rate_pct: float | None = Field(default=None, description="Passive organic recovery rate in 5% holdout")
    treatment_recovery_rate_pct: float | None = Field(default=None, description="Active recovery rate in AI treatment arm")
    incremental_uplift_pct: float | None = Field(default=None, description="Net percentage uplift over organic baseline")
    true_incremental_recovery_inr: float | None = Field(default=None, description="True incremental revenue won back by AI in INR")
    channel_attribution: dict[str, float] | None = Field(default=None, description="Attributed incremental revenue by channel/touchpoint")
