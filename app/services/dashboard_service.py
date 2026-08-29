"""Dashboard service: read model from bootstrap data."""

from __future__ import annotations

from typing import Any

from app.services.data_bootstrap import get_bootstrap
from app.api.schemas.dashboard import DashboardResponse, ActionDistribution


def get_dashboard() -> DashboardResponse:
    bootstrap = get_bootstrap()
    summary = bootstrap.summary

    action_counts = summary.get("action_counts", {})
    distribution = [
        ActionDistribution(action=action, count=count)
        for action, count in sorted(action_counts.items())
    ]

    total_erv = summary.get("total_candidate_erv_inr", 0.0)
    noop_count = summary.get("noop_count", 0)
    case_count = summary.get("case_count", 0)
    candidate_count = summary.get("candidate_count", 0)
    stop_count = summary.get("stop_count", 0)

    revenue_at_risk = sum(
        trace.amount_inr for trace in bootstrap.traces
    )

    return DashboardResponse(
        total_cases=case_count,
        revenue_at_risk_inr=round(revenue_at_risk, 2),
        estimated_recoverable_value_inr=round(float(total_erv), 2),
        candidate_count=candidate_count,
        stop_count=stop_count,
        noop_count=noop_count,
        action_distribution=distribution,
        demo_mode=True,
    )
