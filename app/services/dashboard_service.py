"""Dashboard service: read model from bootstrap data and counterfactual uplift ledger."""

from __future__ import annotations

from typing import Any

from app.services.data_bootstrap import get_bootstrap
from app.api.schemas.dashboard import DashboardResponse, ActionDistribution
from recovery.counterfactual_ledger import compute_counterfactual_uplift_ledger


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

    # Compute counterfactual uplift metrics from bootstrap traces
    records = []
    for trace in bootstrap.traces:
        # Determine simulated recovery based on ERV and action
        is_recovered = trace.authorized_action in ("RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE") and (trace.recovery_probability >= 0.50)
        touchpoints = [trace.authorized_action]
        if trace.failure_category == "temporary_decline":
            touchpoints.append("WHATSAPP_LINK")
        records.append({
            "case_id": trace.attempt_id,
            "customer_id": trace.customer_id,
            "amount_inr": trace.amount_inr,
            "recovered": is_recovered,
            "touchpoints": touchpoints,
        })

    _, uplift_summary = compute_counterfactual_uplift_ledger(records, holdout_pct=0.05)

    return DashboardResponse(
        total_cases=case_count,
        revenue_at_risk_inr=round(revenue_at_risk, 2),
        estimated_recoverable_value_inr=round(float(total_erv), 2),
        candidate_count=candidate_count,
        stop_count=stop_count,
        noop_count=noop_count,
        action_distribution=distribution,
        demo_mode=True,
        control_holdout_rate_pct=uplift_summary.control_recovery_rate_pct,
        treatment_recovery_rate_pct=uplift_summary.treatment_recovery_rate_pct,
        incremental_uplift_pct=uplift_summary.incremental_uplift_pct,
        true_incremental_recovery_inr=uplift_summary.true_incremental_recovery_inr,
        channel_attribution=uplift_summary.channel_attribution_inr,
    )
