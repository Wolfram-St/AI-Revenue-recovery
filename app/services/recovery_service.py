"""Recovery case service: listing and detail from bootstrap traces."""

from __future__ import annotations

from app.errors import CaseNotFoundError
from app.services.data_bootstrap import get_bootstrap
from app.api.schemas.cases import CaseSummary, CaseListResponse
from recovery.audit import DecisionTrace, trace_to_dict


def list_cases(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    failure_category: str | None = None,
    recommendation: str | None = None,
    is_stop: bool | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
) -> CaseListResponse:
    bootstrap = get_bootstrap()
    traces = bootstrap.traces

    filtered = list(traces)

    if failure_category is not None:
        filtered = [t for t in filtered if t.failure_category == failure_category]
    if recommendation is not None:
        filtered = [t for t in filtered if t.scoring_recommendation == recommendation]
    if is_stop is not None:
        filtered = [t for t in filtered if t.is_stop == is_stop]
    if amount_min is not None:
        filtered = [t for t in filtered if t.amount_inr >= amount_min]
    if amount_max is not None:
        filtered = [t for t in filtered if t.amount_inr <= amount_max]

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_traces = filtered[start:end]

    cases = [
        CaseSummary(
            attempt_id=t.attempt_id,
            payment_id=t.payment_id,
            customer_id=t.customer_id,
            amount_inr=t.amount_inr,
            failure_category=t.failure_category,
            scoring_recommendation=t.scoring_recommendation,
            authorized_action=t.authorized_action,
            expected_recovery_value_inr=t.expected_recovery_value_inr,
            recovery_probability=t.recovery_probability,
            is_stop=t.is_stop,
        )
        for t in page_traces
    ]

    return CaseListResponse(
        cases=cases,
        total=total,
        page=page,
        page_size=page_size,
    )


def get_case_detail(case_id: str) -> dict:
    bootstrap = get_bootstrap()

    for trace in bootstrap.traces:
        if trace.attempt_id == case_id:
            trace_dict = trace_to_dict(trace)
            return {
                "case": trace_dict,
                "audit_history": [
                    {
                        "event_type": "decision_recorded",
                        "actor_type": "system",
                        "action": trace.authorized_action,
                        "decision_reason": trace.authorization_reason,
                        "event_payload": trace_dict,
                    }
                ],
            }

    raise CaseNotFoundError(case_id)
