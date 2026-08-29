"""Audit trail service: expose available audit data from bootstrap traces."""

from __future__ import annotations

from typing import Any

from app.services.data_bootstrap import get_bootstrap
from recovery.audit import trace_to_dict


def get_audit_trail(
    case_id: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    bootstrap = get_bootstrap()

    entries = []
    for trace in bootstrap.traces:
        if case_id is not None and trace.attempt_id != case_id:
            continue
        trace_dict = trace_to_dict(trace)
        entries.append({
            "event_type": "decision_recorded",
            "actor_type": "system",
            "recovery_case_id": trace.attempt_id,
            "action": trace.authorized_action,
            "decision_reason": trace.authorization_reason,
            "event_payload": trace_dict,
        })

    total = len(entries)
    start = (page - 1) * page_size
    end = start + page_size
    page_entries = entries[start:end]

    return {
        "entries": page_entries,
        "total": total,
        "page": page,
        "page_size": page_size,
    }
