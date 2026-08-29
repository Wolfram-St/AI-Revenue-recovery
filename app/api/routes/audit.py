"""Audit trail endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.services.audit_service import get_audit_trail

router = APIRouter(prefix="/api", tags=["audit"])


@router.get("/audit")
def audit(
    case_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    return get_audit_trail(case_id=case_id, page=page, page_size=page_size)
