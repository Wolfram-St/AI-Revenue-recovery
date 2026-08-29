"""Recovery case endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.api.schemas.cases import CaseListResponse
from app.services.recovery_service import list_cases, get_case_detail

router = APIRouter(prefix="/api", tags=["cases"])


@router.get("/cases", response_model=CaseListResponse)
def cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    failure_category: str | None = Query(None),
    recommendation: str | None = Query(None),
    is_stop: bool | None = Query(None),
    amount_min: float | None = Query(None, ge=0),
    amount_max: float | None = Query(None, ge=0),
) -> CaseListResponse:
    return list_cases(
        page=page,
        page_size=page_size,
        failure_category=failure_category,
        recommendation=recommendation,
        is_stop=is_stop,
        amount_min=amount_min,
        amount_max=amount_max,
    )


@router.get("/cases/{case_id}")
def case_detail(case_id: str) -> dict[str, Any]:
    return get_case_detail(case_id)
