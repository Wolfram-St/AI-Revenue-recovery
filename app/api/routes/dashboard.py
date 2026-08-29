"""Dashboard endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.dashboard import DashboardResponse
from app.services.dashboard_service import get_dashboard

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard() -> DashboardResponse:
    return get_dashboard()
