"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(settings: Settings | None = None) -> dict[str, str]:
    service_name = settings.service_name if settings else "recoverai-api"
    return {"status": "ok", "service": service_name}
