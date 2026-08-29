"""Case analysis endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.analysis import AnalysisResponse
from app.services.analysis_service import analyze_case

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/cases/{case_id}/analyze", response_model=AnalysisResponse)
def analyze(case_id: str) -> AnalysisResponse:
    return analyze_case(case_id)
