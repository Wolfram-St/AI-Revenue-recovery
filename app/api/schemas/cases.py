"""Recovery case API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CaseSummary(BaseModel):
    attempt_id: str
    payment_id: str
    customer_id: str
    amount_inr: float
    failure_category: str
    scoring_recommendation: str
    authorized_action: str
    expected_recovery_value_inr: float
    recovery_probability: float
    is_stop: bool


class CaseListResponse(BaseModel):
    cases: list[CaseSummary]
    total: int
    page: int
    page_size: int
