"""Analysis API schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PolicyInfo(BaseModel):
    decision: str
    authorized_action: str
    reason: str
    matched_rule_id: str | None
    matched_rule_name: str | None
    is_stop: bool


class CandidateAction(BaseModel):
    arm: str
    probability: float
    expected_recovery_value_inr: float


class AnalysisResponse(BaseModel):
    attempt_id: str
    amount_inr: float
    failure_category: str
    recovery_probability: float
    scoring_recommendation: str
    expected_recovery_value_inr: float
    worth_intervening: bool
    candidate_actions: list[CandidateAction]
    policy: PolicyInfo
    audit_context: dict[str, Any]
