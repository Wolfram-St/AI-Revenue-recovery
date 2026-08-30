"""Recovery agent API routes.

Exposes the recovery agent pipeline and webhook ingestion via HTTP endpoints:

- ``POST /api/recovery/run`` — run the agent on synthetic or live data
- ``GET  /api/recovery/status`` — current agent status and audit summary
- ``GET  /api/recovery/audit`` — full audit trail
- ``POST /api/recovery/simulate`` — run with configurable synthetic failures
- ``POST /api/recovery/webhook`` — ingest real-time Razorpay webhooks
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from recovery.failure_classifier import classify_failure
from recovery.recovery_agent import (
    AgentConfig,
    BatchResult,
    RecoveryAgent,
    generate_synthetic_failures,
)
from recovery.recovery_strategy import (
    StrategyContext,
    select_recovery_action,
)
from recovery.razorpay_client import (
    RazorpayAPIError,
    WebhookEvent,
    parse_webhook,
)

router = APIRouter(prefix="/api/recovery", tags=["recovery"])

# Global agent state (reset on server restart)
_agent: RecoveryAgent | None = None
_last_result: BatchResult | None = None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    """Request body for ``POST /api/recovery/simulate``."""

    count: int = Field(default=20, ge=1, le=500, description="Number of synthetic failures to generate")
    budget_limit_inr: float = Field(default=1000.0, gt=0, description="Budget limit in INR")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries per payment")


class RunLiveRequest(BaseModel):
    """Request body for ``POST /api/recovery/run`` (live mode)."""

    from_ts: int | None = Field(default=None, description="Start Unix timestamp")
    to_ts: int | None = Field(default=None, description="End Unix timestamp")
    budget_limit_inr: float = Field(default=5000.0, gt=0, description="Budget limit in INR")


class RecoveryStatusResponse(BaseModel):
    """Response for ``GET /api/recovery/status``."""

    mode: str
    total_processed: int
    total_amount_inr: float
    recovered_count: int
    recovered_inr: float
    recovery_rate_pct: float
    escalated_count: int
    stopped_count: int
    budget_used_inr: float
    budget_remaining_inr: float
    audit_summary: dict[str, Any]


class WebhookIngestResponse(BaseModel):
    """Response for ``POST /api/recovery/webhook``."""

    status: str
    event: str
    entity_type: str
    entity_id: str
    amount_inr: float
    classification: dict[str, Any] | None = None
    recommended_action: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def get_status() -> RecoveryStatusResponse:
    """Get current agent status and audit summary."""
    if _agent is None or _last_result is None:
        return RecoveryStatusResponse(
            mode="idle",
            total_processed=0,
            total_amount_inr=0.0,
            recovered_count=0,
            recovered_inr=0.0,
            recovery_rate_pct=0.0,
            escalated_count=0,
            stopped_count=0,
            budget_used_inr=0.0,
            budget_remaining_inr=0.0,
            audit_summary={},
        )

    d = _last_result.to_dict()
    return RecoveryStatusResponse(
        mode="simulation" if _agent._config.simulation else "live",
        total_processed=d["total_processed"],
        total_amount_inr=d["total_amount_inr"],
        recovered_count=d["recovered_count"],
        recovered_inr=d["recovered_inr"],
        recovery_rate_pct=d["recovery_rate_pct"],
        escalated_count=d["escalated_count"],
        stopped_count=d["stopped_count"],
        budget_used_inr=d["budget_used_inr"],
        budget_remaining_inr=d["budget_remaining_inr"],
        audit_summary=d["audit_summary"],
    )


@router.post("/simulate")
def run_simulation(req: SimulateRequest) -> dict[str, Any]:
    """Run the recovery agent on synthetic data.

    No Razorpay credentials needed — this generates realistic failure
    scenarios and runs the full pipeline in simulation mode.
    """
    global _agent, _last_result

    config = AgentConfig(
        budget_limit_paise=int(round(req.budget_limit_inr * 100)),
        max_retries_per_payment=req.max_retries,
        simulation=True,
    )
    _agent = RecoveryAgent.simulation(config=config)

    failures = generate_synthetic_failures(count=req.count)
    _last_result = _agent.run_batch(failures)

    return _last_result.to_dict()


@router.post("/run")
def run_live(req: RunLiveRequest) -> dict[str, Any]:
    """Run the recovery agent on live Razorpay data.

    Requires ``RAZORPAY_KEY_ID`` and ``RAZORPAY_KEY_SECRET`` environment
    variables. If not configured, returns a 500 error with instructions.
    """
    global _agent, _last_result

    try:
        config = AgentConfig(
            budget_limit_paise=int(round(req.budget_limit_inr * 100)),
            max_retries_per_payment=3,
            human_review_capacity=10,
            simulation=False,
        )
        _agent = RecoveryAgent.from_env(config=config)
        _last_result = _agent.run_live(from_ts=req.from_ts, to_ts=req.to_ts)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Razorpay credentials not configured: {exc}. "
                   "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in your .env file.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recovery agent failed: {exc}")

    return _last_result.to_dict()


@router.get("/audit")
def get_audit() -> list[dict[str, Any]]:
    """Get the full audit trail from the last agent run."""
    if _agent is None:
        return []
    return [entry.to_dict() for entry in _agent.audit.entries]


@router.post("/webhook", response_model=WebhookIngestResponse)
async def ingest_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(None, alias="X-Razorpay-Signature"),
) -> WebhookIngestResponse:
    """Ingest and process real-time Razorpay webhooks.

    Verifies the HMAC-SHA256 signature against ``RAZORPAY_WEBHOOK_SECRET``,
    parses the event payload, diagnoses root cause on payment failures, and
    determines the recovery action.
    """
    raw_body = await request.body()
    webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip() or None

    try:
        event = parse_webhook(
            body=raw_body,
            signature=x_razorpay_signature,
            secret=webhook_secret,
            verify=bool(webhook_secret),
        )
    except RazorpayAPIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    classification_dict = None
    action_dict = None

    if event.event == "payment.failed" and event.entity_type == "payment":
        payment_dict = event.raw_payload.get("payload", {}).get("payment", {}).get("entity", {})
        classification = classify_failure(payment_dict)
        classification_dict = {
            "category": classification.category.value,
            "confidence": classification.confidence,
            "reason": classification.reason,
            "raw_error_code": classification.raw_error_code,
            "raw_error_reason": classification.raw_error_reason,
        }

        ctx = StrategyContext(
            classification=classification,
            amount_paise=event.amount_paise,
            retry_count=0,
            max_retries=3,
        )
        action = select_recovery_action(ctx)
        action_dict = {
            "action_type": action.action_type.value,
            "reason": action.reason,
            "estimated_cost_paise": action.estimated_cost_paise,
        }

    return WebhookIngestResponse(
        status="processed",
        event=event.event,
        entity_type=event.entity_type,
        entity_id=event.entity_id,
        amount_inr=event.amount_paise / 100.0,
        classification=classification_dict,
        recommended_action=action_dict,
    )
