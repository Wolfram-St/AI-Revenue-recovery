"""Tests for Counterfactual Uplift Ledger and Multi-Touch Causal Attribution.

Covers:
- 5% randomized holdout partitioning (assign_holdout_arm)
- Multi-touch causal attribution models (Linear, First-Touch, Last-Touch, Time-Decay)
- Counterfactual uplift calculation (True Incremental Recovery)
- Dashboard service and API schema integration
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.main import create_app
from app.services.dashboard_service import get_dashboard
from recovery.counterfactual_ledger import (
    AttributionModel,
    CounterfactualUpliftSummary,
    UpliftLedgerEntry,
    assign_holdout_arm,
    compute_counterfactual_uplift_ledger,
    compute_multi_touch_attribution,
)


class TestHoldoutPartitioning:
    def test_assign_holdout_arm_deterministic(self):
        arm1 = assign_holdout_arm("case_001", holdout_pct=0.05)
        arm2 = assign_holdout_arm("case_001", holdout_pct=0.05)
        assert arm1 == arm2

    def test_holdout_ratio_approximately_5_percent(self):
        cases = [f"case_{i:04d}" for i in range(1000)]
        holdout_count = sum(1 for c in cases if assign_holdout_arm(c, holdout_pct=0.05) == "CONTROL_HOLDOUT")
        # In 1000 cases, expected ~50 (allow 30 to 70 range)
        assert 30 <= holdout_count <= 70


class TestMultiTouchAttribution:
    def test_linear_attribution(self):
        touches = ["WEBHOOK", "WHATSAPP_LINK", "VOICE_BOT"]
        weights = compute_multi_touch_attribution(touches, AttributionModel.LINEAR)
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 0.3333) < 0.001
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_first_touch_attribution(self):
        touches = ["WEBHOOK", "WHATSAPP_LINK", "VOICE_BOT"]
        weights = compute_multi_touch_attribution(touches, AttributionModel.FIRST_TOUCH)
        assert weights["WEBHOOK"] == 1.0
        assert weights["WHATSAPP_LINK"] == 0.0
        assert weights["VOICE_BOT"] == 0.0

    def test_last_touch_attribution(self):
        touches = ["WEBHOOK", "WHATSAPP_LINK", "VOICE_BOT"]
        weights = compute_multi_touch_attribution(touches, AttributionModel.LAST_TOUCH)
        assert weights["VOICE_BOT"] == 1.0
        assert weights["WEBHOOK"] == 0.0

    def test_time_decay_attribution(self):
        touches = ["TOUCH_1", "TOUCH_2"]
        weights = compute_multi_touch_attribution(touches, AttributionModel.TIME_DECAY)
        # Touch 2 should have 2x weight of Touch 1 (1/3 vs 2/3)
        assert weights["TOUCH_2"] > weights["TOUCH_1"]
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestCounterfactualUpliftCalculation:
    def test_counterfactual_ledger_computes_incremental_recovery(self):
        # 10 treatment cases (5 recovered at ₹1,000 each = ₹5,000 gross)
        # 2 control cases (0 recovered -> 0% baseline)
        records = [
            {"case_id": f"t_{i}", "customer_id": f"c_{i}", "amount_inr": 1000.0, "recovered": i < 5, "assigned_arm": "TREATMENT", "touchpoints": ["WHATSAPP", "PTP"]}
            for i in range(10)
        ] + [
            {"case_id": "c_0", "customer_id": "cust_c0", "amount_inr": 1000.0, "recovered": False, "assigned_arm": "CONTROL_HOLDOUT"},
            {"case_id": "c_1", "customer_id": "cust_c1", "amount_inr": 1000.0, "recovered": False, "assigned_arm": "CONTROL_HOLDOUT"},
        ]

        entries, summary = compute_counterfactual_uplift_ledger(records)

        assert summary.total_cases == 12
        assert summary.treatment_count == 10
        assert summary.control_holdout_count == 2
        assert summary.gross_recovered_inr == 5000.0
        assert summary.control_recovery_rate_pct == 0.0
        assert summary.treatment_recovery_rate_pct == 50.0
        assert summary.incremental_uplift_pct == 50.0
        assert summary.true_incremental_recovery_inr == 5000.0

        # Verify multi-touch attribution
        assert "WHATSAPP" in summary.channel_attribution_inr
        assert "PTP" in summary.channel_attribution_inr
        assert summary.channel_attribution_inr["WHATSAPP"] == 2500.0
        assert summary.channel_attribution_inr["PTP"] == 2500.0


class TestDashboardServiceIntegration:
    def test_get_dashboard_returns_counterfactual_uplift(self):
        resp = get_dashboard()
        assert resp.total_cases > 0
        assert resp.revenue_at_risk_inr > 0
        assert resp.treatment_recovery_rate_pct is not None
        assert resp.control_holdout_rate_pct is not None
        assert resp.true_incremental_recovery_inr is not None

    def test_dashboard_api_endpoint(self):
        app = create_app()
        client = TestClient(app)
        res = client.get("/api/dashboard")
        assert res.status_code == 200
        data = res.json()
        assert "treatment_recovery_rate_pct" in data
        assert "true_incremental_recovery_inr" in data
        assert "channel_attribution" in data
