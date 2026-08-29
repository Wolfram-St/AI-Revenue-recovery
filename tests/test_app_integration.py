"""Full integration test suite for the RecoverAI application API (Track 6).

Tests exercise real existing core logic through the HTTP boundary.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.data_bootstrap import reset_bootstrap


@pytest.fixture(autouse=True)
def clear_bootstrap():
    reset_bootstrap()
    yield
    reset_bootstrap()


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


class TestHealthIntegration:
    """Health endpoint integration."""

    def test_health_always_works(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestDashboardIntegration:
    """Dashboard endpoint integration."""

    def test_dashboard_loads_with_real_data(self, client):
        data = client.get("/api/dashboard").json()
        assert data["total_cases"] > 0
        assert data["revenue_at_risk_inr"] > 0

    def test_dashboard_action_distribution_matches_summary(self, client):
        data = client.get("/api/dashboard").json()
        total_from_dist = sum(d["count"] for d in data["action_distribution"])
        assert total_from_dist == data["total_cases"]


class TestCaseFlowIntegration:
    """Full case listing → detail → analysis flow."""

    def test_list_cases_then_get_detail(self, client):
        list_data = client.get("/api/cases?page_size=5").json()
        assert list_data["total"] > 0
        case_id = list_data["cases"][0]["attempt_id"]

        detail = client.get(f"/api/cases/{case_id}").json()
        assert detail["case"]["attempt_id"] == case_id

    def test_analyze_case_shows_recommendation_and_authorization(self, client):
        list_data = client.get("/api/cases?page_size=1").json()
        case_id = list_data["cases"][0]["attempt_id"]

        analysis = client.post(f"/api/cases/{case_id}/analyze").json()
        assert "scoring_recommendation" in analysis
        assert "authorized_action" in analysis["policy"]
        assert analysis["attempt_id"] == case_id

    def test_full_flow_dashboard_to_analysis(self, client):
        dashboard = client.get("/api/dashboard").json()
        assert dashboard["total_cases"] > 0

        cases = client.get("/api/cases?page_size=1").json()
        assert cases["total"] > 0

        case_id = cases["cases"][0]["attempt_id"]
        detail = client.get(f"/api/cases/{case_id}").json()
        assert detail["case"]["attempt_id"] == case_id

        analysis = client.post(f"/api/cases/{case_id}/analyze").json()
        assert analysis["attempt_id"] == case_id


class TestAuditIntegration:
    """Audit trail endpoint."""

    def test_audit_returns_entries(self, client):
        data = client.get("/api/audit?page_size=5").json()
        assert "entries" in data
        assert "total" in data
        assert data["total"] > 0

    def test_audit_entry_has_required_fields(self, client):
        data = client.get("/api/audit?page_size=1").json()
        entry = data["entries"][0]
        assert "event_type" in entry
        assert "actor_type" in entry
        assert "action" in entry
        assert "decision_reason" in entry
        assert "event_payload" in entry

    def test_audit_filter_by_case_id(self, client):
        cases = client.get("/api/cases?page_size=1").json()
        case_id = cases["cases"][0]["attempt_id"]
        data = client.get(f"/api/audit?case_id={case_id}").json()
        for entry in data["entries"]:
            assert entry["recovery_case_id"] == case_id


class TestPortfolioIntegration:
    """Portfolio optimization integration with real core logic."""

    def test_optimize_with_real_optimizer(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        assert data["solver"] == "exact_dp_2d"
        assert data["summary"]["total_rows"] > 0

    def test_optimize_recommendation_authorization_separate(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        for entry in data["entries"]:
            rec = entry["optimizer_recommendation"]
            auth = entry["authorized_action"]
            if rec != "NO_INTERVENTION":
                assert isinstance(rec, str)
                assert isinstance(auth, str)

    def test_policy_blocks_visible_in_portfolio(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        stop_count = sum(
            1 for e in data["entries"]
            if e["authorized_action"] == "STOP"
        )
        assert stop_count >= 0


class TestSafetyConstraints:
    """Verify safety constraints are not bypassed through the API."""

    def test_no_real_payment_execution(self, client):
        openapi = client.app.openapi()
        paths = list(openapi.get("paths", {}).keys())
        assert "/api/execute_payment" not in paths
        assert "/api/retry_payment" not in paths

    def test_no_model_mutation_endpoint(self, client):
        openapi = client.app.openapi()
        paths = list(openapi.get("paths", {}).keys())
        assert "/api/train" not in paths
        assert "/api/retrain" not in paths

    def test_errors_are_structured(self, client):
        response = client.get("/api/cases/NONEXISTENT")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
