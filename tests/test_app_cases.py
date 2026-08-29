"""Tests for dashboard and recovery case APIs (Track 3)."""

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


class TestDashboardEndpoint:
    """GET /api/dashboard contract."""

    def test_dashboard_returns_200(self, client):
        response = client.get("/api/dashboard")
        assert response.status_code == 200

    def test_dashboard_response_schema(self, client):
        data = client.get("/api/dashboard").json()
        assert "total_cases" in data
        assert "revenue_at_risk_inr" in data
        assert "estimated_recoverable_value_inr" in data
        assert "candidate_count" in data
        assert "stop_count" in data
        assert "action_distribution" in data
        assert "demo_mode" in data

    def test_dashboard_has_positive_case_count(self, client):
        data = client.get("/api/dashboard").json()
        assert data["total_cases"] > 0

    def test_dashboard_action_distribution_is_list(self, client):
        data = client.get("/api/dashboard").json()
        assert isinstance(data["action_distribution"], list)
        for item in data["action_distribution"]:
            assert "action" in item
            assert "count" in item

    def test_dashboard_demo_mode_true(self, client):
        data = client.get("/api/dashboard").json()
        assert data["demo_mode"] is True


class TestCaseListEndpoint:
    """GET /api/cases contract."""

    def test_cases_returns_200(self, client):
        response = client.get("/api/cases")
        assert response.status_code == 200

    def test_cases_response_schema(self, client):
        data = client.get("/api/cases").json()
        assert "cases" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    def test_cases_has_entries(self, client):
        data = client.get("/api/cases").json()
        assert data["total"] > 0
        assert len(data["cases"]) > 0

    def test_cases_pagination(self, client):
        data = client.get("/api/cases?page=1&page_size=5").json()
        assert len(data["cases"]) <= 5
        assert data["page"] == 1
        assert data["page_size"] == 5

    def test_cases_second_page(self, client):
        page1 = client.get("/api/cases?page=1&page_size=10").json()
        page2 = client.get("/api/cases?page=2&page_size=10").json()
        if page1["total"] > 10:
            assert page2["cases"] != page1["cases"]

    def test_cases_filter_by_failure_category(self, client):
        data = client.get("/api/cases?failure_category=temporary_decline").json()
        for case in data["cases"]:
            assert case["failure_category"] == "temporary_decline"

    def test_cases_filter_by_recommendation(self, client):
        data = client.get("/api/cases?recommendation=INTERVENE").json()
        for case in data["cases"]:
            assert case["scoring_recommendation"] == "INTERVENE"

    def test_cases_filter_by_is_stop(self, client):
        data = client.get("/api/cases?is_stop=true").json()
        for case in data["cases"]:
            assert case["is_stop"] is True

    def test_case_summary_fields(self, client):
        data = client.get("/api/cases?page_size=1").json()
        case = data["cases"][0]
        assert "attempt_id" in case
        assert "payment_id" in case
        assert "amount_inr" in case
        assert "failure_category" in case
        assert "scoring_recommendation" in case
        assert "authorized_action" in case
        assert "expected_recovery_value_inr" in case
        assert "recovery_probability" in case


class TestCaseDetailEndpoint:
    """GET /api/cases/{case_id} contract."""

    def test_case_detail_returns_200(self, client):
        list_data = client.get("/api/cases?page_size=1").json()
        case_id = list_data["cases"][0]["attempt_id"]
        response = client.get(f"/api/cases/{case_id}")
        assert response.status_code == 200

    def test_case_detail_has_required_sections(self, client):
        list_data = client.get("/api/cases?page_size=1").json()
        case_id = list_data["cases"][0]["attempt_id"]
        data = client.get(f"/api/cases/{case_id}").json()
        assert "case" in data
        assert "audit_history" in data

    def test_case_detail_has_core_fields(self, client):
        list_data = client.get("/api/cases?page_size=1").json()
        case_id = list_data["cases"][0]["attempt_id"]
        data = client.get(f"/api/cases/{case_id}").json()
        case = data["case"]
        assert case["attempt_id"] == case_id
        assert "amount_inr" in case
        assert "scoring_recommendation" in case
        assert "authorized_action" in case

    def test_case_detail_recommendation_and_authorization_separate(self, client):
        list_data = client.get("/api/cases?page_size=1").json()
        case_id = list_data["cases"][0]["attempt_id"]
        data = client.get(f"/api/cases/{case_id}").json()
        case = data["case"]
        assert "scoring_recommendation" in case
        assert "authorized_action" in case

    def test_case_detail_missing_returns_404(self, client):
        response = client.get("/api/cases/NONEXISTENT-123")
        assert response.status_code == 404

    def test_case_detail_404_error_shape(self, client):
        response = client.get("/api/cases/NONEXISTENT-123")
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert data["error"]["code"] == "CASE_NOT_FOUND"
