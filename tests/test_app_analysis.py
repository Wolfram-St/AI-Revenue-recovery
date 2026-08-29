"""Tests for case analysis API (Track 4)."""

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


@pytest.fixture
def valid_case_id(client):
    data = client.get("/api/cases?page_size=1").json()
    return data["cases"][0]["attempt_id"]


class TestAnalysisEndpoint:
    """POST /api/cases/{case_id}/analyze contract."""

    def test_analyze_returns_200(self, client, valid_case_id):
        response = client.post(f"/api/cases/{valid_case_id}/analyze")
        assert response.status_code == 200

    def test_analyze_response_schema(self, client, valid_case_id):
        data = client.post(f"/api/cases/{valid_case_id}/analyze").json()
        assert "attempt_id" in data
        assert "amount_inr" in data
        assert "failure_category" in data
        assert "recovery_probability" in data
        assert "scoring_recommendation" in data
        assert "expected_recovery_value_inr" in data
        assert "worth_intervening" in data
        assert "candidate_actions" in data
        assert "policy" in data
        assert "audit_context" in data

    def test_recommendation_and_authorization_separate(self, client, valid_case_id):
        data = client.post(f"/api/cases/{valid_case_id}/analyze").json()
        assert "scoring_recommendation" in data
        assert "policy" in data
        assert "authorized_action" in data["policy"]

    def test_policy_has_required_fields(self, client, valid_case_id):
        data = client.post(f"/api/cases/{valid_case_id}/analyze").json()
        policy = data["policy"]
        assert "decision" in policy
        assert "authorized_action" in policy
        assert "reason" in policy
        assert "is_stop" in policy

    def test_policy_block_visible(self, client):
        list_data = client.get("/api/cases?is_stop=true&page_size=1").json()
        if list_data["cases"]:
            case_id = list_data["cases"][0]["attempt_id"]
            data = client.post(f"/api/cases/{case_id}/analyze").json()
            assert data["policy"]["is_stop"] is True
            assert data["policy"]["authorized_action"] == "STOP"

    def test_missing_case_returns_404(self, client):
        response = client.post("/api/cases/NONEXISTENT-999/analyze")
        assert response.status_code == 404

    def test_missing_case_error_shape(self, client):
        response = client.post("/api/cases/NONEXISTENT-999/analyze")
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "CASE_NOT_FOUND"

    def test_analyze_matches_trace_data(self, client, valid_case_id):
        analyze_data = client.post(f"/api/cases/{valid_case_id}/analyze").json()
        case_detail = client.get(f"/api/cases/{valid_case_id}").json()["case"]
        assert analyze_data["attempt_id"] == case_detail["attempt_id"]
        assert analyze_data["amount_inr"] == case_detail["amount_inr"]
        assert analyze_data["scoring_recommendation"] == case_detail["scoring_recommendation"]

    def test_candidate_actions_list(self, client, valid_case_id):
        data = client.post(f"/api/cases/{valid_case_id}/analyze").json()
        assert isinstance(data["candidate_actions"], list)

    def test_audit_context_has_evaluated_rules(self, client, valid_case_id):
        data = client.post(f"/api/cases/{valid_case_id}/analyze").json()
        assert "evaluated_rules" in data["audit_context"]
        assert isinstance(data["audit_context"]["evaluated_rules"], list)
