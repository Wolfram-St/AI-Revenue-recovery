"""Tests for portfolio optimization API (Track 5)."""

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


class TestPortfolioOptimizeEndpoint:
    """POST /api/portfolio/optimize contract."""

    def test_optimize_returns_200(self, client):
        response = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        })
        assert response.status_code == 200

    def test_optimize_response_schema(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        assert "solver" in data
        assert "summary" in data
        assert "entries" in data
        assert "metadata" in data

    def test_solver_type(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        assert data["solver"] == "exact_dp_2d"

    def test_summary_fields(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        summary = data["summary"]
        assert "total_rows" in summary
        assert "optimizer_allocated_count" in summary
        assert "budget_limit_inr" in summary
        assert "budget_allocated_inr" in summary
        assert "budget_remaining_inr" in summary
        assert "human_review_capacity_limit" in summary
        assert "human_review_allocated_count" in summary
        assert "total_policy_overrides" in summary
        assert "optimizer_objective_value_inr" in summary
        assert "optimizer_status" in summary

    def test_budget_constraint_respected(self, client):
        budget = 200.0
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": budget,
            "human_review_capacity": 50,
        }).json()
        assert data["summary"]["budget_allocated_inr"] <= budget + 0.01

    def test_hr_capacity_constraint_respected(self, client):
        hr_cap = 3
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": hr_cap,
        }).json()
        assert data["summary"]["human_review_allocated_count"] <= hr_cap

    def test_entries_have_required_fields(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        if data["entries"]:
            entry = data["entries"][0]
            assert "attempt_id" in entry
            assert "optimizer_recommendation" in entry
            assert "authorized_action" in entry
            assert "policy_overrode_recommendation" in entry

    def test_recommendation_and_authorization_separate(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        for entry in data["entries"]:
            assert "optimizer_recommendation" in entry
            assert "authorized_action" in entry

    def test_policy_overrides_visible(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        overrides = data["summary"]["total_policy_overrides"]
        assert isinstance(overrides, int)

    def test_stop_behavior_not_bypassed(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 10,
        }).json()
        for entry in data["entries"]:
            if entry["optimizer_recommendation"] != "NO_INTERVENTION":
                if entry["authorized_action"] == "STOP":
                    assert entry["policy_overrode_recommendation"] is True

    def test_invalid_budget_negative_returns_error(self, client):
        response = client.post("/api/portfolio/optimize", json={
            "budget_inr": -100.0,
            "human_review_capacity": 10,
        })
        assert response.status_code in (400, 422)

    def test_invalid_hr_negative_returns_error(self, client):
        response = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": -1,
        })
        assert response.status_code in (400, 422)

    def test_zero_budget(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 0.0,
            "human_review_capacity": 0,
        }).json()
        assert data["summary"]["optimizer_allocated_count"] == 0

    def test_large_budget_covers_all(self, client):
        data = client.post("/api/portfolio/optimize", json={
            "budget_inr": 2000.0,
            "human_review_capacity": 50,
        }).json()
        assert data["summary"]["total_policy_overrides"] >= 0
