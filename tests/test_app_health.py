"""Tests for FastAPI app, health endpoint, and error boundary (Track 2)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """GET /health contract."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["status"] == "ok"

    def test_health_service_name(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "recoverai-api"


class TestErrorBoundary:
    """Structured error responses."""

    def test_404_for_unknown_route(self, client):
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_error_response_schema(self, client):
        response = client.get("/api/nonexistent")
        data = response.json()
        assert "detail" in data

    def test_method_not_allowed(self, client):
        response = client.post("/health")
        assert response.status_code == 405
