"""Tests for application configuration (Track 1)."""

from __future__ import annotations

import os
import dataclasses

import pytest

from app.config import Settings


class TestSettingsDefaults:
    """Default configuration values."""

    def test_database_url_default(self):
        settings = Settings()
        assert settings.database_url == "postgresql://recoverai:recoverai@localhost:5432/recoverai"

    def test_api_host_default(self):
        settings = Settings()
        assert settings.api_host == "0.0.0.0"

    def test_api_port_default(self):
        settings = Settings()
        assert settings.api_port == 8000

    def test_cors_origins_default(self):
        settings = Settings()
        assert settings.cors_origins == ["http://localhost:5173", "http://localhost:3000"]

    def test_demo_mode_default(self):
        settings = Settings()
        assert settings.demo_mode is True

    def test_service_name(self):
        settings = Settings()
        assert settings.service_name == "recoverai-api"


class TestSettingsFromEnv:
    """Environment variable overrides via from_env()."""

    def test_database_url_override(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://custom:pass@host:5432/db")
        settings = Settings.from_env()
        assert settings.database_url == "postgresql://custom:pass@host:5432/db"

    def test_api_port_override(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        settings = Settings.from_env()
        assert settings.api_port == 9000

    def test_demo_mode_override(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "false")
        settings = Settings.from_env()
        assert settings.demo_mode is False

    def test_cors_origins_json_override(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", '["http://example.com"]')
        settings = Settings.from_env()
        assert settings.cors_origins == ["http://example.com"]

    def test_cors_origins_csv_override(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com")
        settings = Settings.from_env()
        assert settings.cors_origins == ["http://a.com", "http://b.com"]

    def test_from_env_defaults_when_no_env(self, monkeypatch):
        for key in ("DATABASE_URL", "API_HOST", "API_PORT", "CORS_ORIGINS", "DEMO_MODE"):
            monkeypatch.delenv(key, raising=False)
        settings = Settings.from_env()
        assert settings.api_port == 8000
        assert settings.demo_mode is True


class TestSettingsImmutability:
    """Settings should be frozen."""

    def test_settings_are_frozen(self):
        settings = Settings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.api_port = 9999
