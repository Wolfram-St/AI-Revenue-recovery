"""Application configuration using environment variables.

Settings are loaded from environment variables with sensible defaults for
local development. Secrets are never exposed to the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import json
import os


def _parse_cors_origins(raw: str | None) -> List[str]:
    if raw is None:
        return ["http://localhost:5173", "http://localhost:3000"]
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("CORS_ORIGINS must be a JSON array")
        return parsed
    except (json.JSONDecodeError, ValueError):
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    """Frozen application settings loaded from environment variables."""

    database_url: str = "postgresql://recoverai:recoverai@localhost:5432/recoverai"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])
    demo_mode: bool = True
    service_name: str = "recoverai-api"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.environ.get("DATABASE_URL", cls.__dataclass_fields__["database_url"].default),
            api_host=os.environ.get("API_HOST", cls.__dataclass_fields__["api_host"].default),
            api_port=_parse_int(os.environ.get("API_PORT"), cls.__dataclass_fields__["api_port"].default),
            cors_origins=_parse_cors_origins(os.environ.get("CORS_ORIGINS")),
            demo_mode=_parse_bool(os.environ.get("DEMO_MODE"), cls.__dataclass_fields__["demo_mode"].default),
            service_name=os.environ.get("SERVICE_NAME", cls.__dataclass_fields__["service_name"].default),
        )
