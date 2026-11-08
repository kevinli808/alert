"""Application configuration loaded from environment variables / .env.

Every secret is a ``SecretStr`` so it is never printed in logs or reprs.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General -----------------------------------------------------------------
    app_name: str = "Alert API"
    environment: Literal["development", "production", "test"] = Field(
        default_factory=lambda: "production" if os.getenv("VERCEL") else "development"
    )
    log_level: str = "INFO"
    # Comma-separated list of allowed frontend origins (no wildcards in production).
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
    # Local development only: when Auth0 is NOT configured, let /api/detect-sound run without a token
    # (as one shared anonymous user, with no history saved) so the frontend works before Auth0 is set up.
    # Ignored in test/production and whenever Auth0 is configured.
    allow_anonymous_detection: bool = True

    # --- Auth0 ---------------------------------------------------------------------
    auth0_domain: str = ""  # e.g. "your-tenant.us.auth0.com" (no scheme, no trailing slash)
    auth0_audience: str = ""  # API identifier, e.g. "https://api.deaf-accessibility.app"
    # Namespace used by the Auth0 Action that adds email/name to access tokens.
    auth0_claim_namespace: str = "https://deaf-accessibility.app"
    jwt_leeway_seconds: int = Field(default=30, ge=0, le=300)

    # --- Supabase ------------------------------------------------------------------
    supabase_url: str = ""  # e.g. "https://abcd1234.supabase.co"
    supabase_service_key: SecretStr = SecretStr("")  # service_role / sb_secret_ key. Server-side only.
    database_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    # --- Audio input ---------------------------------------------------------------
    max_upload_bytes: int = Field(default=2_000_000, gt=0, le=20_000_000)
    min_audio_seconds: float = Field(default=0.5, gt=0)
    max_audio_seconds: float = Field(default=10.0, gt=0, le=60)
    # RMS (linear, full scale = 1.0) below which a chunk is treated as silence and not classified.
    silence_rms_threshold: float = Field(default=0.003, ge=0, le=1)

    # --- Inference -----------------------------------------------------------------
    inference_provider: Literal["local", "databricks"] = Field(
        default_factory=lambda: "databricks" if os.getenv("VERCEL") else "local"
    )
    yamnet_model_handle: str = "https://tfhub.dev/google/yamnet/1"
    score_aggregation: Literal["max", "mean"] = "max"
    preload_model: bool = Field(default_factory=lambda: not bool(os.getenv("VERCEL")))

    # --- Prediction processing ----------------------------------------------------
    confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    # Optional JSON object of per-category overrides, e.g. {"glass_breaking": 0.6}
    category_thresholds: str = ""
    detection_cooldown_seconds: float = Field(default=10.0, ge=0, le=600)
    record_history: bool = True

    # --- Databricks (optional) ----------------------------------------------------
    databricks_host: str = ""  # e.g. "https://adb-1234567890123456.7.azuredatabricks.net"
    databricks_token: SecretStr = SecretStr("")
    databricks_endpoint_name: str = ""
    databricks_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    databricks_fallback_to_local: bool = Field(default_factory=lambda: not bool(os.getenv("VERCEL")))

    # --- Rate limiting (per authenticated user, per minute) -------------------------
    rate_limit_detect_per_minute: int = Field(default=120, ge=1)
    rate_limit_default_per_minute: int = Field(default=60, ge=1)

    @field_validator("auth0_domain")
    @classmethod
    def _strip_domain(cls, value: str) -> str:
        value = value.strip().removeprefix("https://").removeprefix("http://")
        return value.rstrip("/")

    @field_validator("auth0_audience")
    @classmethod
    def _strip_audience(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def auth0_configured(self) -> bool:
        return bool(self.auth0_domain and self.auth0_audience)

    @property
    def anonymous_detection_enabled(self) -> bool:
        return self.allow_anonymous_detection and self.environment == "development" and not self.auth0_configured

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key.get_secret_value())

    @property
    def databricks_configured(self) -> bool:
        return bool(
            self.databricks_host
            and self.databricks_endpoint_name
            and self.databricks_token.get_secret_value()
        )

    @property
    def category_threshold_map(self) -> dict[str, float]:
        if not self.category_thresholds.strip():
            return {}
        try:
            raw = json.loads(self.category_thresholds)
        except json.JSONDecodeError as exc:
            raise ValueError("CATEGORY_THRESHOLDS must be a JSON object") from exc
        if not isinstance(raw, dict):
            raise ValueError("CATEGORY_THRESHOLDS must be a JSON object")
        result: dict[str, float] = {}
        for key, val in raw.items():
            fval = float(val)
            if not 0.0 <= fval <= 1.0:
                raise ValueError(f"Threshold for {key!r} must be between 0 and 1")
            result[str(key)] = fval
        return result

    def validate_for_startup(self) -> None:
        """Fail fast on configurations that are unsafe to run."""
        if self.min_audio_seconds >= self.max_audio_seconds:
            raise ValueError("MIN_AUDIO_SECONDS must be smaller than MAX_AUDIO_SECONDS")
        _ = self.category_threshold_map  # raises on invalid JSON
        if self.environment == "production":
            missing = []
            if not self.auth0_configured:
                missing.append("AUTH0_DOMAIN/AUTH0_AUDIENCE")
            if not self.supabase_configured:
                missing.append("SUPABASE_URL/SUPABASE_SERVICE_KEY")
            if self.inference_provider == "databricks" and not self.databricks_configured:
                missing.append("DATABRICKS_HOST/DATABRICKS_TOKEN/DATABRICKS_ENDPOINT_NAME")
            if os.getenv("VERCEL") and self.inference_provider != "databricks":
                raise ValueError("Vercel deployments require INFERENCE_PROVIDER=databricks")
            if os.getenv("VERCEL") and self.databricks_fallback_to_local:
                raise ValueError("Vercel deployments require DATABRICKS_FALLBACK_TO_LOCAL=false")
            if any("*" in o for o in self.cors_origin_list):
                raise ValueError("Wildcard CORS origins are not allowed in production")
            if missing:
                raise ValueError("Missing required production configuration: " + ", ".join(missing))


@lru_cache
def get_settings() -> Settings:
    return Settings()
