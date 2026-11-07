"""Builds and holds the long-lived service objects (one set per process)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.security import RateLimiter
from app.services.audio_processor import AudioProcessor
from app.services.auth_service import Auth0TokenVerifier
from app.services.database_service import Database, SupabaseDatabase
from app.services.databricks_service import DatabricksClassifier
from app.services.detection_service import DetectionService
from app.services.prediction_processor import PredictionProcessor
from app.services.sound_classifier import FallbackClassifier, SoundClassifier, YamnetClassifier

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    settings: Settings
    token_verifier: Auth0TokenVerifier | None
    database: Database | None
    detection: DetectionService
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    known_profiles: set[str] = field(default_factory=set)

    async def aclose(self) -> None:
        if self.database is not None:
            await self.database.aclose()
        self.detection.close()


def build_classifier(settings: Settings) -> SoundClassifier:
    local = YamnetClassifier(settings.yamnet_model_handle, settings.score_aggregation)
    if settings.inference_provider == "local":
        return local
    remote = DatabricksClassifier(
        settings.databricks_host,
        settings.databricks_token.get_secret_value(),
        settings.databricks_endpoint_name,
        settings.databricks_timeout_seconds,
    )
    if not settings.databricks_configured:
        logger.warning("INFERENCE_PROVIDER=databricks but Databricks is not configured")
    return FallbackClassifier(remote, local) if settings.databricks_fallback_to_local else remote


def build_services(settings: Settings) -> ServiceContainer:
    verifier = None
    if settings.auth0_configured:
        verifier = Auth0TokenVerifier(
            settings.auth0_domain,
            settings.auth0_audience,
            settings.auth0_claim_namespace,
            settings.jwt_leeway_seconds,
        )
    else:
        logger.warning("Auth0 is not configured: protected endpoints will return 503")

    database = None
    if settings.supabase_configured:
        database = SupabaseDatabase(
            settings.supabase_url,
            settings.supabase_service_key.get_secret_value(),
            settings.database_timeout_seconds,
        )
    else:
        logger.warning("Supabase is not configured: database endpoints will return 503")

    detection = DetectionService(
        AudioProcessor(
            max_bytes=settings.max_upload_bytes,
            min_seconds=settings.min_audio_seconds,
            max_seconds=settings.max_audio_seconds,
            silence_rms_threshold=settings.silence_rms_threshold,
        ),
        build_classifier(settings),
        PredictionProcessor(
            threshold=settings.confidence_threshold,
            cooldown_seconds=settings.detection_cooldown_seconds,
            category_thresholds=settings.category_threshold_map,
        ),
    )
    return ServiceContainer(settings=settings, token_verifier=verifier, database=database, detection=detection)
