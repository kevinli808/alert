from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.errors import PayloadTooLargeError, UnsupportedMediaTypeError
from app.dependencies.auth import get_detection_user, rate_limit
from app.dependencies.services import get_services
from app.schemas.common import error_responses
from app.schemas.detection import (
    DetectionResponse,
    InferenceInfo,
    ScoreItem,
    SoundTypeInfo,
    SoundTypesResponse,
)
from app.services.auth_service import AuthenticatedUser
from app.services.container import ServiceContainer
from app.services.databricks_service import DatabricksClassifier
from app.services.prediction_processor import DetectionOutcome
from app.services.sound_categories import SOUND_CATEGORIES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["detection"])

# Browsers label the same audio differently; the real check is whether it decodes.
ALLOWED_NON_AUDIO_TYPES = {"application/octet-stream", "video/webm", "video/mp4", "video/ogg"}


@router.get("/sound-types", response_model=SoundTypesResponse, summary="List supported sound categories")
def list_sound_types(services: ServiceContainer = Depends(get_services)) -> SoundTypesResponse:
    predictions = services.detection.predictions
    return SoundTypesResponse(
        sound_types=[
            SoundTypeInfo(
                id=c.id,
                display_name=c.display_name,
                priority=c.priority,
                default_color=c.default_color,
                default_flash_speed=c.default_flash_speed,
                model_labels=list(c.yamnet_labels),
                confidence_threshold=predictions.threshold_for(c.id),
            )
            for c in SOUND_CATEGORIES
        ]
    )


@router.post(
    "/wake-model",
    summary="Start the sound model now (call when the microphone turns on)",
    description=(
        "With Databricks Model Serving and scale-to-zero, the endpoint sleeps while nobody is listening. "
        "The frontend calls this when the microphone turns on so the model starts before the first "
        "audio chunk arrives. Returns `ready`, `waking` (try detection anyway; chunks return "
        "`model_unavailable` until it is up) or the local model's status."
    ),
    responses=error_responses(401, 429),
)
async def wake_model(
    user: AuthenticatedUser = Depends(get_detection_user),
    _rl: None = Depends(rate_limit("default", detection=True)),
    services: ServiceContainer = Depends(get_services),
) -> dict[str, str]:
    classifier = services.detection.classifier
    remote = getattr(classifier, "primary", classifier)  # FallbackClassifier wraps the remote one
    if isinstance(remote, DatabricksClassifier):
        return {"status": await run_in_threadpool(remote.wake)}
    return {"status": classifier.status}


@router.post(
    "/detect-sound",
    response_model=DetectionResponse,
    summary="Classify a short audio chunk",
    description=(
        "Upload one audio chunk (multipart field `audio`, 0.5-10 s; 16 kHz mono 16-bit WAV recommended). "
        "Audio is processed in memory and never stored. Trigger the visual alert when "
        "`detected && is_new_event`."
    ),
    responses=error_responses(401, 413, 415, 422, 429, 502, 503),
)
async def detect_sound(
    audio: UploadFile = File(..., description="Audio chunk (WAV/FLAC/OGG; WebM/MP4 if ffmpeg is installed)"),
    user: AuthenticatedUser = Depends(get_detection_user),
    _rl: None = Depends(rate_limit("detect", detection=True)),
    services: ServiceContainer = Depends(get_services),
) -> DetectionResponse:
    content_type = (audio.content_type or "application/octet-stream").split(";")[0].strip().lower()
    if not (content_type.startswith("audio/") or content_type in ALLOWED_NON_AUDIO_TYPES):
        raise UnsupportedMediaTypeError(f"Unsupported content type '{content_type}'")

    max_bytes = services.settings.max_upload_bytes
    data = await audio.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"Audio exceeds the {max_bytes}-byte limit")

    # CPU-bound decode + inference runs in a worker thread so the event loop stays responsive.
    outcome = await run_in_threadpool(services.detection.detect, data, user.user_id)
    del data  # raw audio is never persisted or logged; detections are not stored either

    # Labels and scores only (never audio): enough to see why a sound did or didn't trigger.
    if outcome.candidates and (outcome.detected or outcome.candidates[0][1] >= 0.2):
        logger.info(
            "chunk: %s %s (%.2f) | categories %s | model %s",
            "DETECTED" if outcome.detected else "below threshold:",
            outcome.sound_type,
            outcome.confidence,
            ", ".join(f"{k}={v:.2f}" for k, v in outcome.candidates[:3]),
            ", ".join(f"{k}={v:.2f}" for k, v in outcome.top_labels[:3]),
        )

    return _to_response(outcome)


def _to_response(o: DetectionOutcome) -> DetectionResponse:
    return DetectionResponse(
        detected=o.detected,
        sound_type=o.sound_type,
        confidence=o.confidence,
        timestamp=o.timestamp,
        reason=o.reason,
        is_new_event=o.is_new_event,
        event_id=o.event_id,
        event_started_at=o.event_started_at,
        candidates=[ScoreItem(label=k, confidence=v) for k, v in o.candidates],
        model_labels=[ScoreItem(label=k, confidence=v) for k, v in o.top_labels],
        inference=InferenceInfo(
            provider=o.provider,
            model=o.model,
            preprocess_ms=o.timings_ms.get("preprocess", 0.0),
            inference_ms=o.timings_ms.get("inference", 0.0),
            total_ms=o.timings_ms.get("total", 0.0),
        ),
    )
