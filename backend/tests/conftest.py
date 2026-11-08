"""Shared fixtures. Test doubles live here only — never in the production code path."""

from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import jwt
import numpy as np
import pytest
import soundfile as sf
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.audio_processor import AudioProcessor
from app.services.auth_service import Auth0TokenVerifier
from app.services.container import ServiceContainer
from app.services.database_service import Database
from app.services.detection_service import DetectionService
from app.services.prediction_processor import PredictionProcessor
from app.services.sound_classifier import ClassificationResult, SoundClassifier
from app.services.sound_categories import CATEGORY_IDS

DOMAIN = "test-tenant.us.auth0.com"
AUDIENCE = "https://api.test"
NAMESPACE = "https://deaf-accessibility.app"


# ---------------------------------------------------------------- auth -----------------
class _Key:
    def __init__(self, key: Any) -> None:
        self.key = key


class StaticJwks:
    """Stands in for PyJWKClient: resolves the token's `kid` to a local public key."""

    def __init__(self, keys: dict[str, Any]) -> None:
        self.keys = keys

    def get_signing_key_from_jwt(self, token: str) -> _Key:
        kid = jwt.get_unverified_header(token).get("kid")
        if kid not in self.keys:
            raise jwt.PyJWKClientError("Unable to find a signing key that matches")
        return _Key(self.keys[kid])


@pytest.fixture(scope="session")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def verifier(rsa_key) -> Auth0TokenVerifier:
    return Auth0TokenVerifier(DOMAIN, AUDIENCE, NAMESPACE, 0, jwks_client=StaticJwks({"k1": rsa_key.public_key()}))


@pytest.fixture
def make_token(rsa_key):
    def _make(sub: str = "auth0|user-a", *, kid: str = "k1", key=None, alg: str = "RS256", **overrides) -> str:
        now = int(time.time())
        claims = {
            "sub": sub,
            "iss": f"https://{DOMAIN}/",
            "aud": [AUDIENCE, f"https://{DOMAIN}/userinfo"],
            "iat": now,
            "exp": now + 3600,
            f"{NAMESPACE}/email": f"{(sub or 'x').split('|')[-1]}@example.com",
            f"{NAMESPACE}/name": "Test User",
        }
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, key or rsa_key, algorithm=alg, headers={"kid": kid})

    return _make


# ---------------------------------------------------------------- database ---------------
class InMemoryDatabase(Database):
    def __init__(self) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}
        self.settings: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []

    async def get_profile(self, user_id):
        return self.profiles.get(user_id)

    async def create_profile_if_missing(self, user_id, email, username):
        return self.profiles.setdefault(
            user_id, {"id": user_id, "email": email, "username": username, "created_at": datetime.now(timezone.utc)}
        )

    async def get_settings(self, user_id):
        row = self.settings.get(user_id)
        return dict(row) if row else None

    async def upsert_settings(self, user_id, settings):
        assert user_id in self.profiles, "FK violation: profile must exist"
        self.settings[user_id] = {"settings": settings, "updated_at": datetime.now(timezone.utc)}
        return dict(self.settings[user_id])

    async def insert_alert_history(self, user_id, sound_type, confidence, detected_at):
        assert user_id in self.profiles, "FK violation: profile must exist"
        self.history.append(
            {"id": str(uuid.uuid4()), "user_id": user_id, "sound_type": sound_type,
             "confidence": confidence, "detected_at": detected_at}
        )

    async def list_alert_history(self, user_id, limit, before=None):
        rows = [h for h in self.history if h["user_id"] == user_id and (before is None or h["detected_at"] < before)]
        rows.sort(key=lambda h: h["detected_at"], reverse=True)
        return [{k: v for k, v in r.items() if k != "user_id"} for r in rows[:limit]]

    async def delete_alert_history(self, user_id):
        self.history = [h for h in self.history if h["user_id"] != user_id]


# ---------------------------------------------------------------- classifier -------------
class ScriptedClassifier(SoundClassifier):
    """Returns pre-set category scores. TEST DOUBLE ONLY."""

    name = "scripted"

    def __init__(self) -> None:
        self.scores: dict[str, float] = {}
        self.error: Exception | None = None
        self.calls = 0

    @property
    def status(self) -> str:
        return "ready"

    def set(self, **scores: float) -> None:
        self.scores = scores

    def classify(self, waveform):
        self.calls += 1
        if self.error:
            raise self.error
        full = {c: 0.0 for c in CATEGORY_IDS} | self.scores
        return ClassificationResult(category_scores=full, top_labels=[("Fire alarm", 0.9)], provider=self.name)


# ---------------------------------------------------------------- audio helpers ----------
def wav_bytes(seconds: float = 1.0, sr: int = 16_000, channels: int = 1, freq: float = 1000.0, amp: float = 0.5) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------- app --------------------
@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        auth0_domain=DOMAIN,
        auth0_audience=AUDIENCE,
        preload_model=False,
        detection_cooldown_seconds=10,
        rate_limit_detect_per_minute=1000,
        rate_limit_default_per_minute=1000,
    )


@pytest.fixture
def db() -> InMemoryDatabase:
    return InMemoryDatabase()


@pytest.fixture
def classifier() -> ScriptedClassifier:
    return ScriptedClassifier()


@pytest.fixture
def container(settings, verifier, db, classifier) -> ServiceContainer:
    detection = DetectionService(
        AudioProcessor(settings.max_upload_bytes, settings.min_audio_seconds, settings.max_audio_seconds,
                       settings.silence_rms_threshold, ffmpeg_path=None),
        classifier,
        PredictionProcessor(settings.confidence_threshold, settings.detection_cooldown_seconds),
    )
    return ServiceContainer(settings=settings, token_verifier=verifier, database=db, detection=detection)


@pytest.fixture
def client(settings, container):
    with TestClient(create_app(settings, container)) as c:
        yield c


@pytest.fixture
def auth_headers(make_token):
    def _headers(sub: str = "auth0|user-a") -> dict[str, str]:
        return {"Authorization": f"Bearer {make_token(sub)}"}

    return _headers
