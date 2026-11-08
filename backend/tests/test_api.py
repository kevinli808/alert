from datetime import datetime, timedelta, timezone

from tests.conftest import wav_bytes


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["auth_configured"] and body["database_configured"]


def test_cors_preflight_allows_frontend_origin(client):
    resp = client.options(
        "/api/users/me",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET",
                 "Access-Control-Request-Headers": "authorization,content-type"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    resp = client.options(
        "/api/users/me",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in resp.headers


def test_users_me_records_user_once_and_updates_last_login(client, auth_headers, db):
    r1 = client.get("/api/users/me", headers=auth_headers("auth0|alice"))
    r2 = client.get("/api/users/me", headers=auth_headers("auth0|alice"))
    assert r1.status_code == r2.status_code == 200
    first, second = r1.json(), r2.json()
    assert first["id"] == "auth0|alice" and first["email"] == "alice@example.com"
    assert first["username"] == "Test User"
    assert second["created_at"] == first["created_at"]
    assert second["last_login_at"] >= first["last_login_at"]
    assert list(db.users) == ["auth0|alice"]


def test_users_are_keyed_by_token_not_request(client, auth_headers, db):
    client.get("/api/users/me?id=auth0|mallory", headers=auth_headers("auth0|bob"))
    assert list(db.users) == ["auth0|bob"]


def test_users_me_requires_login(client):
    assert client.get("/api/users/me").status_code == 401


def test_users_me_503_without_database(settings, container, auth_headers):
    from fastapi.testclient import TestClient
    from app.main import create_app

    container.database = None
    with TestClient(create_app(settings, container)) as c:
        assert c.get("/health").json()["database_configured"] is False
        resp = c.get("/api/users/me", headers=auth_headers())
        assert resp.status_code == 503 and resp.json()["error"]["code"] == "service_not_configured"


def test_detect_sound_and_cooldown(client, auth_headers, classifier, db):
    h = auth_headers()
    classifier.set(fire_alarm=0.93)
    files = {"audio": ("chunk.wav", wav_bytes(1.0), "audio/wav")}

    first = client.post("/api/detect-sound", headers=h, files=files).json()
    assert first["detected"] is True and first["sound_type"] == "fire_alarm"
    assert first["confidence"] == 0.93 and first["is_new_event"] is True
    assert first["inference"]["total_ms"] >= 0

    second = client.post("/api/detect-sound", headers=h, files=files).json()
    assert second["detected"] is True and second["is_new_event"] is False
    assert second["event_id"] == first["event_id"]


def test_detect_below_threshold_is_unknown(client, auth_headers, classifier, db):
    classifier.set(doorbell=0.4)
    body = client.post("/api/detect-sound", headers=auth_headers(),
                       files={"audio": ("c.wav", wav_bytes(), "audio/wav")}).json()
    assert body["detected"] is False and body["sound_type"] == "unknown" and body["reason"] == "below_threshold"
    assert body["confidence"] == 0.4


def test_detect_silence_skips_model(client, auth_headers, classifier):
    body = client.post("/api/detect-sound", headers=auth_headers(),
                       files={"audio": ("c.wav", wav_bytes(amp=0.0), "audio/wav")}).json()
    assert body["reason"] == "silence" and body["sound_type"] is None
    assert classifier.calls == 0


def test_detect_model_failure_is_an_error_not_a_detection(client, auth_headers, classifier):
    from app.core.errors import ModelUnavailableError

    classifier.error = ModelUnavailableError("Sound model is still loading")
    resp = client.post("/api/detect-sound", headers=auth_headers(),
                       files={"audio": ("c.wav", wav_bytes(), "audio/wav")})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_unavailable"


def test_detect_rejects_bad_uploads(client, auth_headers, settings):
    h = auth_headers()
    corrupt = client.post("/api/detect-sound", headers=h, files={"audio": ("c.wav", b"RIFF....garbage", "audio/wav")})
    assert corrupt.status_code == 422 and corrupt.json()["error"]["code"] == "invalid_audio"

    wrong_type = client.post("/api/detect-sound", headers=h, files={"audio": ("c.txt", b"hello", "text/plain")})
    assert wrong_type.status_code == 415

    too_big = client.post("/api/detect-sound", headers=h,
                          files={"audio": ("c.wav", b"0" * (settings.max_upload_bytes + 1), "audio/wav")})
    assert too_big.status_code == 413

    missing = client.post("/api/detect-sound", headers=h)
    assert missing.status_code == 422


def test_rate_limit(settings, container, auth_headers):
    from fastapi.testclient import TestClient
    from app.main import create_app

    settings.rate_limit_default_per_minute = 2
    with TestClient(create_app(settings, container)) as c:
        h = auth_headers()
        assert c.get("/api/users/me", headers=h).status_code == 200
        assert c.get("/api/users/me", headers=h).status_code == 200
        resp = c.get("/api/users/me", headers=h)
        assert resp.status_code == 429 and int(resp.headers["retry-after"]) >= 1


def test_wake_model_requires_login_and_reports_status(client, auth_headers):
    assert client.post("/api/wake-model").status_code == 401
    resp = client.post("/api/wake-model", headers=auth_headers())
    assert resp.status_code == 200 and resp.json() == {"status": "ready"}  # local/scripted model


def test_wake_model_wakes_databricks(client, auth_headers, container, monkeypatch):
    from app.services.databricks_service import DatabricksClassifier

    remote = DatabricksClassifier("https://dbc-1.cloud.databricks.com", "dapi-x", "yamnet-sound-alerts")
    monkeypatch.setattr(remote, "wake", lambda: "waking")
    container.detection.classifier = remote
    assert client.post("/api/wake-model", headers=auth_headers()).json() == {"status": "waking"}


def test_sound_types_public(client):
    types = client.get("/api/sound-types").json()["sound_types"]
    ids = {t["id"] for t in types}
    assert {"fire_alarm", "smoke_detector", "doorbell", "baby_crying", "glass_breaking", "emergency_siren"} <= ids


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/users/me", "/api/detect-sound", "/api/sound-types", "/api/wake-model"]:
        assert p in paths
    assert "/api/settings" not in paths and "/api/alert-history" not in paths  # nothing else is stored


# ------------------------------------------------- local dev: anonymous detection ----------
def _anon_client(settings, container, **overrides):
    from fastapi.testclient import TestClient
    from app.main import create_app

    for key, value in overrides.items():
        setattr(settings, key, value)
    if not settings.auth0_configured:
        container.token_verifier = None
    return TestClient(create_app(settings, container))


def test_anonymous_detection_in_development_without_auth0(settings, container, classifier, db):
    classifier.set(doorbell=0.9)
    with _anon_client(settings, container, environment="development", auth0_domain="", auth0_audience="") as c:
        assert c.get("/health").json()["anonymous_detection"] is True
        resp = c.post("/api/detect-sound", files={"audio": ("c.wav", wav_bytes(), "audio/wav")})
        assert resp.status_code == 200 and resp.json()["sound_type"] == "doorbell"
        # Everything else still needs a real login.
        assert c.get("/api/users/me").status_code == 401


def test_anonymous_detection_refused_when_auth0_configured(settings, container):
    with _anon_client(settings, container, environment="development") as c:
        assert c.get("/health").json()["anonymous_detection"] is False
        resp = c.post("/api/detect-sound", files={"audio": ("c.wav", wav_bytes(), "audio/wav")})
        assert resp.status_code == 401


def test_anonymous_detection_refused_outside_development(settings, container):
    with _anon_client(settings, container, auth0_domain="", auth0_audience="") as c:  # environment="test"
        resp = c.post("/api/detect-sound", files={"audio": ("c.wav", wav_bytes(), "audio/wav")})
        assert resp.status_code == 401
