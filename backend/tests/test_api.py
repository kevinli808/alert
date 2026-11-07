from datetime import datetime, timedelta, timezone

from tests.conftest import wav_bytes


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["auth_configured"] and body["database_configured"]


def test_cors_preflight_allows_frontend_origin(client):
    resp = client.options(
        "/api/preferences",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "PUT",
                 "Access-Control-Request-Headers": "authorization,content-type"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    resp = client.options(
        "/api/preferences",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in resp.headers


def test_users_me_creates_profile_once(client, auth_headers, db):
    r1 = client.get("/api/users/me", headers=auth_headers("auth0|alice"))
    r2 = client.get("/api/users/me", headers=auth_headers("auth0|alice"))
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["id"] == "auth0|alice"
    assert r1.json()["email"] == "alice@example.com"
    assert len(db.profiles) == 1


def test_preferences_defaults_then_update(client, auth_headers):
    h = auth_headers()
    prefs = client.get("/api/preferences", headers=h).json()["preferences"]
    door = next(p for p in prefs if p["sound_type"] == "doorbell")
    assert door["is_default"] is True and door["updated_at"] is None

    resp = client.put(
        "/api/preferences",
        headers=h,
        json={"preferences": [{"sound_type": "doorbell", "color": "#00ff00", "enabled": False, "flash_speed": 2}]},
    )
    assert resp.status_code == 200
    door = next(p for p in resp.json()["preferences"] if p["sound_type"] == "doorbell")
    assert door == {**door, "color": "#00FF00", "enabled": False, "flash_speed": 2, "is_default": False}
    assert len(resp.json()["preferences"]) == len(prefs)  # still one entry per sound type


def test_preferences_are_isolated_per_user(client, auth_headers):
    client.put("/api/preferences", headers=auth_headers("auth0|alice"),
               json={"preferences": [{"sound_type": "doorbell", "color": "#123456"}]})
    bob = client.get("/api/preferences", headers=auth_headers("auth0|bob")).json()["preferences"]
    assert next(p for p in bob if p["sound_type"] == "doorbell")["is_default"] is True


def test_preferences_validation(client, auth_headers):
    h = auth_headers()
    bad_bodies = [
        {"preferences": [{"sound_type": "doorbell", "color": "blue"}]},
        {"preferences": [{"sound_type": "not_a_sound", "color": "#000000"}]},
        {"preferences": [{"sound_type": "doorbell", "color": "#000000", "flash_speed": 10}]},  # > 3 Hz
        {"preferences": [{"sound_type": "doorbell", "color": "#000000", "user_id": "auth0|victim"}]},
        {"preferences": [{"sound_type": "doorbell", "color": "#000000"}] * 2},
        {"preferences": []},
    ]
    for body in bad_bodies:
        resp = client.put("/api/preferences", headers=h, json=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


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

    assert len(db.history) == 1  # history written once per event, not per chunk
    assert db.history[0]["sound_type"] == "fire_alarm"

    hist = client.get("/api/alert-history", headers=h).json()
    assert [i["sound_type"] for i in hist["items"]] == ["fire_alarm"]


def test_detect_below_threshold_is_unknown(client, auth_headers, classifier, db):
    classifier.set(doorbell=0.4)
    body = client.post("/api/detect-sound", headers=auth_headers(),
                       files={"audio": ("c.wav", wav_bytes(), "audio/wav")}).json()
    assert body["detected"] is False and body["sound_type"] == "unknown" and body["reason"] == "below_threshold"
    assert body["confidence"] == 0.4
    assert db.history == []


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


def test_history_pagination_and_delete(client, auth_headers, db):
    h = auth_headers()
    client.get("/api/users/me", headers=h)
    base = datetime(2026, 11, 7, tzinfo=timezone.utc)
    for i in range(3):
        db.history.append({"id": f"id{i}", "user_id": "auth0|user-a", "sound_type": "doorbell",
                           "confidence": 0.9, "detected_at": base + timedelta(minutes=i)})
    db.history.append({"id": "other", "user_id": "auth0|other", "sound_type": "doorbell",
                       "confidence": 0.9, "detected_at": base})

    page1 = client.get("/api/alert-history?limit=2", headers=h).json()
    assert [i["id"] for i in page1["items"]] == ["id2", "id1"] and page1["next_before"]
    page2 = client.get("/api/alert-history", headers=h, params={"limit": 2, "before": page1["next_before"]}).json()
    assert [i["id"] for i in page2["items"]] == ["id0"] and page2["next_before"] is None

    assert client.delete("/api/alert-history", headers=h).status_code == 204
    assert [r["user_id"] for r in db.history] == ["auth0|other"]  # other users untouched


def test_rate_limit(settings, container, auth_headers):
    from fastapi.testclient import TestClient
    from app.main import create_app

    settings.rate_limit_default_per_minute = 2
    with TestClient(create_app(settings, container)) as c:
        h = auth_headers()
        assert c.get("/api/preferences", headers=h).status_code == 200
        assert c.get("/api/preferences", headers=h).status_code == 200
        resp = c.get("/api/preferences", headers=h)
        assert resp.status_code == 429 and int(resp.headers["retry-after"]) >= 1


def test_sound_types_public(client):
    types = client.get("/api/sound-types").json()["sound_types"]
    ids = {t["id"] for t in types}
    assert {"fire_alarm", "smoke_detector", "doorbell", "baby_crying", "glass_breaking", "emergency_siren"} <= ids


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/users/me", "/api/preferences", "/api/detect-sound", "/api/alert-history"]:
        assert p in paths
