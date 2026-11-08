from datetime import datetime, timedelta, timezone

from tests.conftest import wav_bytes


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["auth_configured"] and body["database_configured"]


def test_cors_preflight_allows_frontend_origin(client):
    resp = client.options(
        "/api/settings",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "PUT",
                 "Access-Control-Request-Headers": "authorization,content-type"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    resp = client.options(
        "/api/settings",
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


FRONTEND_SETTINGS = {  # shape of StoredSettings in src/components/AlertApp.tsx
    "simple": False,
    "rules": [
        {"id": "doorbell", "label": "Doorbell", "kind": "sound", "enabled": True, "effects": ["warning", "vibration"]},
        {"id": "phrase-1", "label": "Watch out", "kind": "phrase", "enabled": True, "effects": ["custom-1", "tts"],
         "ttsResponse": "On my way", "lightColor": "#2DD4BF"},
    ],
    "customStates": [{"id": "custom-1", "name": "Information", "color": "#4F7FD8", "priority": 1, "duration": 4000}],
    "stateOrder": ["urgent", "custom-1", "warning", "default"],
    "introComplete": True,
}


def test_settings_empty_then_saved_round_trip(client, auth_headers):
    h = auth_headers()
    first = client.get("/api/settings", headers=h).json()
    assert first == {"settings": None, "updatedAt": None}

    resp = client.put("/api/settings", headers=h, json={"settings": FRONTEND_SETTINGS})
    assert resp.status_code == 200
    assert resp.json()["settings"] == FRONTEND_SETTINGS  # stored and returned in the frontend's own shape
    again = client.get("/api/settings", headers=h).json()
    assert again["settings"] == FRONTEND_SETTINGS and again["updatedAt"]


def test_settings_are_isolated_per_user(client, auth_headers, db):
    client.put("/api/settings", headers=auth_headers("auth0|alice"), json={"settings": FRONTEND_SETTINGS})
    assert client.get("/api/settings", headers=auth_headers("auth0|bob")).json()["settings"] is None
    assert set(db.settings) == {"auth0|alice"}


def test_settings_validation(client, auth_headers):
    import copy

    def variant(mutate):
        body = copy.deepcopy(FRONTEND_SETTINGS)
        mutate(body)
        return {"settings": body}

    bad_bodies = [
        variant(lambda b: b.update(userId="auth0|victim")),  # unknown fields are rejected
        variant(lambda b: b["rules"][0].update(kind="music")),
        variant(lambda b: b["rules"].append(dict(b["rules"][0]))),  # duplicate rule id
        variant(lambda b: b["rules"][0]["effects"].append("<script>")),
        variant(lambda b: b["customStates"][0].update(color="blue")),
        variant(lambda b: b["customStates"][0].update(id="urgent")),  # can't overwrite a system state
        variant(lambda b: b["customStates"][0].update(duration=10**9)),
        variant(lambda b: b["stateOrder"].append("ghost")),  # unknown state in the priority list
        {"settings": {"rules": "nope"}},
        {},
    ]
    h = auth_headers()
    for body in bad_bodies:
        resp = client.put("/api/settings", headers=h, json=body)
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
        assert c.get("/api/settings", headers=h).status_code == 200
        assert c.get("/api/settings", headers=h).status_code == 200
        resp = c.get("/api/settings", headers=h)
        assert resp.status_code == 429 and int(resp.headers["retry-after"]) >= 1


def test_sound_types_public(client):
    types = client.get("/api/sound-types").json()["sound_types"]
    ids = {t["id"] for t in types}
    assert {"fire_alarm", "smoke_detector", "doorbell", "baby_crying", "glass_breaking", "emergency_siren"} <= ids


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/users/me", "/api/settings", "/api/detect-sound", "/api/alert-history"]:
        assert p in paths


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
        assert db.history == []  # anonymous detections are never stored
        # Everything else still needs a real login.
        assert c.get("/api/settings").status_code == 401


def test_anonymous_detection_refused_when_auth0_configured(settings, container):
    with _anon_client(settings, container, environment="development") as c:
        assert c.get("/health").json()["anonymous_detection"] is False
        resp = c.post("/api/detect-sound", files={"audio": ("c.wav", wav_bytes(), "audio/wav")})
        assert resp.status_code == 401


def test_anonymous_detection_refused_outside_development(settings, container):
    with _anon_client(settings, container, auth0_domain="", auth0_audience="") as c:  # environment="test"
        resp = c.post("/api/detect-sound", files={"audio": ("c.wav", wav_bytes(), "audio/wav")})
        assert resp.status_code == 401
