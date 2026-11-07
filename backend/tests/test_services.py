"""SupabaseDatabase and DatabricksClassifier against mocked HTTP (no real services needed)."""

import asyncio
import base64
import json
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest

from app.core.errors import DatabaseError, InferenceError, ModelUnavailableError
from app.services.database_service import SupabaseDatabase
from app.services.databricks_service import DatabricksClassifier
from app.services.sound_classifier import FallbackClassifier
from tests.conftest import ScriptedClassifier


def _db_with(handler) -> SupabaseDatabase:
    db = SupabaseDatabase("https://proj.supabase.co", "sb_secret_abc")
    db._client = httpx.AsyncClient(base_url="https://proj.supabase.co/rest/v1/", transport=httpx.MockTransport(handler),
                                   headers=db._client.headers)
    return db


def test_supabase_queries_always_scope_to_user():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        if req.method == "POST":
            return httpx.Response(201, json=json.loads(req.content))
        return httpx.Response(200, json=[])

    db = _db_with(handler)

    async def run():
        await db.list_preferences("auth0|a")
        await db.upsert_preferences("auth0|a", [{"sound_type": "doorbell", "color": "#000000", "enabled": True, "flash_speed": 1}])
        await db.list_alert_history("auth0|a", 10, before=datetime(2026, 1, 1, tzinfo=timezone.utc))
        await db.delete_alert_history("auth0|a")

    asyncio.run(run())
    get_prefs, upsert, hist, delete = seen
    assert get_prefs.url.params["user_id"] == "eq.auth0|a"
    assert json.loads(upsert.content)[0]["user_id"] == "auth0|a"
    assert upsert.url.params["on_conflict"] == "user_id,sound_type"
    assert "merge-duplicates" in upsert.headers["prefer"]
    assert hist.url.params["detected_at"].startswith("lt.2026-01-01")
    assert delete.method == "DELETE" and delete.url.params["user_id"] == "eq.auth0|a"
    assert get_prefs.headers["apikey"] == "sb_secret_abc"
    assert "authorization" not in get_prefs.headers  # sb_secret_ keys are not JWTs


def test_supabase_errors_become_database_error():
    db = _db_with(lambda req: httpx.Response(500, json={"message": "boom"}))
    with pytest.raises(DatabaseError):
        asyncio.run(db.get_profile("auth0|a"))

    def down(req):
        raise httpx.ConnectError("refused")

    db = _db_with(down)
    with pytest.raises(DatabaseError):
        asyncio.run(db.get_profile("auth0|a"))


def _databricks_with(handler) -> DatabricksClassifier:
    clf = DatabricksClassifier("https://adb-1.azuredatabricks.net", "dapi-token", "yamnet-endpoint")
    clf._client = httpx.Client(transport=httpx.MockTransport(handler), headers=clf._client.headers)
    return clf


def test_databricks_request_and_response_contract():
    captured = {}

    def handler(req: httpx.Request):
        captured["req"] = req
        return httpx.Response(200, json={"predictions": [
            {"category_scores": {"fire_alarm": 0.91, "doorbell": 0.02}, "top_labels": [["Fire alarm", 0.91]]}]})

    clf = _databricks_with(handler)
    res = clf.classify(np.full(16_000, 0.5, dtype=np.float32))
    req = captured["req"]
    assert str(req.url) == "https://adb-1.azuredatabricks.net/serving-endpoints/yamnet-endpoint/invocations"
    assert req.headers["authorization"] == "Bearer dapi-token"
    record = json.loads(req.content)["dataframe_records"][0]
    pcm = np.frombuffer(base64.b64decode(record["audio_b64"]), dtype="<i2")
    assert len(pcm) == 16_000 and pcm[0] == 16383
    assert res.category_scores["fire_alarm"] == 0.91 and res.provider == "databricks"


@pytest.mark.parametrize("response, exc", [
    (httpx.Response(401), InferenceError),
    (httpx.Response(404), ModelUnavailableError),
    (httpx.Response(200, json={"unexpected": True}), InferenceError),
])
def test_databricks_errors(response, exc):
    with pytest.raises(exc):
        _databricks_with(lambda req: response).classify(np.zeros(16_000, dtype=np.float32))


def test_databricks_timeout():
    def slow(req):
        raise httpx.ReadTimeout("timeout")

    with pytest.raises(InferenceError, match="timed out"):
        _databricks_with(slow).classify(np.zeros(16_000, dtype=np.float32))


def test_unconfigured_databricks_and_fallback():
    remote = DatabricksClassifier("", "", "")
    assert remote.status == "not_configured"
    with pytest.raises(ModelUnavailableError):
        remote.classify(np.zeros(16_000, dtype=np.float32))

    failing = _databricks_with(lambda req: httpx.Response(500))
    local = ScriptedClassifier()
    local.set(doorbell=0.9)
    res = FallbackClassifier(failing, local).classify(np.zeros(16_000, dtype=np.float32))
    assert res.provider == "scripted" and local.calls == 1
