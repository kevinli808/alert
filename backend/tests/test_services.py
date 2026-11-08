"""DatabricksDatabase and DatabricksClassifier against mocked HTTP (no real services needed)."""

import asyncio
import base64
import json

import httpx
import numpy as np
import pytest

from app.core.errors import DatabaseError, InferenceError, ModelUnavailableError
from app.services.database_service import DatabricksDatabase, quote_table_name
from app.services.databricks_service import DatabricksClassifier
from app.services.sound_classifier import FallbackClassifier
from tests.conftest import ScriptedClassifier

USER_COLUMNS = ["id", "email", "username", "created_at", "last_login_at"]


def _sql_db_with(handler, **kw) -> DatabricksDatabase:
    db = DatabricksDatabase("https://dbc-1.cloud.databricks.com/", "dapi-secret", "wh123", "main.alert.users", **kw)
    db._client = httpx.AsyncClient(base_url="https://dbc-1.cloud.databricks.com",
                                   transport=httpx.MockTransport(handler), headers=db._client.headers)
    return db


def _succeeded(rows=None, columns=USER_COLUMNS):
    return {"statement_id": "s1", "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": c} for c in columns]}},
            "result": {"data_array": rows or []}}


def test_record_login_sends_parameterized_sql_to_warehouse():
    sent = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        sent.append((req, body))
        if body["statement"].startswith("SELECT"):
            return httpx.Response(200, json=_succeeded(
                [["auth0|a", None, "Ann", "2026-11-07T18:00:00.000Z", "2026-11-07T18:05:00.000Z"]]))
        return httpx.Response(200, json=_succeeded(columns=[]))

    db = _sql_db_with(handler)
    row = asyncio.run(db.record_login("auth0|a", None, "Ann"))
    assert row == {"id": "auth0|a", "email": None, "username": "Ann",
                   "created_at": "2026-11-07T18:00:00.000Z", "last_login_at": "2026-11-07T18:05:00.000Z"}

    create, merge, select = sent
    req, _ = create
    assert req.url.path == "/api/2.0/sql/statements/" and req.headers["authorization"] == "Bearer dapi-secret"
    assert "CREATE TABLE IF NOT EXISTS `main`.`alert`.`users`" in create[1]["statement"]
    body = merge[1]
    assert body["warehouse_id"] == "wh123" and body["statement"].startswith("MERGE INTO `main`.`alert`.`users`")
    assert "auth0|a" not in body["statement"]  # values are bound, never formatted into SQL
    params = {p["name"]: p for p in body["parameters"]}
    assert params["id"]["value"] == "auth0|a" and params["username"]["value"] == "Ann"
    assert "value" not in params["email"]  # omitted value = SQL NULL
    assert select[1]["parameters"] == [{"name": "id", "type": "STRING", "value": "auth0|a"}]

    # The table is only created once per process.
    sent.clear()
    asyncio.run(db.get_user("auth0|a"))
    assert len(sent) == 1 and sent[0][1]["statement"].startswith("SELECT")


def test_profile_settings_use_separate_table_and_parameterized_user_id():
    settings_json = '{"simple":true}'
    sent = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        sent.append(body)
        if body["statement"].startswith("SELECT"):
            return httpx.Response(200, json=_succeeded([[settings_json]], columns=["settings_json"]))
        return httpx.Response(200, json=_succeeded(columns=[]))

    db = _sql_db_with(handler)
    assert asyncio.run(db.get_settings("auth0|alice")) == {"simple": True}
    assert "CREATE TABLE IF NOT EXISTS `main`.`alert`.`user_settings`" in sent[0]["statement"]
    assert sent[1]["statement"].startswith("SELECT settings_json FROM `main`.`alert`.`user_settings`")
    assert sent[1]["parameters"] == [{"name": "user_id", "type": "STRING", "value": "auth0|alice"}]

    sent.clear()
    asyncio.run(db.save_settings("auth0|alice", {"simple": False}))
    merge = sent[0]
    assert merge["statement"].startswith("MERGE INTO `main`.`alert`.`user_settings`")
    assert "auth0|alice" not in merge["statement"] and '{"simple":false}' not in merge["statement"]
    params = {item["name"]: item for item in merge["parameters"]}
    assert params["user_id"]["value"] == "auth0|alice"
    assert params["settings_json"]["value"] == '{"simple":false}'


def test_statement_polls_while_warehouse_starts():
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.method)
        if req.method == "POST":
            return httpx.Response(200, json={"statement_id": "s9", "status": {"state": "PENDING"}})
        if len(calls) < 3:
            return httpx.Response(200, json={"statement_id": "s9", "status": {"state": "RUNNING"}})
        return httpx.Response(200, json=_succeeded([]))

    db = _sql_db_with(handler, poll_interval=0)
    db._table_ready = True
    assert asyncio.run(db.get_user("auth0|nobody")) is None
    assert calls == ["POST", "GET", "GET"]


def test_statement_timeout_cancels_and_raises():
    paths = []

    def handler(req: httpx.Request):
        paths.append(req.url.path)
        return httpx.Response(200, json={"statement_id": "s9", "status": {"state": "PENDING"}})

    db = _sql_db_with(handler, timeout=0, poll_interval=0)
    db._table_ready = True
    with pytest.raises(DatabaseError):
        asyncio.run(db.get_user("auth0|a"))
    assert paths[-1] == "/api/2.0/sql/statements/s9/cancel"


@pytest.mark.parametrize("response", [
    httpx.Response(403, json={"message": "invalid token"}),
    httpx.Response(200, json={"statement_id": "s1", "status": {"state": "FAILED", "error": {"message": "TABLE_OR_VIEW_NOT_FOUND"}}}),
    httpx.Response(200, text="not json"),
])
def test_sql_errors_become_database_error(response):
    db = _sql_db_with(lambda req: response)
    with pytest.raises(DatabaseError):
        asyncio.run(db.get_user("auth0|a"))


def test_sql_connection_failure_becomes_database_error():
    def down(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(DatabaseError):
        asyncio.run(_sql_db_with(down).get_user("auth0|a"))


@pytest.mark.parametrize("name", ["users", "alert.users", "main.alert.users", "Main_1.a_b.c9"])
def test_table_name_accepts_identifiers(name):
    assert quote_table_name(name).count("`") == 2 * len(name.split("."))


@pytest.mark.parametrize("name", ["", "a.b.c.d", "main.alert.users; DROP TABLE x", "main.`x`", "1abc", "a..b"])
def test_table_name_rejects_injection(name):
    with pytest.raises(ValueError):
        quote_table_name(name)


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
    (httpx.Response(503), ModelUnavailableError),
    (httpx.Response(504), ModelUnavailableError),
    (httpx.Response(500), InferenceError),
    (httpx.Response(200, json={"unexpected": True}), InferenceError),
])
def test_databricks_errors(response, exc):
    with pytest.raises(exc):
        _databricks_with(lambda req: response).classify(np.zeros(16_000, dtype=np.float32))


def test_databricks_timeout_means_waking_up_not_failure():
    # A scaled-to-zero endpoint holds the first requests for ~1 min while a replica starts.
    def slow(req):
        raise httpx.ReadTimeout("timeout")

    with pytest.raises(ModelUnavailableError, match="waking up"):
        _databricks_with(slow).classify(np.zeros(16_000, dtype=np.float32))


def test_databricks_wake():
    seen = []

    def ok(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"predictions": [{"category_scores": {}, "top_labels": []}]})

    def slow(req):
        raise httpx.ReadTimeout("timeout")

    assert _databricks_with(ok).wake() == "ready"
    assert seen[0]["dataframe_records"][0]["sample_rate"] == 16000  # same contract as classify
    assert _databricks_with(slow).wake() == "waking"
    assert _databricks_with(lambda req: httpx.Response(503)).wake() == "waking"
    assert DatabricksClassifier("", "", "").wake() == "not_configured"


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
