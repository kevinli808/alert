"""User records stored in a Databricks SQL table, via the SQL Statement Execution API.

The database records one thing: who has logged in. Each row is keyed by the Auth0 `sub` of the
verified access token. The user id is never taken from the request body.

Every value is sent as a named statement parameter (never string-formatted into SQL). The table
name is the only interpolated piece, and it is validated as a plain identifier at startup.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.core.errors import DatabaseError

logger = logging.getLogger(__name__)

USER_COLUMNS = "id, email, username, created_at, last_login_at"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUNNING_STATES = {"PENDING", "RUNNING"}


def quote_table_name(name: str) -> str:
    """`catalog.schema.table` -> `catalog`.`schema`.`table`, rejecting anything that isn't an identifier."""
    parts = name.strip().split(".")
    if not 1 <= len(parts) <= 3 or not all(_IDENTIFIER.match(p) for p in parts):
        raise ValueError(f"DATABRICKS_USERS_TABLE must look like catalog.schema.table, got {name!r}")
    return ".".join(f"`{p}`" for p in parts)


class Database(ABC):
    """Storage interface. Implemented by DatabricksDatabase (and an in-memory fake in tests)."""

    @abstractmethod
    async def get_user(self, user_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def record_login(self, user_id: str, email: str | None, username: str | None) -> dict[str, Any]:
        """Create the user on first login; afterwards refresh last_login_at (and fill in missing email/name)."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


class DatabricksDatabase(Database):
    def __init__(
        self,
        host: str,
        token: str,
        warehouse_id: str,
        table: str,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> None:
        self._warehouse_id = warehouse_id
        self._table = quote_table_name(table)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._table_ready = False
        self._client = httpx.AsyncClient(
            base_url=host.rstrip("/"),
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=httpx.Timeout(timeout + 10),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- low-level -----------------------------------------------------------------------
    async def _call(self, method: str, path: str, json: Any = None) -> dict[str, Any]:
        try:
            resp = await self._client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            logger.error("Databricks SQL %s %s failed: %s", method, path, type(exc).__name__)
            raise DatabaseError("Database is temporarily unavailable") from exc
        if resp.status_code >= 400:
            logger.error("Databricks SQL %s -> HTTP %s: %s", path, resp.status_code, resp.text[:300])
            raise DatabaseError("Database request failed")
        try:
            return resp.json()
        except ValueError as exc:
            raise DatabaseError("Database returned an invalid response") from exc

    async def _execute(self, statement: str, params: dict[str, str | None] | None = None) -> list[dict[str, Any]]:
        """Run one statement and return its rows as dicts. Waits for a cold warehouse up to `timeout`."""
        body = {
            "warehouse_id": self._warehouse_id,
            "statement": statement,
            # An omitted "value" is bound as SQL NULL.
            "parameters": [
                {"name": k, "type": "STRING", **({"value": v} if v is not None else {})}
                for k, v in (params or {}).items()
            ],
            "wait_timeout": f"{int(min(50, max(5, self._timeout)))}s",
            "on_wait_timeout": "CONTINUE",
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
        }
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        data = await self._call("POST", "/api/2.0/sql/statements/", json=body)
        statement_id = data.get("statement_id")
        while (data.get("status") or {}).get("state") in _RUNNING_STATES:
            if loop.time() >= deadline or not statement_id:
                if statement_id:
                    try:
                        await self._call("POST", f"/api/2.0/sql/statements/{statement_id}/cancel")
                    except DatabaseError:
                        pass
                raise DatabaseError("Database timed out (is the SQL warehouse starting up?)")
            await asyncio.sleep(self._poll_interval)
            data = await self._call("GET", f"/api/2.0/sql/statements/{statement_id}")

        status = data.get("status") or {}
        if status.get("state") != "SUCCEEDED":
            err = status.get("error") or {}
            logger.error("Databricks SQL statement %s: %s", status.get("state"), str(err.get("message", ""))[:300])
            raise DatabaseError("Database request failed")

        columns = [c["name"] for c in ((data.get("manifest") or {}).get("schema") or {}).get("columns", [])]
        rows = (data.get("result") or {}).get("data_array") or []
        return [dict(zip(columns, row)) for row in rows]

    async def _ensure_table(self) -> None:
        if self._table_ready:
            return
        await self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            "id STRING NOT NULL COMMENT 'Auth0 user id (the access token sub claim)', "
            "email STRING, "
            "username STRING, "
            "created_at TIMESTAMP NOT NULL, "
            "last_login_at TIMESTAMP NOT NULL"
            ") COMMENT 'Alert app users, one row per Auth0 account'"
        )
        self._table_ready = True

    # --- users -----------------------------------------------------------------------------
    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        await self._ensure_table()
        rows = await self._execute(
            f"SELECT {USER_COLUMNS} FROM {self._table} WHERE id = :id LIMIT 1", {"id": user_id}
        )
        return rows[0] if rows else None

    async def record_login(self, user_id: str, email: str | None, username: str | None) -> dict[str, Any]:
        await self._ensure_table()
        await self._execute(
            f"MERGE INTO {self._table} AS u "
            "USING (SELECT :id AS id, :email AS email, :username AS username) AS s "
            "ON u.id = s.id "
            "WHEN MATCHED THEN UPDATE SET "
            "u.email = coalesce(s.email, u.email), "
            "u.username = coalesce(s.username, u.username), "
            "u.last_login_at = current_timestamp() "
            "WHEN NOT MATCHED THEN INSERT (id, email, username, created_at, last_login_at) "
            "VALUES (s.id, s.email, s.username, current_timestamp(), current_timestamp())",
            {"id": user_id, "email": email, "username": username},
        )
        user = await self.get_user(user_id)
        if user is None:
            raise DatabaseError("Could not record user")
        return user
