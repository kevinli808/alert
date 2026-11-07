"""Data access for Supabase PostgreSQL via its PostgREST API.

The backend uses the service-role/secret key, which bypasses RLS. Authorization is therefore
enforced HERE: every method takes the authenticated Auth0 user id and filters on it. Never expose
a method that reads or writes rows without a user_id filter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.errors import DatabaseError

logger = logging.getLogger(__name__)

PREFERENCE_COLUMNS = "sound_type,color,enabled,flash_speed,updated_at"
HISTORY_COLUMNS = "id,sound_type,confidence,detected_at"
PROFILE_COLUMNS = "id,email,username,created_at"


class Database(ABC):
    """Storage interface. Implemented by SupabaseDatabase (and an in-memory fake in tests)."""

    @abstractmethod
    async def get_profile(self, user_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def create_profile_if_missing(
        self, user_id: str, email: str | None, username: str | None
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def list_preferences(self, user_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def upsert_preferences(self, user_id: str, preferences: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def insert_alert_history(
        self, user_id: str, sound_type: str, confidence: float, detected_at: datetime
    ) -> None: ...

    @abstractmethod
    async def list_alert_history(
        self, user_id: str, limit: int, before: datetime | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def delete_alert_history(self, user_id: str) -> None: ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


class SupabaseDatabase(Database):
    def __init__(self, url: str, service_key: str, timeout: float = 5.0) -> None:
        headers = {"apikey": service_key, "Accept": "application/json"}
        # Legacy service_role keys are JWTs and go in Authorization as well; new sb_secret_ keys do not.
        if service_key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {service_key}"
        self._client = httpx.AsyncClient(
            base_url=f"{url.rstrip('/')}/rest/v1/", headers=headers, timeout=timeout
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        json: Any = None,
        prefer: str | None = None,
    ) -> Any:
        headers = {"Prefer": prefer} if prefer else None
        try:
            resp = await self._client.request(method, table, params=params, json=json, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("Supabase %s %s failed: %s", method, table, type(exc).__name__)
            raise DatabaseError("Database is temporarily unavailable") from exc
        if resp.status_code >= 400:
            # PostgREST error bodies contain no secrets; truncate to keep logs tidy.
            logger.error("Supabase %s %s -> HTTP %s: %s", method, table, resp.status_code, resp.text[:300])
            raise DatabaseError("Database request failed")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # --- profiles ----------------------------------------------------------------------
    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        rows = await self._request(
            "GET", "profiles", params={"select": PROFILE_COLUMNS, "id": f"eq.{user_id}", "limit": "1"}
        )
        return rows[0] if rows else None

    async def create_profile_if_missing(
        self, user_id: str, email: str | None, username: str | None
    ) -> dict[str, Any]:
        rows = await self._request(
            "POST",
            "profiles",
            params={"on_conflict": "id", "select": PROFILE_COLUMNS},
            json={"id": user_id, "email": email, "username": username},
            prefer="resolution=ignore-duplicates,return=representation",
        )
        if rows:
            return rows[0]
        existing = await self.get_profile(user_id)  # row already existed
        if existing is None:
            raise DatabaseError("Could not create user profile")
        return existing

    # --- alert_preferences ---------------------------------------------------------------
    async def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        return await self._request(
            "GET",
            "alert_preferences",
            params={"select": PREFERENCE_COLUMNS, "user_id": f"eq.{user_id}", "order": "sound_type.asc"},
        ) or []

    async def upsert_preferences(self, user_id: str, preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        payload = [
            {
                "user_id": user_id,  # always the authenticated user — never taken from the request body
                "sound_type": p["sound_type"],
                "color": p["color"],
                "enabled": p["enabled"],
                "flash_speed": p["flash_speed"],
                "updated_at": now,
            }
            for p in preferences
        ]
        return await self._request(
            "POST",
            "alert_preferences",
            params={"on_conflict": "user_id,sound_type", "select": PREFERENCE_COLUMNS},
            json=payload,
            prefer="resolution=merge-duplicates,return=representation",
        ) or []

    # --- alert_history -------------------------------------------------------------------
    async def insert_alert_history(
        self, user_id: str, sound_type: str, confidence: float, detected_at: datetime
    ) -> None:
        await self._request(
            "POST",
            "alert_history",
            json={
                "user_id": user_id,
                "sound_type": sound_type,
                "confidence": round(float(confidence), 4),
                "detected_at": detected_at.isoformat(),
            },
            prefer="return=minimal",
        )

    async def list_alert_history(
        self, user_id: str, limit: int, before: datetime | None = None
    ) -> list[dict[str, Any]]:
        params = {
            "select": HISTORY_COLUMNS,
            "user_id": f"eq.{user_id}",
            "order": "detected_at.desc",
            "limit": str(limit),
        }
        if before is not None:
            params["detected_at"] = f"lt.{before.isoformat()}"
        return await self._request("GET", "alert_history", params=params) or []

    async def delete_alert_history(self, user_id: str) -> None:
        await self._request(
            "DELETE", "alert_history", params={"user_id": f"eq.{user_id}"}, prefer="return=minimal"
        )
