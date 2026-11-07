"""Lazily provisions a `profiles` row the first time an Auth0 user calls the API."""

from __future__ import annotations

from typing import Any

from app.services.auth_service import AuthenticatedUser, fetch_userinfo
from app.services.container import ServiceContainer
from app.services.database_service import Database


async def get_or_create_profile(
    db: Database, user: AuthenticatedUser, services: ServiceContainer
) -> dict[str, Any]:
    profile = await db.get_profile(user.user_id)
    if profile is None:
        email, username = user.email, user.name
        if (email is None or username is None) and services.settings.auth0_domain and user.token:
            info = await fetch_userinfo(services.settings.auth0_domain, user.token)
            email = email or _str_or_none(info.get("email"), 320)
            username = username or _str_or_none(info.get("name") or info.get("nickname"), 100)
        profile = await db.create_profile_if_missing(user.user_id, email, _truncate(username, 100))
    services.known_profiles.add(user.user_id)
    return profile


async def ensure_profile_exists(db: Database, user: AuthenticatedUser, services: ServiceContainer) -> None:
    """Cheap check (cached per process) used before writing rows that reference profiles.id."""
    if user.user_id not in services.known_profiles:
        await get_or_create_profile(db, user, services)


def _str_or_none(value: Any, max_len: int) -> str | None:
    return _truncate(value, max_len) if isinstance(value, str) and value else None


def _truncate(value: str | None, max_len: int) -> str | None:
    return value[:max_len] if value else None
