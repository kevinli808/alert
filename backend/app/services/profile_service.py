"""Records the Auth0 user in the users table each time the frontend reports a login."""

from __future__ import annotations

from typing import Any

from app.services.auth_service import AuthenticatedUser, fetch_userinfo
from app.services.container import ServiceContainer
from app.services.database_service import Database


async def record_user_login(db: Database, user: AuthenticatedUser, services: ServiceContainer) -> dict[str, Any]:
    email, username = user.email, user.name
    # Access tokens only carry email/name when the Auth0 Action adds them. Otherwise ask /userinfo,
    # once per user per process (the table keeps values it already has).
    if (
        (email is None or username is None)
        and user.user_id not in services.known_profiles
        and services.settings.auth0_domain
        and user.token
    ):
        info = await fetch_userinfo(services.settings.auth0_domain, user.token)
        email = email or _str_or_none(info.get("email"), 320)
        username = username or _str_or_none(info.get("name") or info.get("nickname"), 100)
    row = await db.record_login(user.user_id, _truncate(email, 320), _truncate(username, 100))
    services.known_profiles.add(user.user_id)
    return row


def _str_or_none(value: Any, max_len: int) -> str | None:
    return _truncate(value, max_len) if isinstance(value, str) and value else None


def _truncate(value: str | None, max_len: int) -> str | None:
    return value[:max_len] if value else None
