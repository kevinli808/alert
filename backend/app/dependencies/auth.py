"""Reusable FastAPI authentication dependencies.

Usage::

    @router.get("/thing")
    def handler(user: AuthenticatedUser = Depends(get_current_user)): ...
"""

from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.errors import AuthError, ServiceNotConfiguredError
from app.dependencies.services import get_services
from app.services.auth_service import AuthenticatedUser
from app.services.container import ServiceContainer

# auto_error=False so we can return our own consistent 401 error envelope.
bearer_scheme = HTTPBearer(auto_error=False, description="Auth0 access token")


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    services: ServiceContainer = Depends(get_services),
) -> AuthenticatedUser:
    # Sync dependency: FastAPI runs it in a worker thread, so a JWKS cache refresh never blocks the loop.
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise AuthError("Missing bearer token")
    if services.token_verifier is None:
        raise ServiceNotConfiguredError("Authentication is not configured on the server")
    return services.token_verifier.verify(credentials.credentials)


ANONYMOUS_USER_ID = "dev|anonymous"
ANONYMOUS_DEV_USER = AuthenticatedUser(user_id=ANONYMOUS_USER_ID)


def get_detection_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    services: ServiceContainer = Depends(get_services),
) -> AuthenticatedUser:
    """Like get_current_user, but allows token-less local development (see allow_anonymous_detection)."""
    # In this mode Auth0 isn't configured, so a token (e.g. from a logged-in frontend) can't be
    # verified anyway; treat every caller as the anonymous dev user rather than failing.
    if services.settings.anonymous_detection_enabled:
        return ANONYMOUS_DEV_USER
    return get_current_user(credentials, services)


def rate_limit(bucket: str, *, detection: bool = False):  # type: ignore[no-untyped-def]
    """Build a dependency that rate-limits the authenticated user for a given bucket."""
    user_dependency = get_detection_user if detection else get_current_user

    def _dependency(
        user: AuthenticatedUser = Depends(user_dependency),
        services: ServiceContainer = Depends(get_services),
    ) -> None:
        settings = services.settings
        limit = settings.rate_limit_detect_per_minute if detection else settings.rate_limit_default_per_minute
        services.rate_limiter.hit(f"{bucket}:{user.user_id}", limit, 60.0)

    return _dependency
