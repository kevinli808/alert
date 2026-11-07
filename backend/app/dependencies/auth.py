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


def rate_limit(bucket: str, *, detection: bool = False):  # type: ignore[no-untyped-def]
    """Build a dependency that rate-limits the authenticated user for a given bucket."""

    def _dependency(
        user: AuthenticatedUser = Depends(get_current_user),
        services: ServiceContainer = Depends(get_services),
    ) -> None:
        settings = services.settings
        limit = settings.rate_limit_detect_per_minute if detection else settings.rate_limit_default_per_minute
        services.rate_limiter.hit(f"{bucket}:{user.user_id}", limit, 60.0)

    return _dependency
