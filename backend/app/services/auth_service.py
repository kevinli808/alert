"""Auth0 access-token verification.

Tokens are verified locally against Auth0's public JWKS (cached), so no network call is made per
request. We check signature, algorithm (RS256 only), issuer, audience, expiry and required claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import jwt
from jwt import PyJWKClient

from app.core.errors import AuthError, ServiceNotConfiguredError

logger = logging.getLogger(__name__)

ALLOWED_ALGORITHMS = ["RS256"]


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str  # Auth0 "sub", e.g. "google-oauth2|1234" or "auth0|abcd"
    email: str | None = None
    name: str | None = None
    permissions: tuple[str, ...] = ()
    token: str = field(default="", repr=False, compare=False)


class SigningKeyProvider(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any: ...


class Auth0TokenVerifier:
    def __init__(
        self,
        domain: str,
        audience: str,
        claim_namespace: str = "",
        leeway_seconds: int = 30,
        jwks_client: SigningKeyProvider | None = None,
    ) -> None:
        self.domain = domain
        self.audience = audience
        self.issuer = f"https://{domain}/"
        self.claim_namespace = claim_namespace.rstrip("/")
        self.leeway = leeway_seconds
        self._jwks = jwks_client or PyJWKClient(
            f"https://{domain}/.well-known/jwks.json", cache_keys=True, lifespan=3600, timeout=5
        )

    def verify(self, token: str) -> AuthenticatedUser:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise AuthError("Malformed access token") from None
        if header.get("alg") not in ALLOWED_ALGORITHMS:
            raise AuthError("Unsupported token signing algorithm")

        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientConnectionError:
            logger.error("Could not reach Auth0 JWKS endpoint")
            raise ServiceNotConfiguredError("Authentication service temporarily unavailable") from None
        except jwt.PyJWTError:
            raise AuthError("Access token signed with an unknown key") from None

        try:
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=ALLOWED_ALGORITHMS,
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            raise AuthError("Access token has expired") from None
        except jwt.InvalidAudienceError:
            raise AuthError("Access token has an invalid audience") from None
        except jwt.InvalidIssuerError:
            raise AuthError("Access token has an invalid issuer") from None
        except jwt.PyJWTError:
            raise AuthError("Invalid access token") from None

        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub or len(sub) > 255:
            raise AuthError("Access token has an invalid subject")

        return AuthenticatedUser(
            user_id=sub,
            email=self._claim(claims, "email"),
            name=self._claim(claims, "name"),
            permissions=tuple(p for p in claims.get("permissions", []) if isinstance(p, str)),
            token=token,
        )

    def _claim(self, claims: dict[str, Any], name: str) -> str | None:
        # Auth0 access tokens only carry custom profile claims if an Action adds them (namespaced).
        value = claims.get(f"{self.claim_namespace}/{name}") if self.claim_namespace else None
        if value is None:
            value = claims.get(name)
        return value if isinstance(value, str) and value else None


async def fetch_userinfo(domain: str, access_token: str, timeout: float = 5.0) -> dict[str, Any]:
    """Fetch the user's profile from Auth0 /userinfo. Returns {} on any failure (best-effort).

    Works when the frontend requested the ``openid profile email`` scopes along with the API audience.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"https://{domain}/userinfo", headers={"Authorization": f"Bearer {access_token}"}
            )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, dict) else {}
        logger.info("Auth0 /userinfo returned HTTP %s", resp.status_code)
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("Auth0 /userinfo request failed: %s", type(exc).__name__)
    return {}
