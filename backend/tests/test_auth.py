import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from app.core.errors import AuthError


def test_valid_token(verifier, make_token):
    user = verifier.verify(make_token("google-oauth2|123"))
    assert user.user_id == "google-oauth2|123"
    assert user.email == "123@example.com"
    assert user.name == "Test User"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"exp": 1}, "expired"),
        ({"aud": "https://other-api"}, "audience"),
        ({"iss": "https://evil.example.com/"}, "issuer"),
        ({"sub": None}, "Invalid access token"),  # required claim missing
    ],
)
def test_rejects_bad_claims(verifier, make_token, overrides, message):
    with pytest.raises(AuthError, match=message):
        verifier.verify(make_token(**overrides))


def test_rejects_unknown_key(verifier, make_token):
    with pytest.raises(AuthError, match="unknown key"):
        verifier.verify(make_token(kid="other"))


def test_rejects_forged_signature(verifier, make_token):
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError):
        verifier.verify(make_token(key=attacker_key))  # right kid, wrong key


def test_rejects_hs256_algorithm_confusion(verifier):
    import jwt

    forged = jwt.encode({"sub": "auth0|x"}, "secret", algorithm="HS256", headers={"kid": "k1"})
    with pytest.raises(AuthError, match="algorithm"):
        verifier.verify(forged)


def test_rejects_garbage(verifier):
    with pytest.raises(AuthError, match="Malformed"):
        verifier.verify("not-a-jwt")


def test_protected_endpoint_requires_token(client):
    resp = client.get("/api/users/me")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    assert resp.headers["www-authenticate"] == "Bearer"


def test_protected_endpoint_rejects_expired_token(client, make_token):
    resp = client.get("/api/users/me", headers={"Authorization": f"Bearer {make_token(exp=1)}"})
    assert resp.status_code == 401
    assert "expired" in resp.json()["error"]["message"]
