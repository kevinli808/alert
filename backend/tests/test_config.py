import pytest

from app.core.config import Settings


def test_vercel_defaults_to_production_databricks_without_local_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ENVIRONMENT",
        "INFERENCE_PROVIDER",
        "PRELOAD_MODEL",
        "DATABRICKS_FALLBACK_TO_LOCAL",
        "AUTH0_DOMAIN",
        "AUTH0_AUDIENCE",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_ENDPOINT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.inference_provider == "databricks"
    assert settings.preload_model is False
    assert settings.databricks_fallback_to_local is False
    with pytest.raises(ValueError, match="Missing required production configuration"):
        settings.validate_for_startup()


def test_vercel_rejects_local_inference_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings(_env_file=None, inference_provider="local")

    with pytest.raises(ValueError, match="Vercel deployments require INFERENCE_PROVIDER=databricks"):
        settings.validate_for_startup()


def test_vercel_rejects_local_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings(_env_file=None, databricks_fallback_to_local=True)

    with pytest.raises(ValueError, match="Vercel deployments require DATABRICKS_FALLBACK_TO_LOCAL=false"):
        settings.validate_for_startup()


def test_auth0_audience_removes_accidental_trailing_slash() -> None:
    settings = Settings(_env_file=None, auth0_audience="https://api.deaf-accessibility.app/")

    assert settings.auth0_audience == "https://api.deaf-accessibility.app"
