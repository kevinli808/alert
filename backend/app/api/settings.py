from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies.auth import get_current_user, rate_limit
from app.dependencies.services import get_database, get_services
from app.schemas.common import error_responses
from app.schemas.settings import SettingsResponse, SettingsUpdate, UserSettings
from app.services.auth_service import AuthenticatedUser
from app.services.container import ServiceContainer
from app.services.database_service import Database
from app.services.profile_service import ensure_profile_exists

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _to_response(row: dict | None) -> SettingsResponse:
    if row is None:
        return SettingsResponse(settings=None, updated_at=None)
    return SettingsResponse(settings=UserSettings.model_validate(row["settings"]), updated_at=row["updated_at"])


@router.get(
    "",
    response_model=SettingsResponse,
    response_model_by_alias=True,
    summary="Get the user's saved alert settings (null if never saved)",
    responses=error_responses(401, 429, 503),
    dependencies=[Depends(rate_limit("default"))],
)
async def get_settings(
    user: AuthenticatedUser = Depends(get_current_user),
    db: Database = Depends(get_database),
) -> SettingsResponse:
    return _to_response(await db.get_settings(user.user_id))


@router.put(
    "",
    response_model=SettingsResponse,
    response_model_by_alias=True,
    summary="Replace the user's alert settings",
    description="Body is `{\"settings\": {...}}` in the frontend's model: simple, rules, customStates, "
    "stateOrder, introComplete. Unknown fields are rejected.",
    responses=error_responses(401, 422, 429, 503),
    dependencies=[Depends(rate_limit("default"))],
)
async def put_settings(
    body: SettingsUpdate,
    user: AuthenticatedUser = Depends(get_current_user),
    db: Database = Depends(get_database),
    services: ServiceContainer = Depends(get_services),
) -> SettingsResponse:
    await ensure_profile_exists(db, user, services)
    stored = body.settings.model_dump(mode="json", by_alias=True, exclude_none=True)
    return _to_response(await db.upsert_settings(user.user_id, stored))
