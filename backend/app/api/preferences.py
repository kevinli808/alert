from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.dependencies.auth import get_current_user, rate_limit
from app.dependencies.services import get_database, get_services
from app.schemas.common import error_responses
from app.schemas.preferences import AlertPreference, PreferencesResponse, PreferencesUpdate
from app.services.auth_service import AuthenticatedUser
from app.services.container import ServiceContainer
from app.services.database_service import Database
from app.services.profile_service import ensure_profile_exists
from app.services.sound_categories import CATEGORY_IDS, SOUND_CATEGORIES

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


def _merge_with_defaults(rows: list[dict[str, Any]]) -> PreferencesResponse:
    """Return one preference per supported sound type: the user's saved row, or the default."""
    saved = {r["sound_type"]: r for r in rows if r.get("sound_type") in CATEGORY_IDS}
    items = []
    for cat in SOUND_CATEGORIES:
        row = saved.get(cat.id)
        if row is not None:
            items.append(AlertPreference.model_validate({**row, "is_default": False}))
        else:
            items.append(
                AlertPreference(
                    sound_type=cat.id,
                    color=cat.default_color,
                    enabled=True,
                    flash_speed=cat.default_flash_speed,
                    updated_at=None,
                    is_default=True,
                )
            )
    return PreferencesResponse(preferences=items)


@router.get(
    "",
    response_model=PreferencesResponse,
    summary="Get alert preferences for every supported sound type",
    responses=error_responses(401, 429, 503),
    dependencies=[Depends(rate_limit("default"))],
)
async def get_preferences(
    user: AuthenticatedUser = Depends(get_current_user),
    db: Database = Depends(get_database),
) -> PreferencesResponse:
    return _merge_with_defaults(await db.list_preferences(user.user_id))


@router.put(
    "",
    response_model=PreferencesResponse,
    summary="Create or replace preferences for the listed sound types",
    description="Only the sound types included are changed; others keep their current values. "
    "Returns the full, merged preference list.",
    responses=error_responses(401, 422, 429, 503),
    dependencies=[Depends(rate_limit("default"))],
)
async def put_preferences(
    body: PreferencesUpdate,
    user: AuthenticatedUser = Depends(get_current_user),
    db: Database = Depends(get_database),
    services: ServiceContainer = Depends(get_services),
) -> PreferencesResponse:
    await ensure_profile_exists(db, user, services)
    await db.upsert_preferences(user.user_id, [p.model_dump() for p in body.preferences])
    return _merge_with_defaults(await db.list_preferences(user.user_id))
