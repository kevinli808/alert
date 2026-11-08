from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies.auth import get_current_user, rate_limit
from app.dependencies.services import get_database, get_services
from app.schemas.common import error_responses
from app.schemas.user import UserProfile
from app.services.auth_service import AuthenticatedUser
from app.services.container import ServiceContainer
from app.services.database_service import Database
from app.services.profile_service import record_user_login

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserProfile,
    summary="Record the logged-in user (created on first call) and return their row",
    description="The frontend calls this once after each Auth0 login. "
    "It creates the user on first login and updates `last_login_at` afterwards.",
    responses=error_responses(401, 429, 503),
    dependencies=[Depends(rate_limit("default"))],
)
async def read_me(
    user: AuthenticatedUser = Depends(get_current_user),
    db: Database = Depends(get_database),
    services: ServiceContainer = Depends(get_services),
) -> UserProfile:
    row = await record_user_login(db, user, services)
    return UserProfile.model_validate(row)
