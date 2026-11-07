from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response

from app.dependencies.auth import get_current_user, rate_limit
from app.dependencies.services import get_database
from app.schemas.common import error_responses
from app.schemas.detection import AlertHistoryItem, AlertHistoryResponse
from app.services.auth_service import AuthenticatedUser
from app.services.database_service import Database

router = APIRouter(prefix="/api/alert-history", tags=["history"])


@router.get(
    "",
    response_model=AlertHistoryResponse,
    summary="List the authenticated user's detected sound events, newest first",
    responses=error_responses(401, 422, 429, 503),
    dependencies=[Depends(rate_limit("default"))],
)
async def list_history(
    limit: int = Query(50, ge=1, le=200),
    before: datetime | None = Query(None, description="Only return events detected before this time (ISO 8601)"),
    user: AuthenticatedUser = Depends(get_current_user),
    db: Database = Depends(get_database),
) -> AlertHistoryResponse:
    rows = await db.list_alert_history(user.user_id, limit=limit, before=before)
    items = [AlertHistoryItem.model_validate({**r, "id": str(r["id"])}) for r in rows]
    next_before = items[-1].detected_at if len(items) == limit else None
    return AlertHistoryResponse(items=items, next_before=next_before)


@router.delete(
    "",
    status_code=204,
    summary="Delete all of the authenticated user's detection history",
    responses=error_responses(401, 429, 503),
    dependencies=[Depends(rate_limit("default"))],
)
async def clear_history(
    user: AuthenticatedUser = Depends(get_current_user),
    db: Database = Depends(get_database),
) -> Response:
    await db.delete_alert_history(user.user_id)
    return Response(status_code=204)
