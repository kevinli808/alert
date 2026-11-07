from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    id: str = Field(description="Auth0 user id (the token's `sub` claim)", examples=["google-oauth2|1098765"])
    email: str | None = None
    username: str | None = None
    created_at: datetime
