"""Dependencies that expose the shared service container (created once at startup)."""

from __future__ import annotations

from fastapi import Depends, Request

from app.core.errors import ServiceNotConfiguredError
from app.services.container import ServiceContainer
from app.services.database_service import Database


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services


def get_database(services: ServiceContainer = Depends(get_services)) -> Database:
    if services.database is None:
        raise ServiceNotConfiguredError("Database is not configured on the server")
    return services.database
