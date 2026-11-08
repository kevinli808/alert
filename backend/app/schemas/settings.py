"""The user's alert configuration, in the frontend's own model (src/components/AlertApp.tsx).

Sound rules map recognised sounds/phrases onto effects; one of a rule's effects is a visual state
(system: default/warning/urgent, or a user-made custom state). The backend stores this as one
document per user and validates it strictly (unknown fields are rejected) so nothing but these
settings can be saved. JSON keys are camelCase to match the frontend exactly.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator
from pydantic.alias_generators import to_camel

HEX_COLOR = r"^#[0-9A-Fa-f]{6}$"
ITEM_ID = r"^[A-Za-z0-9_-]{1,64}$"
SYSTEM_STATE_IDS = frozenset({"default", "warning", "urgent"})


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)


class CustomVisualState(_Model):
    id: str = Field(pattern=ITEM_ID)
    name: str = Field(min_length=1, max_length=60)
    color: str = Field(pattern=HEX_COLOR)
    priority: Literal[0, 1, 2] = 1
    duration: int = Field(ge=0, le=60_000, description="How long the state stays active, in ms")
    system: Literal[False] | None = None  # system states are built into the app, never stored

    @field_validator("id")
    @classmethod
    def _not_system(cls, value: str) -> str:
        if value in SYSTEM_STATE_IDS:
            raise ValueError(f"'{value}' is a built-in state id")
        return value


class SoundRule(_Model):
    id: str = Field(pattern=ITEM_ID)
    label: str = Field(min_length=1, max_length=80)
    kind: Literal["sound", "phrase"]
    enabled: bool = True
    effects: list[str] = Field(default_factory=list, max_length=20)
    tts_response: str | None = Field(default=None, max_length=300)
    light_color: str | None = Field(default=None, pattern=HEX_COLOR)

    @field_validator("effects")
    @classmethod
    def _check_effects(cls, value: list[str]) -> list[str]:
        for effect in value:
            if not re.fullmatch(ITEM_ID, effect):
                raise ValueError(f"Invalid effect id {effect!r}")
        if len(value) != len(set(value)):
            raise ValueError("Effect ids must be unique")
        return value


class UserSettings(_Model):
    simple: bool = False
    rules: list[SoundRule] = Field(max_length=100)
    custom_states: list[CustomVisualState] = Field(default_factory=list, max_length=30)
    state_order: list[str] = Field(default_factory=list, max_length=40)
    intro_complete: bool = False

    @model_validator(mode="after")
    def _check_references(self) -> "UserSettings":
        rule_ids = [r.id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Rule ids must be unique")
        state_ids = [s.id for s in self.custom_states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("Custom state ids must be unique")
        known = SYSTEM_STATE_IDS | set(state_ids)
        unknown = [s for s in self.state_order if s not in known]
        if unknown:
            raise ValueError(f"stateOrder references unknown states: {unknown}")
        if len(self.state_order) != len(set(self.state_order)):
            raise ValueError("stateOrder entries must be unique")
        return self


class SettingsUpdate(_Model):
    settings: UserSettings


class SettingsResponse(_Model):
    settings: UserSettings | None = Field(description="null until the user saves settings for the first time")
    updated_at: datetime | None = None

    @field_serializer("settings")
    def _settings_as_saved(self, value: UserSettings | None) -> dict | None:
        # Return the document exactly as the frontend stores it: camelCase, no null placeholders.
        return None if value is None else value.model_dump(mode="json", by_alias=True, exclude_none=True)
