from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.sound_categories import CATEGORY_IDS, SOUND_CATEGORIES

HEX_COLOR = r"^#[0-9A-Fa-f]{6}$"


def _validate_sound_type(value: str) -> str:
    if value not in CATEGORY_IDS:
        raise ValueError(f"Unknown sound_type '{value}'. See GET /api/sound-types.")
    return value


class AlertPreferenceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sound_type: str = Field(examples=["doorbell"])
    color: str = Field(pattern=HEX_COLOR, examples=["#0000FF"], description="Hex colour #RRGGBB")
    enabled: bool = True
    flash_speed: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Flashes per second (1-3). Capped at 3 to stay within WCAG 2.3.1 seizure guidance.",
    )

    @field_validator("sound_type")
    @classmethod
    def _check_sound_type(cls, value: str) -> str:
        return _validate_sound_type(value)

    @field_validator("color")
    @classmethod
    def _upper_color(cls, value: str) -> str:
        return value.upper()


class AlertPreference(AlertPreferenceIn):
    updated_at: datetime | None = Field(default=None, description="null when the default is in effect")
    is_default: bool = False


class PreferencesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferences: list[AlertPreferenceIn] = Field(min_length=1, max_length=len(SOUND_CATEGORIES))

    @model_validator(mode="after")
    def _unique_sound_types(self) -> "PreferencesUpdate":
        seen = [p.sound_type for p in self.preferences]
        if len(seen) != len(set(seen)):
            raise ValueError("Each sound_type may appear only once")
        return self


class PreferencesResponse(BaseModel):
    preferences: list[AlertPreference]
