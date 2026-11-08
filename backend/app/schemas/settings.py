from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


EffectId = Annotated[str, Field(min_length=1, max_length=40)]
StateId = Annotated[str, Field(min_length=1, max_length=100)]


class SoundRuleSettings(StrictSettingsModel):
    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    label: str = Field(min_length=1, max_length=80)
    kind: Literal["sound", "phrase"]
    enabled: bool
    effects: list[EffectId] = Field(max_length=20)
    tts_response: str | None = Field(default=None, alias="ttsResponse", max_length=300)
    light_color: str | None = Field(
        default=None,
        alias="lightColor",
        pattern=r"^#[0-9a-fA-F]{6}$",
    )


class VisualStateSettings(StrictSettingsModel):
    id: str = Field(min_length=1, max_length=100, pattern=r"^custom-[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=50)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    priority: Literal[0, 1, 2] = 1
    duration: int = Field(ge=0, le=60_000)
    system: bool = False


class ProfileSettings(StrictSettingsModel):
    simple: bool
    rules: list[SoundRuleSettings] = Field(max_length=50)
    custom_states: list[VisualStateSettings] = Field(
        default_factory=list,
        alias="customStates",
        max_length=20,
    )
    state_order: list[StateId] = Field(alias="stateOrder", max_length=70)
    selected_rule_id: str = Field(alias="selectedRuleId", min_length=1, max_length=100)
    priority_collapsed: bool = Field(alias="priorityCollapsed")
    intro_complete: bool = Field(alias="introComplete")

    @model_validator(mode="after")
    def validate_references(self) -> ProfileSettings:
        rule_ids = [rule.id for rule in self.rules]
        # "phrase" is the purple state for spoken phrases. Settings saved before it existed have no
        # "phrase" in stateOrder; they stay valid and the frontend inserts it on load.
        state_ids = ["default", "warning", "phrase", "urgent", *(state.id for state in self.custom_states)]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rule ids must be unique")
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("custom state ids must be unique")
        if self.selected_rule_id not in rule_ids:
            raise ValueError("selectedRuleId must identify a saved rule")
        if len(self.state_order) != len(set(self.state_order)) or set(self.state_order) not in (
            set(state_ids),
            set(state_ids) - {"phrase"},
        ):
            raise ValueError("stateOrder must contain each available state exactly once")
        allowed_effects = {
            *state_ids,
            "vibration",
            "tts",
            "smart-lights",
            "watch-haptics",
            "screen-flash",
            "persistent-notifications",
            "escalating-alerts",
            "emergency-notifications",
        }
        if any(effect not in allowed_effects for rule in self.rules for effect in rule.effects):
            raise ValueError("rules may only use available effect ids")
        return self


class ProfileSettingsResponse(StrictSettingsModel):
    settings: ProfileSettings | None
