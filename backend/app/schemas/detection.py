from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ScoreItem(BaseModel):
    label: str
    confidence: float


class InferenceInfo(BaseModel):
    provider: str | None = Field(default=None, examples=["local_yamnet"])
    model: str | None = None
    preprocess_ms: float
    inference_ms: float
    total_ms: float


class DetectionResponse(BaseModel):
    detected: bool = Field(description="True when an app sound category passed its confidence threshold")
    sound_type: str | None = Field(
        description="Category id when detected; 'unknown' when sound was present but nothing passed the "
        "threshold; null for silence",
        examples=["fire_alarm"],
    )
    confidence: float = Field(ge=0, le=1, description="Score of sound_type (or best candidate)")
    timestamp: datetime
    reason: Literal["detected", "below_threshold", "silence"]
    is_new_event: bool = Field(
        description="True only for the first chunk of a new event. Trigger the visual alert when true; "
        "later chunks of the same continuing sound return false (cooldown/grouping)."
    )
    event_id: str | None = None
    event_started_at: datetime | None = None
    candidates: list[ScoreItem] = Field(default_factory=list, description="Top app categories by score")
    model_labels: list[ScoreItem] = Field(default_factory=list, description="Top raw model labels (debugging)")
    inference: InferenceInfo


class SoundTypeInfo(BaseModel):
    id: str
    display_name: str
    priority: Literal["critical", "high", "normal"]
    default_color: str
    default_flash_speed: int
    model_labels: list[str]
    confidence_threshold: float


class SoundTypesResponse(BaseModel):
    sound_types: list[SoundTypeInfo]


class AlertHistoryItem(BaseModel):
    id: str
    sound_type: str
    confidence: float
    detected_at: datetime


class AlertHistoryResponse(BaseModel):
    items: list[AlertHistoryItem]
    next_before: datetime | None = Field(
        default=None, description="Pass as `before` to fetch the next (older) page; null when no more"
    )
