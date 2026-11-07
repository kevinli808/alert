"""Turns raw classifier scores into an alert decision.

Responsibilities:
  * Confidence threshold (global, with optional per-category overrides). The 0.75 default is an
    EXPERIMENTAL starting point — calibrate it with scripts/evaluate_model.py on real recordings.
  * Choosing one category when several pass (critical sounds win over normal ones).
  * Event grouping / cooldown: while the same sound keeps being detected for a user, every chunk
    belongs to the same event (``is_new_event`` is False) so the frontend does not re-trigger, and
    history is written once per event, not once per chunk.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

from app.core.errors import InferenceError
from app.services import sound_categories as sc
from app.services.sound_classifier import ClassificationResult

Reason = Literal["detected", "below_threshold", "silence"]


@dataclass
class DetectionOutcome:
    detected: bool
    sound_type: str | None
    confidence: float
    timestamp: datetime
    reason: Reason
    is_new_event: bool = False
    event_id: str | None = None
    event_started_at: datetime | None = None
    candidates: list[tuple[str, float]] = field(default_factory=list)
    top_labels: list[tuple[str, float]] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class _Event:
    event_id: str
    started_at: datetime
    last_seen: float  # monotonic seconds


class EventTracker:
    """Groups repeated detections of the same sound for the same user into one event.

    In-memory and per-process (fine for a single-instance MVP).
    """

    def __init__(self, cooldown_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.cooldown = cooldown_seconds
        self._clock = clock
        self._events: dict[tuple[str, str], _Event] = {}
        self._lock = threading.Lock()

    def register(self, user_id: str, sound_type: str, timestamp: datetime) -> tuple[_Event, bool]:
        now = self._clock()
        key = (user_id, sound_type)
        with self._lock:
            if len(self._events) > 10_000:
                self._prune(now)
            event = self._events.get(key)
            if event is not None and now - event.last_seen <= self.cooldown:
                event.last_seen = now
                return event, False
            event = _Event(event_id=str(uuid.uuid4()), started_at=timestamp, last_seen=now)
            self._events[key] = event
            return event, True

    def _prune(self, now: float) -> None:
        for key in [k for k, e in self._events.items() if now - e.last_seen > self.cooldown]:
            del self._events[key]


class PredictionProcessor:
    def __init__(
        self,
        threshold: float,
        cooldown_seconds: float,
        category_thresholds: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = threshold
        self.category_thresholds = category_thresholds or {}
        self.events = EventTracker(cooldown_seconds, clock)

    def threshold_for(self, sound_type: str) -> float:
        return self.category_thresholds.get(sound_type, self.threshold)

    def silence(self, timestamp: datetime | None = None) -> DetectionOutcome:
        return DetectionOutcome(
            detected=False,
            sound_type=None,
            confidence=0.0,
            timestamp=timestamp or datetime.now(timezone.utc),
            reason="silence",
        )

    def process(
        self, user_id: str, result: ClassificationResult, timestamp: datetime | None = None
    ) -> DetectionOutcome:
        ts = timestamp or datetime.now(timezone.utc)

        scores: dict[str, float] = {}
        for cat, score in result.category_scores.items():
            if cat not in sc.CATEGORY_IDS:
                continue  # ignore categories a remote model knows but this app does not
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise InferenceError("Classifier returned invalid scores")
            scores[cat] = float(score)
        if not scores:
            raise InferenceError("Classifier returned no category scores")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        common = dict(
            timestamp=ts,
            candidates=[(k, round(v, 4)) for k, v in ranked[:3]],
            top_labels=[(k, round(v, 4)) for k, v in result.top_labels],
            provider=result.provider,
            model=result.model,
        )

        passing = [(k, v) for k, v in ranked if v >= self.threshold_for(k)]
        if not passing:
            return DetectionOutcome(
                detected=False,
                sound_type=sc.UNKNOWN_SOUND,
                confidence=round(ranked[0][1], 4),
                reason="below_threshold",
                **common,
            )

        # Several categories may pass (e.g. fire alarm + smoke detector). Prefer the most urgent.
        passing.sort(key=lambda kv: (sc.PRIORITY_RANK[sc.CATEGORY_BY_ID[kv[0]].priority], -kv[1]))
        sound_type, score = passing[0]
        event, is_new = self.events.register(user_id, sound_type, ts)
        return DetectionOutcome(
            detected=True,
            sound_type=sound_type,
            confidence=round(score, 4),
            reason="detected",
            is_new_event=is_new,
            event_id=event.event_id,
            event_started_at=event.started_at,
            **common,
        )
