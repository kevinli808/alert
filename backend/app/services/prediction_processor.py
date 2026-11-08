"""Turns raw classifier scores into an alert decision.

Responsibilities:
  * Confidence threshold (global, with optional per-category overrides). The 0.6 default is an
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

# Quiet "normal" sounds (doorbell, knock, dog, phone...) often score below their threshold when the room
# is louder than they are. Two weak hits in a row for the same sound still count: a real doorbell or dog
# repeats, random noise rarely looks like the same sound twice. Measured on ESC-50 clips at -25 dB this
# caught 15/48 instead of 11/48 with no extra false alarms in 5 min of everyday sounds; applying it to
# critical/high sounds too added false alarms, so it is limited to "normal" priority.
WEAK_FRACTION = 0.5  # weak hit = at least half of the category's threshold...
WEAK_MIN = 0.15  # ...and never below this
CONFIRM_WINDOW_SECONDS = 5.0  # the two chunks must be consecutive (the browser sends one every 2 s)

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
        self._clock = clock
        self._previous: dict[str, tuple[float, dict[str, float]]] = {}  # user -> (time, last chunk's scores)
        self._previous_lock = threading.Lock()

    def threshold_for(self, sound_type: str) -> float:
        return self.category_thresholds.get(sound_type, self.threshold)

    def _weak_threshold(self, sound_type: str) -> float | None:
        if sc.CATEGORY_BY_ID[sound_type].priority != "normal":
            return None
        return max(WEAK_MIN, self.threshold_for(sound_type) * WEAK_FRACTION)

    def _previous_scores(self, user_id: str, scores: dict[str, float]) -> dict[str, float]:
        """Remember this chunk and return the previous one if it was consecutive."""
        now = self._clock()
        with self._previous_lock:
            if len(self._previous) > 10_000:
                self._previous.clear()
            prev_time, prev = self._previous.get(user_id, (None, {}))
            self._previous[user_id] = (now, scores)
        if prev_time is None or now - prev_time > CONFIRM_WINDOW_SECONDS:
            return {}
        return prev

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

        previous = self._previous_scores(user_id, scores)

        def passes(cat: str, score: float) -> bool:
            if score >= self.threshold_for(cat):
                return True
            weak = self._weak_threshold(cat)
            return weak is not None and score >= weak and previous.get(cat, 0.0) >= weak

        passing = [(k, v) for k, v in ranked if passes(k, v)]
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
