"""Application sound categories and their mapping onto YAMNet / AudioSet class labels.

This module deliberately depends only on numpy so it can be shipped unchanged inside the
Databricks MLflow model (see databricks/yamnet_pyfunc.py).

Notes on the mapping (verified against yamnet_class_map.csv at model load time — any label that
does not exist in the model's class map is logged as a warning and ignored):
  * "Fire alarm" and "Smoke detector, smoke alarm" are separate AudioSet classes, but they are
    acoustically similar (high-pitched beeps) and the model often confuses them. Both are "critical".
  * There is no "glass breaking" class. "Shatter" is the closest; "Glass" alone also fires on
    clinking glasses, so it is not included.
  * "Doorbell" and "Ding-dong" both cover doorbells; electronic chime doorbells vary a lot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

Priority = Literal["critical", "high", "normal"]
PRIORITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "normal": 2}

UNKNOWN_SOUND = "unknown"


@dataclass(frozen=True)
class SoundCategory:
    id: str
    display_name: str
    yamnet_labels: tuple[str, ...]
    priority: Priority
    default_color: str  # informational only; the app's colours come from the user's settings
    default_flash_speed: int  # flashes per second (1-3)


SOUND_CATEGORIES: tuple[SoundCategory, ...] = (
    SoundCategory("fire_alarm", "Fire alarm", ("Fire alarm",), "critical", "#FF0000", 3),
    SoundCategory("smoke_detector", "Smoke detector", ("Smoke detector, smoke alarm",), "critical", "#FF8C00", 3),
    SoundCategory(
        "emergency_siren",
        "Emergency siren",
        (
            "Siren",
            "Civil defense siren",
            "Police car (siren)",
            "Ambulance (siren)",
            "Fire engine, fire truck (siren)",
            "Emergency vehicle",
        ),
        "critical",
        "#DC143C",
        3,
    ),
    SoundCategory("glass_breaking", "Glass breaking", ("Shatter",), "high", "#9400D3", 2),
    SoundCategory("baby_crying", "Baby crying", ("Baby cry, infant cry",), "high", "#FFD700", 2),
    SoundCategory("car_alarm", "Car alarm", ("Car alarm",), "high", "#FF4500", 2),
    SoundCategory("alarm", "Other alarm / buzzer", ("Alarm", "Buzzer"), "high", "#FF6347", 2),
    SoundCategory("doorbell", "Doorbell", ("Doorbell", "Ding-dong"), "normal", "#0000FF", 1),
    SoundCategory("door_knock", "Door knock", ("Knock",), "normal", "#1E90FF", 1),
    SoundCategory(
        "phone_ringing", "Phone ringing", ("Telephone bell ringing", "Ringtone", "Telephone"), "normal", "#00C853", 1
    ),
    SoundCategory("alarm_clock", "Alarm clock", ("Alarm clock",), "normal", "#00BCD4", 1),
    SoundCategory(
        "dog_barking", "Dog barking", ("Bark", "Dog", "Bow-wow", "Yip", "Howl", "Growling"), "normal", "#8B4513", 1
    ),
)

CATEGORY_BY_ID: dict[str, SoundCategory] = {c.id: c for c in SOUND_CATEGORIES}
CATEGORY_IDS: frozenset[str] = frozenset(CATEGORY_BY_ID)


def build_label_index(
    class_names: Sequence[str], categories: Iterable[SoundCategory] = SOUND_CATEGORIES
) -> tuple[dict[str, list[int]], list[str]]:
    """Map each category id to the model class indices that represent it.

    Returns (index, missing_labels). Missing labels are labels we expected but the model lacks.
    """
    position = {name: i for i, name in enumerate(class_names)}
    index: dict[str, list[int]] = {}
    missing: list[str] = []
    for cat in categories:
        idx = []
        for label in cat.yamnet_labels:
            if label in position:
                idx.append(position[label])
            else:
                missing.append(label)
        index[cat.id] = idx
    return index, missing


def clip_scores(frame_scores: np.ndarray, method: str = "max") -> np.ndarray:
    """Collapse per-frame scores (frames x classes) into one score per class."""
    scores = np.asarray(frame_scores, dtype=np.float32)
    if scores.ndim == 1:
        return scores
    if scores.shape[0] == 0:
        return np.zeros(scores.shape[1], dtype=np.float32)
    return scores.max(axis=0) if method == "max" else scores.mean(axis=0)


def category_scores(per_class: np.ndarray, label_index: dict[str, list[int]]) -> dict[str, float]:
    """Score of a category = highest score among its model classes."""
    return {cat: (float(per_class[idx].max()) if idx else 0.0) for cat, idx in label_index.items()}


def top_labels(per_class: np.ndarray, class_names: Sequence[str], k: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(per_class)[::-1][:k]
    return [(class_names[i], float(per_class[i])) for i in order]
