"""Sound classification providers.

``SoundClassifier`` is the replaceable interface. Implementations:
  * ``YamnetClassifier``      — local TensorFlow inference with Google's pretrained YAMNet (default).
  * ``DatabricksClassifier``  — remote inference on Databricks Model Serving (databricks_service.py).
  * ``FallbackClassifier``    — try a primary provider, fall back to another on failure.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.core.errors import AppError, InferenceError, ModelUnavailableError
from app.services import sound_categories as sc

logger = logging.getLogger(__name__)

YAMNET_FRAME_SAMPLES = 15_600  # one 0.975 s YAMNet frame at 16 kHz


@dataclass(frozen=True)
class ClassificationResult:
    category_scores: dict[str, float]  # app category id -> score in [0, 1]
    top_labels: list[tuple[str, float]] = field(default_factory=list)  # raw model labels, for debugging
    provider: str = ""
    model: str = ""
    inference_ms: float = 0.0


class SoundClassifier(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def status(self) -> str:
        """One of: not_loaded, loading, ready, failed, not_configured."""

    def is_ready(self) -> bool:
        return self.status == "ready"

    def load(self) -> None:
        """Load model resources. Blocking; called from a worker thread at startup."""

    def safe_load(self) -> None:
        try:
            self.load()
        except Exception:  # logged; status reports the failure and requests get 503
            logger.exception("Failed to load %s classifier", self.name)

    @abstractmethod
    def classify(self, waveform: np.ndarray) -> ClassificationResult:
        """Classify a mono float32 16 kHz waveform."""

    def close(self) -> None:
        return None


class YamnetClassifier(SoundClassifier):
    name = "local_yamnet"

    def __init__(self, model_handle: str, aggregation: str = "max") -> None:
        self.model_handle = model_handle
        self.aggregation = aggregation
        self._model = None
        self._class_names: list[str] = []
        self._label_index: dict[str, list[int]] = {}
        self._status = "not_loaded"
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        return self._status

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            self._status = "loading"
            started = time.perf_counter()
            try:
                import tensorflow as tf  # heavy import kept lazy so tests/tools can skip TensorFlow
                import tensorflow_hub as hub

                model = hub.load(self.model_handle)
                class_map_path = model.class_map_path().numpy().decode("utf-8")
                with tf.io.gfile.GFile(class_map_path) as fh:
                    class_names = [row["display_name"] for row in csv.DictReader(fh)]

                label_index, missing = sc.build_label_index(class_names)
                if missing:
                    logger.warning("YAMNet class map lacks expected labels (ignored): %s", missing)
                empty = [cat for cat, idx in label_index.items() if not idx]
                if empty:
                    logger.warning("Categories with no matching YAMNet class (always score 0): %s", empty)

                model(np.zeros(YAMNET_FRAME_SAMPLES, dtype=np.float32))  # warm-up: builds the graph once
            except Exception:
                self._status = "failed"
                raise
            self._model = model
            self._class_names = class_names
            self._label_index = label_index
            self._status = "ready"
            logger.info(
                "YAMNet loaded: %d classes in %.1fs", len(class_names), time.perf_counter() - started
            )

    def classify(self, waveform: np.ndarray) -> ClassificationResult:
        if self._model is None:
            raise ModelUnavailableError(
                "Sound model is still loading" if self._status == "loading" else "Sound model is not available"
            )
        started = time.perf_counter()
        try:
            scores, _embeddings, _spectrogram = self._model(np.asarray(waveform, dtype=np.float32))
            frame_scores = scores.numpy()
        except Exception as exc:
            logger.exception("YAMNet inference failed")
            raise InferenceError("Sound classification failed") from exc
        per_class = sc.clip_scores(frame_scores, self.aggregation)
        return ClassificationResult(
            category_scores=sc.category_scores(per_class, self._label_index),
            top_labels=sc.top_labels(per_class, self._class_names, k=3),
            provider=self.name,
            model=self.model_handle,
            inference_ms=(time.perf_counter() - started) * 1000,
        )


class FallbackClassifier(SoundClassifier):
    """Use ``primary``; on an inference/availability error use ``fallback`` instead."""

    def __init__(self, primary: SoundClassifier, fallback: SoundClassifier) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fallback:{fallback.name}"

    @property
    def status(self) -> str:
        if self.primary.is_ready() or self.fallback.is_ready():
            return "ready"
        return self.primary.status

    def load(self) -> None:
        self.primary.safe_load()
        self.fallback.safe_load()

    def classify(self, waveform: np.ndarray) -> ClassificationResult:
        if self.primary.is_ready():
            try:
                return self.primary.classify(waveform)
            except AppError as exc:
                if not self.fallback.is_ready():
                    raise
                logger.warning("%s failed (%s); using %s", self.primary.name, exc.code, self.fallback.name)
        return self.fallback.classify(waveform)

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()
