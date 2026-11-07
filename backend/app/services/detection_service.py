"""Orchestrates the detection pipeline: preprocess -> classify -> post-process. Blocking (CPU bound);
call it from a worker thread."""

from __future__ import annotations

import time

from app.services.audio_processor import AudioProcessor
from app.services.prediction_processor import DetectionOutcome, PredictionProcessor
from app.services.sound_classifier import SoundClassifier


class DetectionService:
    def __init__(
        self, audio_processor: AudioProcessor, classifier: SoundClassifier, predictions: PredictionProcessor
    ) -> None:
        self.audio_processor = audio_processor
        self.classifier = classifier
        self.predictions = predictions

    def detect(self, audio_bytes: bytes, user_id: str) -> DetectionOutcome:
        started = time.perf_counter()
        audio = self.audio_processor.process(audio_bytes)
        preprocess_ms = (time.perf_counter() - started) * 1000

        if audio.is_silent:
            # Skip inference entirely for silent chunks: saves CPU and avoids spurious detections.
            outcome = self.predictions.silence()
            inference_ms = 0.0
        else:
            t = time.perf_counter()
            result = self.classifier.classify(audio.waveform)
            inference_ms = (time.perf_counter() - t) * 1000
            outcome = self.predictions.process(user_id, result)

        outcome.timings_ms = {
            "preprocess": round(preprocess_ms, 1),
            "inference": round(inference_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        }
        return outcome

    def warm_up(self) -> None:
        """Load the model and pay one-time library start-up costs (soxr/numba ~1 s) before the first
        real request. Blocking; run in a worker thread."""
        self.classifier.safe_load()
        try:
            import librosa
            import numpy as np

            librosa.resample(np.zeros(44_100, dtype=np.float32), orig_sr=44_100, target_sr=16_000, res_type="soxr_hq")
            librosa.to_mono(np.zeros((2, 1_000), dtype=np.float32))
        except Exception:  # noqa: BLE001 - warm-up is best effort
            pass

    def close(self) -> None:
        self.classifier.close()
