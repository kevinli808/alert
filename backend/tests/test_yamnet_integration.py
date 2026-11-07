"""Real-model tests (downloads YAMNet on first run). Opt in with:  RUN_MODEL_TESTS=1 pytest -s

Synthetic signals are NOT a substitute for real recordings; they only prove the full pipeline
(decode -> resample -> YAMNet -> mapping) works and give a latency figure. Calibrate thresholds
with scripts/evaluate_model.py on real audio.
"""

import os
import time

import numpy as np
import pytest

from app.services import sound_categories as sc
from app.services.audio_processor import AudioProcessor
from app.services.detection_service import DetectionService
from app.services.prediction_processor import PredictionProcessor
from app.services.sound_classifier import YamnetClassifier
from tests.conftest import wav_bytes

pytestmark = pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")


@pytest.fixture(scope="module")
def yamnet():
    clf = YamnetClassifier("https://tfhub.dev/google/yamnet/1")
    clf.load()
    return clf


def _wav(signal: np.ndarray, sr: int = 44_100) -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def smoke_alarm_t3(sr=44_100, seconds=3.0):
    """Temporal-3 pattern: 3.1 kHz beeps, 0.5 s on / 0.5 s off."""
    t = np.arange(int(sr * seconds)) / sr
    gate = ((t % 1.0) < 0.5).astype(np.float32)
    return 0.6 * np.sin(2 * np.pi * 3100 * t) * gate


def siren_wail(sr=44_100, seconds=3.0):
    t = np.arange(int(sr * seconds)) / sr
    freq = 900 + 500 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return 0.6 * np.sin(phase)


def test_class_map_covers_every_mapping(yamnet):
    assert len(yamnet.class_names) == 521
    _, missing = sc.build_label_index(yamnet.class_names)
    assert missing == []


@pytest.mark.parametrize("name, signal", [("smoke_alarm_t3", smoke_alarm_t3()), ("siren_wail", siren_wail())])
def test_pipeline_on_synthetic_signals(yamnet, name, signal):
    service = DetectionService(
        AudioProcessor(2_000_000, 0.5, 10, 0.003, ffmpeg_path=None), yamnet, PredictionProcessor(0.75, 10)
    )
    outcome = service.detect(_wav(signal), "integration-user")
    print(f"\n{name}: detected={outcome.detected} sound_type={outcome.sound_type} conf={outcome.confidence} "
          f"candidates={outcome.candidates} labels={outcome.top_labels} timings={outcome.timings_ms}")
    assert outcome.reason in ("detected", "below_threshold")
    assert outcome.top_labels and all(0 <= s <= 1 for _, s in outcome.top_labels)


def test_latency_2s_chunk(yamnet):
    service = DetectionService(
        AudioProcessor(2_000_000, 0.5, 10, 0.003, ffmpeg_path=None), yamnet, PredictionProcessor(0.75, 10)
    )
    data = wav_bytes(2.0, amp=0.3)
    service.detect(data, "warm")
    runs = []
    for _ in range(10):
        t = time.perf_counter()
        service.detect(data, "lat")
        runs.append((time.perf_counter() - t) * 1000)
    p50 = float(np.median(runs))
    print(f"\n2 s chunk pipeline latency: p50={p50:.1f} ms, max={max(runs):.1f} ms")
    assert p50 < 1000
