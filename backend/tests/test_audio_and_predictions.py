import numpy as np
import pytest

from app.core.errors import InferenceError, InvalidAudioError, PayloadTooLargeError
from app.services.audio_processor import AudioProcessor
from app.services.prediction_processor import PredictionProcessor
from app.services.sound_categories import CATEGORY_IDS, build_label_index, category_scores, clip_scores
from app.services.sound_classifier import ClassificationResult
from tests.conftest import wav_bytes


@pytest.fixture
def proc() -> AudioProcessor:
    return AudioProcessor(max_bytes=2_000_000, min_seconds=0.5, max_seconds=10, silence_rms_threshold=0.003,
                          ffmpeg_path=None)


# ------------------------------------------------------------------- audio ------------
def test_stereo_44k_is_converted_to_mono_16k(proc):
    out = proc.process(wav_bytes(2.0, sr=44_100, channels=2))
    assert out.sample_rate == 16_000 and out.waveform.ndim == 1 and out.waveform.dtype == np.float32
    assert abs(len(out.waveform) - 32_000) <= 2
    assert out.original_sample_rate == 44_100 and out.original_channels == 2
    assert not out.is_silent and float(np.max(np.abs(out.waveform))) <= 1.0


def test_16k_mono_passthrough(proc):
    out = proc.process(wav_bytes(1.0))
    assert len(out.waveform) == 16_000


def test_quiet_audio_is_not_boosted(proc):
    out = proc.process(wav_bytes(1.0, amp=0.01))
    assert float(np.max(np.abs(out.waveform))) < 0.02


def test_silence_detected(proc):
    assert proc.process(wav_bytes(1.0, amp=0.0)).is_silent


@pytest.mark.parametrize("data, exc", [
    (b"", InvalidAudioError),
    (b"\x00\x01garbage-bytes" * 50, InvalidAudioError),
    (b"0" * 2_000_001, PayloadTooLargeError),
])
def test_invalid_inputs(proc, data, exc):
    with pytest.raises(exc):
        proc.process(data)


def test_duration_limits(proc):
    with pytest.raises(InvalidAudioError, match="too short"):
        proc.process(wav_bytes(0.2))
    with pytest.raises(InvalidAudioError, match="too long"):
        proc.process(wav_bytes(11.0, sr=8000))


# ------------------------------------------------------------------- mapping -----------
def test_label_mapping_and_aggregation():
    names = ["Speech", "Fire alarm", "Doorbell", "Ding-dong"]
    index, missing = build_label_index(names)
    assert index["fire_alarm"] == [1] and sorted(index["doorbell"]) == [2, 3]
    assert "Smoke detector, smoke alarm" in missing
    frames = np.array([[0.9, 0.1, 0.2, 0.0], [0.1, 0.8, 0.1, 0.6]])
    per_class = clip_scores(frames, "max")
    scores = category_scores(per_class, index)
    assert scores["fire_alarm"] == pytest.approx(0.8) and scores["doorbell"] == pytest.approx(0.6)
    assert scores["smoke_detector"] == 0.0


# ------------------------------------------------------------------- predictions -------
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def result(**scores):
    return ClassificationResult(category_scores={c: 0.0 for c in CATEGORY_IDS} | scores)


def test_threshold_and_priority():
    p = PredictionProcessor(threshold=0.75, cooldown_seconds=10)
    out = p.process("u", result(doorbell=0.95, fire_alarm=0.8))
    assert out.sound_type == "fire_alarm"  # critical beats normal even with a lower score
    assert p.process("fresh-user", result(doorbell=0.74)).detected is False


def test_per_category_threshold():
    p = PredictionProcessor(threshold=0.75, cooldown_seconds=10, category_thresholds={"glass_breaking": 0.5})
    assert p.process("u", result(glass_breaking=0.55)).sound_type == "glass_breaking"


def test_quiet_normal_sound_confirmed_by_two_chunks_in_a_row():
    clock = Clock()
    p = PredictionProcessor(threshold=0.6, cooldown_seconds=10, clock=clock,
                            category_thresholds={"doorbell": 0.4, "fire_alarm": 0.4})
    assert p.process("u", result(doorbell=0.25)).detected is False  # one weak hit: ignored
    clock.t = 2
    out = p.process("u", result(doorbell=0.22))  # second weak hit in a row (>= 0.4 * 0.5)
    assert out.detected and out.sound_type == "doorbell" and out.is_new_event


def test_weak_hits_need_to_be_consecutive_and_same_sound():
    clock = Clock()
    p = PredictionProcessor(threshold=0.6, cooldown_seconds=10, clock=clock)
    p.process("u", result(dog_barking=0.35))
    clock.t = 8  # more than 5 s later: not consecutive
    assert p.process("u", result(dog_barking=0.35)).detected is False
    clock.t = 10
    assert p.process("u", result(doorbell=0.35)).detected is False  # different sound
    assert p.process("other", result(dog_barking=0.35)).detected is False  # other user's history


def test_weak_rule_is_only_for_normal_priority_sounds():
    clock = Clock()
    p = PredictionProcessor(threshold=0.6, cooldown_seconds=10, clock=clock)
    p.process("u", result(baby_crying=0.4, emergency_siren=0.4))  # high + critical
    clock.t = 2
    assert p.process("u", result(baby_crying=0.4, emergency_siren=0.4)).detected is False
    clock.t = 4
    assert p.process("u", result(baby_crying=0.4, emergency_siren=0.4, dog_barking=0.0)).detected is False


def test_previous_hint_from_client_when_server_has_no_memory():
    # Serverless: the second chunk lands on a fresh instance with no memory of the first one.
    p = PredictionProcessor(threshold=0.6, cooldown_seconds=10, category_thresholds={"doorbell": 0.4})
    out = p.process("u", result(doorbell=0.25), previous_hint={"doorbell": 0.3})
    assert out.detected and out.sound_type == "doorbell"
    fresh = PredictionProcessor(threshold=0.6, cooldown_seconds=10)
    assert fresh.process("u", result(emergency_siren=0.4), previous_hint={"emergency_siren": 0.9}).detected is False


def test_previous_hint_is_sanitized():
    from app.services.prediction_processor import sanitize_scores

    assert sanitize_scores({"doorbell": 0.3, "made_up": 0.9, "dog_barking": 7, "phone_ringing": True,
                            "door_knock": float("nan"), "alarm_clock": "0.5"}) == {"doorbell": 0.3}
    assert sanitize_scores(["doorbell"]) == {} and sanitize_scores(None) == {}
    p = PredictionProcessor(threshold=0.6, cooldown_seconds=10)
    assert p.process("u", result(doorbell=0.25), previous_hint={"doorbell": 5.0}).detected is False


def test_weak_rule_floor():
    clock = Clock()
    p = PredictionProcessor(threshold=0.2, cooldown_seconds=10, clock=clock)
    p.process("u", result(doorbell=0.12))
    clock.t = 2
    assert p.process("u", result(doorbell=0.12)).detected is False  # below the 0.15 floor


def test_cooldown_groups_continuing_sound():
    clock = Clock()
    p = PredictionProcessor(threshold=0.75, cooldown_seconds=10, clock=clock)
    a = p.process("u", result(doorbell=0.9))
    clock.t = 8
    b = p.process("u", result(doorbell=0.9))  # continuing — same event
    clock.t = 16
    c = p.process("u", result(doorbell=0.9))  # still within 10 s of the LAST detection
    clock.t = 40
    d = p.process("u", result(doorbell=0.9))  # gap > cooldown: new event
    assert a.is_new_event and not b.is_new_event and not c.is_new_event and d.is_new_event
    assert a.event_id == b.event_id == c.event_id != d.event_id
    assert p.process("other-user", result(doorbell=0.9)).is_new_event  # per-user


def test_invalid_scores_raise():
    p = PredictionProcessor(threshold=0.75, cooldown_seconds=10)
    with pytest.raises(InferenceError):
        p.process("u", result(doorbell=float("nan")))
    with pytest.raises(InferenceError):
        p.process("u", ClassificationResult(category_scores={"not_ours": 0.9}))
