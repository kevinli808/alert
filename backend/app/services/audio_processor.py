"""Audio preprocessing: bytes from the browser -> mono float32 waveform at 16 kHz in [-1, 1].

Independent from any model. Decoding strategy:
  1. libsndfile (via soundfile): WAV, FLAC, OGG/Vorbis, OGG/Opus, MP3 — no external tools needed.
  2. ffmpeg (only if installed): WebM/Opus (Chrome/Firefox MediaRecorder), MP4/AAC (Safari).
The recommended frontend path is to send 16 kHz mono 16-bit PCM WAV (see docs/react-integration.md),
which needs no ffmpeg and no resampling.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

from app.core.errors import InvalidAudioError, PayloadTooLargeError

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000  # YAMNet requirement


@dataclass(frozen=True)
class ProcessedAudio:
    waveform: np.ndarray  # float32, mono, TARGET_SAMPLE_RATE, peak <= 1.0
    sample_rate: int
    duration_seconds: float
    original_sample_rate: int
    original_channels: int
    rms: float
    is_silent: bool


class AudioProcessor:
    def __init__(
        self,
        max_bytes: int,
        min_seconds: float,
        max_seconds: float,
        silence_rms_threshold: float,
        ffmpeg_path: str | None = "auto",
    ) -> None:
        self.max_bytes = max_bytes
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.silence_rms_threshold = silence_rms_threshold
        self.ffmpeg_path = shutil.which("ffmpeg") if ffmpeg_path == "auto" else ffmpeg_path

    def process(self, data: bytes) -> ProcessedAudio:
        if not data:
            raise InvalidAudioError("Audio upload is empty")
        if len(data) > self.max_bytes:
            raise PayloadTooLargeError(f"Audio exceeds the {self.max_bytes}-byte limit")

        samples, sample_rate, channels = self._decode(data)

        if samples.size == 0 or sample_rate <= 0:
            raise InvalidAudioError("Audio contains no samples")
        if not np.all(np.isfinite(samples)):
            raise InvalidAudioError("Audio contains invalid sample values")

        duration = samples.shape[-1] / float(sample_rate)
        if duration < self.min_seconds:
            raise InvalidAudioError(f"Audio is too short ({duration:.2f}s, minimum {self.min_seconds}s)")
        if duration > self.max_seconds:
            raise InvalidAudioError(f"Audio is too long ({duration:.2f}s, maximum {self.max_seconds}s)")

        mono = librosa.to_mono(samples) if samples.ndim > 1 else samples
        if sample_rate != TARGET_SAMPLE_RATE:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE, res_type="soxr_hq")
        mono = np.ascontiguousarray(mono, dtype=np.float32)

        # Only scale DOWN to avoid clipping. We do not boost quiet audio: amplifying background noise
        # to full scale makes the classifier hallucinate events.
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 1.0:
            mono /= peak

        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        return ProcessedAudio(
            waveform=mono,
            sample_rate=TARGET_SAMPLE_RATE,
            duration_seconds=round(duration, 3),
            original_sample_rate=int(sample_rate),
            original_channels=channels,
            rms=rms,
            is_silent=rms < self.silence_rms_threshold,
        )

    # ------------------------------------------------------------------------------------
    def _decode(self, data: bytes) -> tuple[np.ndarray, int, int]:
        """Return (samples shaped (channels, n) or (n,), sample_rate, channels)."""
        try:
            frames, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            channels = frames.shape[1]
            samples = frames[:, 0] if channels == 1 else frames.T  # librosa layout: (channels, n)
            return samples, int(sample_rate), channels
        except Exception as sf_exc:  # libsndfile raises several error types for unknown formats
            if not self.ffmpeg_path:
                raise InvalidAudioError(
                    "Unsupported or corrupted audio. Send WAV/FLAC/OGG, or install ffmpeg on the server "
                    "to accept WebM/MP4."
                ) from sf_exc
        return self._decode_with_ffmpeg(data), TARGET_SAMPLE_RATE, 1

    def _decode_with_ffmpeg(self, data: bytes) -> np.ndarray:
        # A temp file (not a pipe) lets ffmpeg seek, which MP4 containers from Safari need.
        fd, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            proc = subprocess.run(
                [
                    self.ffmpeg_path or "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", path,
                    "-t", str(self.max_seconds + 1),  # bound decoded size; longer clips still fail validation
                    "-vn", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-f", "f32le", "pipe:1",
                ],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise InvalidAudioError("Audio decoding timed out") from exc
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if proc.returncode != 0 or not proc.stdout:
            logger.info("ffmpeg could not decode upload (exit %s)", proc.returncode)
            raise InvalidAudioError("Unsupported or corrupted audio")
        return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
