"""Databricks Model Serving inference provider (optional; local YAMNet is the default).

PLACEHOLDER UNTIL YOU DEPLOY A MODEL: this client only works once you have logged and served the
MLflow model defined in databricks/yamnet_pyfunc.py and set DATABRICKS_HOST, DATABRICKS_TOKEN and
DATABRICKS_ENDPOINT_NAME. Until then the status is "not_configured" and nothing is sent.

Request contract (sent to POST {host}/serving-endpoints/{endpoint}/invocations):
    {"dataframe_records": [{"audio_b64": "<base64 little-endian int16 PCM, mono, 16 kHz>",
                            "sample_rate": 16000}]}
Expected response:
    {"predictions": [{"category_scores": {"fire_alarm": 0.93, ...},
                      "top_labels": [["Fire alarm", 0.93], ...]}]}
int16 PCM keeps the payload ~4x smaller than JSON float lists (a 2 s chunk is ~85 KB as base64).
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import httpx
import numpy as np

from app.core.errors import InferenceError, ModelUnavailableError
from app.services.sound_classifier import ClassificationResult, SoundClassifier

logger = logging.getLogger(__name__)


class DatabricksClassifier(SoundClassifier):
    name = "databricks"

    def __init__(self, host: str, token: str, endpoint_name: str, timeout_seconds: float = 5.0) -> None:
        self.endpoint_name = endpoint_name
        self._configured = bool(host and token and endpoint_name)
        self._url = f"{host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"
        self._client = (
            httpx.Client(
                timeout=httpx.Timeout(timeout_seconds, connect=min(3.0, timeout_seconds)),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            if self._configured
            else None
        )

    @property
    def status(self) -> str:
        return "ready" if self._configured else "not_configured"

    @staticmethod
    def encode_waveform(waveform: np.ndarray) -> str:
        pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
        return base64.b64encode(pcm.tobytes()).decode("ascii")

    def classify(self, waveform: np.ndarray) -> ClassificationResult:
        if self._client is None:
            raise ModelUnavailableError("Databricks inference is not configured")
        payload = {"dataframe_records": [{"audio_b64": self.encode_waveform(waveform), "sample_rate": 16000}]}
        started = time.perf_counter()
        try:
            resp = self._client.post(self._url, json=payload)
        except httpx.TimeoutException as exc:
            # With scale-to-zero, the first requests after an idle period wait while Databricks starts a
            # replica (~1 min). That is "still starting", not a failure: the client keeps sending chunks,
            # which keep the endpoint awake once it is up.
            raise ModelUnavailableError("Sound model is waking up") from exc
        except httpx.HTTPError as exc:
            logger.warning("Databricks request error: %s", type(exc).__name__)
            raise InferenceError("Could not reach Databricks inference endpoint") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000

        if resp.status_code != 200:
            # Do not log the body verbatim at error level; it can echo request details.
            logger.warning("Databricks endpoint %s returned HTTP %s", self.endpoint_name, resp.status_code)
            if resp.status_code in (401, 403):
                raise InferenceError("Databricks rejected the credentials")
            if resp.status_code in (404, 429, 503, 504):
                raise ModelUnavailableError("Databricks model endpoint is not available yet")
            raise InferenceError("Databricks inference failed")

        try:
            return self._parse(resp.json(), elapsed_ms)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise InferenceError("Databricks returned an unexpected response format") from exc

    def _parse(self, body: dict[str, Any], elapsed_ms: float) -> ClassificationResult:
        prediction = body["predictions"][0]
        raw_scores = prediction["category_scores"]
        if not isinstance(raw_scores, dict):
            raise TypeError("category_scores must be an object")
        scores = {str(k): float(v) for k, v in raw_scores.items()}
        top = [(str(lbl), float(score)) for lbl, score in prediction.get("top_labels", [])][:3]
        return ClassificationResult(
            category_scores=scores,
            top_labels=top,
            provider=self.name,
            model=f"databricks:{self.endpoint_name}",
            inference_ms=elapsed_ms,
        )

    def wake(self, timeout_seconds: float = 3.0) -> str:
        """Send one tiny request so a scaled-to-zero endpoint starts now. Returns "ready" or "waking".

        Databricks starts scaling up when the request arrives, so a short timeout is fine: the caller
        does not wait for the replica, it only kicks off the start.
        """
        if self._client is None:
            return "not_configured"
        payload = {"dataframe_records": [{"audio_b64": self.encode_waveform(np.zeros(16000, np.float32)), "sample_rate": 16000}]}
        try:
            resp = self._client.post(self._url, json=payload, timeout=timeout_seconds)
        except httpx.HTTPError:
            return "waking"
        return "ready" if resp.status_code == 200 else "waking"

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
