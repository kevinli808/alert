"""Cross-cutting security helpers: in-memory rate limiting and HTTP hardening middleware.

The rate limiter is per-process. That is fine for a single-instance hackathon deployment; if you
scale to several instances, move it to a shared store (e.g. Redis) or your API gateway.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import RateLimitError, error_body


class RateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary string (e.g. ``detect:<user_id>``)."""

    _PRUNE_EVERY = 1000

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, key: str, limit: int, window_seconds: float = 60.0) -> None:
        now = self._clock()
        cutoff = now - window_seconds
        with self._lock:
            self._calls += 1
            if self._calls % self._PRUNE_EVERY == 0:
                self._prune(cutoff)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = math.ceil(hits[0] + window_seconds - now)
                raise RateLimitError(retry_after)
            hits.append(now)

    def _prune(self, cutoff: float) -> None:
        stale = [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]
        for k in stale:
            del self._hits[k]


def install_http_middleware(app: FastAPI, max_body_bytes: int) -> None:
    """Reject oversized bodies early (before multipart parsing) and add security headers."""

    @app.middleware("http")
    async def _guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > max_body_bytes
            except ValueError:
                return JSONResponse(status_code=400, content=error_body("bad_request", "Invalid Content-Length"))
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content=error_body("payload_too_large", f"Request body exceeds {max_body_bytes} bytes"),
                )

        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response
