"""A small in-process sliding-window rate limiter.

Limits how many calls one client may make in a time window. State lives in this process only,
so with several app instances each would allow the full limit. Behind a reverse proxy the client
address is the proxy's unless the platform forwards the real one. Good enough for a single-instance
demo; use a shared store (for example Redis) or the platform's own limiter for more.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        if limit < 0:
            raise ValueError("limit must be >= 0 (0 disables limiting)")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._calls: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client_id: str) -> Optional[float]:
        """Record a call. Returns None if allowed, otherwise the seconds until a call would be allowed."""
        if self.limit == 0:
            return None
        now = self._clock()
        with self._lock:
            calls = self._calls.setdefault(client_id, deque())
            while calls and now - calls[0] >= self.window:
                calls.popleft()
            if len(calls) >= self.limit:
                return max(self.window - (now - calls[0]), 0.001)
            calls.append(now)
            return None

    def forget_idle(self) -> None:
        """Drop clients with no recent calls so the table cannot grow without bound."""
        now = self._clock()
        with self._lock:
            for client_id in [c for c, d in self._calls.items() if not d or now - d[-1] >= self.window]:
                del self._calls[client_id]
