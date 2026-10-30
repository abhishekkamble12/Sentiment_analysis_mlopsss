"""Sliding-window rate limiter, in process.

shortcut: counts live in this process, so with several workers/replicas each enforces its own limit; move the
counters to Redis when you scale out. Memory is bounded: a client's timestamps are dropped once they leave the window.
"""
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Hashable, Optional, Tuple

WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self) -> None:
        self._hits: Dict[Hashable, Deque[float]] = defaultdict(deque)
        self._calls = 0

    def allow(self, key: Hashable, limit: int, now: Optional[float] = None) -> Tuple[bool, float]:
        """(allowed, seconds until a slot frees up)."""
        now = time.monotonic() if now is None else now
        hits = self._hits[key]
        while hits and hits[0] <= now - WINDOW_SECONDS:
            hits.popleft()
        self._calls += 1
        if self._calls % 1000 == 0:  # forget clients that went quiet
            for k in [k for k, q in self._hits.items() if not q or q[-1] <= now - WINDOW_SECONDS]:
                del self._hits[k]
        if len(hits) >= limit:
            return False, hits[0] + WINDOW_SECONDS - now
        hits.append(now)
        return True, 0.0
