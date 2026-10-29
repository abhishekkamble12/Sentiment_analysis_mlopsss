"""Small bounded in-memory cache with per-entry expiry (swap for Redis later)."""
import time
from collections.abc import MutableMapping


class TTLCache(MutableMapping):
    def __init__(self, ttl_seconds: float = 3600.0, max_size: int = 1000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._data: dict = {}

    def _live(self, key) -> bool:
        item = self._data.get(key)
        if item is None:
            return False
        if item[0] < time.monotonic():
            del self._data[key]
            return False
        return True

    def __contains__(self, key) -> bool:
        return self._live(key)

    def __getitem__(self, key):
        if not self._live(key):
            raise KeyError(key)
        return self._data[key][1]

    def __setitem__(self, key, value) -> None:
        if key not in self._data and len(self._data) >= self.max_size:
            self._data.pop(next(iter(self._data)))  # evict the oldest entry
        self._data[key] = (time.monotonic() + self.ttl, value)

    def __delitem__(self, key) -> None:
        del self._data[key]

    def __iter__(self):
        return (k for k in list(self._data) if self._live(k))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def clear(self) -> None:
        self._data.clear()
