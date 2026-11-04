"""A small in-memory sliding-window rate limiter.

It protects the Groq free-tier quota from a single client looping /ask. State is
per process, which is fine for this deployment (one replica, scale-to-zero); with
several replicas each would enforce its own limit, and a shared store such as
Redis would be the next step.
"""

import math
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests, window_seconds, clock=time.monotonic):
        self.max_requests = max_requests
        self.window = window_seconds
        self.clock = clock
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key):
        """Record a request. Returns (allowed, retry_after_seconds)."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False, max(1, math.ceil(self.window - (now - hits[0])))
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory: drop clients idle for a full window
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return True, 0
