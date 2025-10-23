import os
import time
import threading
from typing import Optional


class TokenBucketLimiter:
    """
    Simple thread-safe token bucket rate limiter for synchronous code.

    rate_per_minute: tokens added per minute
    burst: maximum tokens that can be accumulated
    """

    def __init__(self, rate_per_minute: float, burst: Optional[float] = None):
        self.rate_per_minute = float(max(rate_per_minute, 0.01))
        self.tokens = float(burst if burst is not None else self.rate_per_minute)
        self.capacity = float(burst if burst is not None else self.rate_per_minute)
        self.timestamp = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.timestamp
        # tokens per second
        rps = self.rate_per_minute / 60.0
        self.tokens = min(self.capacity, self.tokens + elapsed * rps)
        self.timestamp = now

    def acquire(self):
        """Block until a token is available, then consume one token."""
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # compute wait time for next token
                rps = self.rate_per_minute / 60.0
                wait = max(0.05, (1.0 - self.tokens) / rps) if rps > 0 else 1.0
            time.sleep(min(wait, 5.0))


_google_limiter: Optional[TokenBucketLimiter] = None


def get_google_limiter() -> TokenBucketLimiter:
    """Singleton Google search limiter configured via env vars.

    Env:
      GOOGLE_SEARCH_RPM: requests per minute (default 12)
      GOOGLE_SEARCH_BURST: bucket size (default equals RPM)
    """
    global _google_limiter
    if _google_limiter is None:
        rpm_str = os.getenv("GOOGLE_SEARCH_RPM", "6")
        burst_str = os.getenv("GOOGLE_SEARCH_BURST", "2")
        try:
            rpm = float(rpm_str)
        except Exception:
            rpm = 12.0
        try:
            burst = float(burst_str) if burst_str else 2.0
        except Exception:
            burst = 2.0
        _google_limiter = TokenBucketLimiter(rate_per_minute=rpm, burst=burst)
    return _google_limiter
