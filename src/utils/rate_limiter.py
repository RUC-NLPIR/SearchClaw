"""
Per-domain rate limiting for web requests.

Uses aiolimiter for async-compatible rate limiting. Each domain gets
its own rate limiter to prevent overwhelming any single server while
allowing concurrent requests to different domains.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Lazy import — aiolimiter may not be installed in tests
_AsyncLimiter = None
_limiter_resolved = False


def _get_limiter_class():
    global _AsyncLimiter, _limiter_resolved
    if not _limiter_resolved:
        try:
            from aiolimiter import AsyncLimiter
            _AsyncLimiter = AsyncLimiter
        except ImportError:
            _AsyncLimiter = None
        _limiter_resolved = True
    return _AsyncLimiter


class DomainRateLimiter:
    """
    Per-domain rate limiter for web requests.

    Each unique domain gets its own rate limiter. This prevents
    overwhelming any single server while allowing full throughput
    across different domains.

    Usage:
        limiter = DomainRateLimiter(max_per_minute=10)
        await limiter.acquire("https://example.com/page")
        # ... make request ...
    """

    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self._limiters: dict[str, object] = {}

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc or url
        except Exception:
            return url

    def _get_limiter(self, domain: str):
        """Get or create a rate limiter for a domain."""
        if domain not in self._limiters:
            LimiterClass = _get_limiter_class()
            if LimiterClass is not None:
                # max_rate requests per 60 seconds
                self._limiters[domain] = LimiterClass(
                    max_rate=self.max_per_minute,
                    time_period=60,
                )
            else:
                # No rate limiter available — create a no-op
                self._limiters[domain] = None
        return self._limiters[domain]

    async def acquire(self, url: str) -> None:
        """
        Acquire rate limit permission for a URL.

        Blocks (async) if the domain's rate limit is exceeded.
        No-op if aiolimiter is not installed.
        """
        domain = self._get_domain(url)
        limiter = self._get_limiter(domain)

        if limiter is not None:
            await limiter.acquire()
            logger.debug(f"Rate limit acquired for {domain}")
        else:
            logger.debug(f"Rate limiting disabled (no aiolimiter), skipping for {domain}")
