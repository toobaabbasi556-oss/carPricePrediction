"""Polite, single-worker HTTP client for scraping (SPEC §2.1 rules 2-3, 6-7).

Enforces the crawl delay, uses one worker (no concurrency), sends a descriptive
User-Agent, logs every request (url, status, timestamp), and — critically — does
NOT add stealth/evasion. On repeated 403/Cloudflare blocks it raises ``BlockedError``
so the caller STOPs and reports instead of trying anti-detection tricks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

from .robots import RobotsRules

logger = logging.getLogger("pkcar.scrape")


class BlockedError(RuntimeError):
    """Raised on 403/Cloudflare blocks. Signal to STOP and report (SPEC §2.1 rule 7)."""


class DisallowedByRobots(RuntimeError):
    """Raised when a URL path is disallowed by robots.txt (SPEC §2.1 rule 1)."""


# Transient server-side errors: retry politely, don't crash the crawl.
_TRANSIENT_STATUS = (500, 502, 503, 504)


@dataclass
class RequestRecord:
    """One logged request (SPEC §2.1 rule 6)."""

    url: str
    status: int | None
    timestamp: str


def _looks_like_cloudflare(resp: requests.Response) -> bool:
    """Heuristic: Cloudflare challenge/block pages carry these markers."""
    if resp.status_code in (403, 503):
        server = resp.headers.get("Server", "").lower()
        body = resp.text[:4000].lower()
        markers = ("cloudflare", "cf-ray", "attention required", "just a moment")
        return "cloudflare" in server or any(m in body for m in markers)
    return False


class PoliteFetcher:
    """Single-worker fetcher that respects crawl-delay and robots.txt."""

    def __init__(
        self,
        user_agent: str,
        delay_sec: float,
        timeout_sec: int = 30,
        max_retries: int = 2,
        rules: RobotsRules | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.delay_sec = delay_sec
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.rules = rules
        self._last_request_ts: float | None = None
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self.log: list[RequestRecord] = []

    def _respect_delay(self) -> None:
        if self._last_request_ts is not None:
            elapsed = time.monotonic() - self._last_request_ts
            wait = self.delay_sec - elapsed
            if wait > 0:
                time.sleep(wait)

    def get(self, url: str, path_for_robots: str | None = None) -> requests.Response:
        """Fetch a URL politely. Raises DisallowedByRobots / BlockedError as needed."""
        if self.rules is not None:
            path = path_for_robots or url
            # Strip scheme+host to a path for the robots check.
            if path.startswith("http"):
                from urllib.parse import urlsplit

                split = urlsplit(path)
                path = split.path + (f"?{split.query}" if split.query else "")
            if not self.rules.can_fetch(path):
                raise DisallowedByRobots(f"robots.txt disallows {path!r}")

        attempt = 0
        while True:
            self._respect_delay()
            ts = datetime.now(UTC).isoformat()
            try:
                resp = self._session.get(url, timeout=self.timeout_sec)
            except requests.RequestException as exc:
                self._last_request_ts = time.monotonic()
                self.log.append(RequestRecord(url=url, status=None, timestamp=ts))
                logger.warning("GET %s -> error %s", url, exc)
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                self._backoff(attempt)
                continue

            self._last_request_ts = time.monotonic()
            self.log.append(RequestRecord(url=url, status=resp.status_code, timestamp=ts))
            logger.info("GET %s -> %s", url, resp.status_code)

            if _looks_like_cloudflare(resp):
                # No evasion. Retry a bounded number of times, then STOP and report.
                if attempt >= self.max_retries:
                    raise BlockedError(
                        f"Blocked (HTTP {resp.status_code}, Cloudflare) on {url}. "
                        "STOP — do not add anti-detection tricks (SPEC §2.1 rule 7)."
                    )
                attempt += 1
                self._backoff(attempt)
                continue

            if resp.status_code in _TRANSIENT_STATUS:
                # Transient server hiccup (e.g. 502 Bad Gateway): retry, then give up on
                # THIS url — the caller skips it rather than aborting the whole crawl.
                if attempt >= self.max_retries:
                    logger.warning("GET %s -> %s after %s retries", url, resp.status_code, attempt)
                    resp.raise_for_status()
                attempt += 1
                self._backoff(attempt)
                continue

            resp.raise_for_status()
            return resp

    def _backoff(self, attempt: int) -> None:
        """Extra polite pause after a transient failure (on top of the crawl delay)."""
        time.sleep(min(self.delay_sec * attempt, 15.0))
