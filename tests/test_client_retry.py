"""PoliteFetcher retry/robustness tests (SPEC §2.1 rules 2, 7).

No network: a fake session returns scripted responses. Verifies transient 5xx are
retried then surfaced (so the crawl skips one url, not aborts), 4xx raise immediately,
and a Cloudflare block raises BlockedError (STOP, no evasion).
"""

from __future__ import annotations

import pytest
import requests

from pkcar.scrape.client import BlockedError, PoliteFetcher


class _FakeResp:
    def __init__(self, status: int, text: str = "", server: str = "") -> None:
        self.status_code = status
        self.text = text
        self.headers = {"Server": server}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


class _FakeSession:
    def __init__(self, responses: list[_FakeResp]) -> None:
        self._responses = responses
        self.headers: dict[str, str] = {}
        self.calls = 0

    def get(self, url: str, timeout: int) -> _FakeResp:
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


def _fetcher(responses: list[_FakeResp], max_retries: int = 3) -> PoliteFetcher:
    f = PoliteFetcher("ua", delay_sec=0.0, max_retries=max_retries)
    f._session = _FakeSession(responses)  # type: ignore[assignment]
    f._backoff = lambda attempt: None  # type: ignore[method-assign]  # no sleeping in tests
    return f


def test_success_returns_response() -> None:
    f = _fetcher([_FakeResp(200, "ok")])
    assert f.get("http://x").status_code == 200


def test_transient_502_retried_then_recovers() -> None:
    f = _fetcher([_FakeResp(502), _FakeResp(502), _FakeResp(200, "ok")])
    resp = f.get("http://x")
    assert resp.status_code == 200
    assert f._session.calls == 3  # type: ignore[attr-defined]


def test_transient_502_exhausts_and_raises() -> None:
    f = _fetcher([_FakeResp(502)], max_retries=2)
    with pytest.raises(requests.HTTPError):
        f.get("http://x")
    assert f._session.calls == 3  # type: ignore[attr-defined]  # initial + 2 retries


def test_404_raises_immediately_no_retry() -> None:
    f = _fetcher([_FakeResp(404)])
    with pytest.raises(requests.HTTPError):
        f.get("http://x")
    assert f._session.calls == 1  # type: ignore[attr-defined]


def test_cloudflare_block_raises_blocked_error() -> None:
    cf = _FakeResp(403, text="Attention Required! | Cloudflare", server="cloudflare")
    f = _fetcher([cf], max_retries=1)
    with pytest.raises(BlockedError):
        f.get("http://x")
