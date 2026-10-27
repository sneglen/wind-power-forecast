"""HTTP plumbing shared by the downloaders: retries on transient errors and throttling.

Retried: transport errors, HTTP 429 and 5xx. A 429 waits as long as the server asks
(``Retry-After`` header, Energi Data Service's "Try again in N seconds", Open-Meteo's
"try again in the next hour"). A daily quota is not retried: it raises ``LimitExceeded``.
Any other status is returned to the caller, which decides what it means.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

USER_AGENT = "wind-power-forecast (https://github.com/sneglen/wind-power-forecast)"
MAX_ATTEMPTS = 6
SLEEP: Callable[[float], None] = time.sleep  # replaced in tests


class RetryableError(RuntimeError):
    def __init__(self, message: str, wait_s: float | None = None) -> None:
        super().__init__(message)
        self.wait_s = wait_s


class LimitExceeded(RuntimeError):
    """A quota that only resets much later (e.g. per day); retrying now is pointless."""


def client(timeout_s: float = 120.0) -> httpx.Client:
    return httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT})


def server_wait_s(response: httpx.Response) -> float | None:
    """Seconds the server asks us to wait, if it says so."""
    header = response.headers.get("Retry-After")
    if header and header.isdigit():
        return float(header)
    text = response.text.lower()
    match = re.search(r"try again in (\d+) seconds", text)
    if match:
        return float(match.group(1))
    if "next hour" in text:
        now = dt.datetime.now(dt.UTC)
        return 3600.0 - (now.minute * 60 + now.second) + 60.0
    return None


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, RetryableError) and exc.wait_s is not None:
        return exc.wait_s + 1.0
    return min(120.0, 5.0 * 2 ** (state.attempt_number - 1))


def get(http: httpx.Client, url: str, params: dict[str, Any]) -> httpx.Response:
    """GET with retries; returns the final response for any non-retryable status."""

    def attempt() -> httpx.Response:
        response = http.get(url, params=params)
        if response.status_code == 429 and "daily" in response.text.lower():
            raise LimitExceeded(f"{url}: {response.text[:200]}")
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableError(
                f"HTTP {response.status_code} from {url}: {response.text[:200]}",
                server_wait_s(response),
            )
        return response

    retrying = Retrying(
        retry=retry_if_exception_type((RetryableError, httpx.TransportError)),
        wait=_wait,
        stop=stop_after_attempt(MAX_ATTEMPTS),
        sleep=lambda s: SLEEP(s),
        reraise=True,
    )
    return retrying(attempt)


class Throttle:
    """Keeps at least ``min_interval_s`` between consecutive calls to ``wait``."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval_s - (time.monotonic() - self._last)
            if remaining > 0:
                SLEEP(remaining)
        self._last = time.monotonic()
