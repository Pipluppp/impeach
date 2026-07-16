"""Retrieve official Senate records through bounded HTTP and browser adapters."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SENATE_FEED_URL = "https://senate.gov.ph/hq/impeachment/published"
SENATE_LISTING_URL = "https://senate.gov.ph/services/impeachment-documents"
SENATE_HTTP_USER_AGENT = (
    "SenatePublicDocumentsMonitor/1.0 (+https://github.com/Pipluppp/impeach)"
)
SENATE_PDF_PATH_RE = re.compile(
    r"^/hq/uploads/impeachment/(?P<filename>[0-9a-f-]+\.pdf)$", re.IGNORECASE
)


class OfficialSourceUnavailable(RuntimeError):
    """Neither official-source adapter returned a valid bounded response."""


Reader = Callable[[str, str], bytes]


def _http_reader(url: str, accept: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": SENATE_LISTING_URL,
            "User-Agent": SENATE_HTTP_USER_AGENT,
        },
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def _browser_reader(url: str, accept: str) -> bytes:
    del accept
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as exc:
        raise OfficialSourceUnavailable(
            "browser fallback requires the pinned selenium dependency"
        ) from exc

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(45)
        driver.set_script_timeout(60)
        driver.get(SENATE_LISTING_URL)
        result = driver.execute_async_script(
            """
            const [url, done] = arguments;
            fetch(url, {credentials: "include", cache: "no-store"})
              .then(async response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const bytes = new Uint8Array(await response.arrayBuffer());
                let binary = "";
                for (let offset = 0; offset < bytes.length; offset += 32768) {
                  binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
                }
                done({ok: true, body: btoa(binary)});
              })
              .catch(error => done({ok: false, error: String(error)}));
            """,
            url,
        )
    finally:
        driver.quit()
    if not isinstance(result, dict) or not result.get("ok"):
        detail = result.get("error") if isinstance(result, dict) else repr(result)
        raise OfficialSourceUnavailable(f"browser fetch failed for {url}: {detail}")
    return base64.b64decode(result["body"], validate=True)


def _configured_relay_reader(url: str, accept: str) -> bytes:
    relay_base = os.environ.get("SENATE_SOURCE_RELAY_URL", "").rstrip("/")
    relay_token = os.environ.get("SENATE_SOURCE_RELAY_TOKEN", "")
    if not relay_base or not relay_token:
        raise OfficialSourceUnavailable("Senate source relay URL/token is not configured")
    if url == SENATE_FEED_URL:
        relay_url = f"{relay_base}/feed"
    else:
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        match = SENATE_PDF_PATH_RE.fullmatch(parsed.path)
        if parsed.scheme != "https" or parsed.hostname != "senate.gov.ph" or not match:
            raise OfficialSourceUnavailable("relay refused a non-Senate document URL")
        relay_url = f"{relay_base}/document/{match.group('filename')}"
    request = Request(
        relay_url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {relay_token}",
            "User-Agent": SENATE_HTTP_USER_AGENT,
        },
    )
    with urlopen(request, timeout=45) as response:
        return response.read()


class OfficialSenateSource:
    """Small interface hiding source retries, browser fallback, and validation."""

    def __init__(
        self,
        *,
        http_reader: Reader = _http_reader,
        browser_reader: Reader = _browser_reader,
        relay_reader: Reader | None = None,
        attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if attempts < 1 or attempts > 5:
            raise ValueError("attempts must be between 1 and 5")
        self._http_reader = http_reader
        self._browser_reader = browser_reader
        self._relay_reader = relay_reader or (
            _configured_relay_reader
            if os.environ.get("SENATE_SOURCE_RELAY_URL")
            and os.environ.get("SENATE_SOURCE_RELAY_TOKEN")
            else None
        )
        self._attempts = attempts
        self._sleeper = sleeper

    def _read(self, url: str, accept: str) -> tuple[bytes, str]:
        errors: list[str] = []
        for attempt in range(self._attempts):
            try:
                return self._http_reader(url, accept), "http"
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                errors.append(f"http attempt {attempt + 1}: {exc}")
                if attempt + 1 < self._attempts:
                    self._sleeper(float(attempt + 1))
        try:
            return self._browser_reader(url, accept), "browser"
        except Exception as exc:
            errors.append(f"browser fallback: {exc}")
        if self._relay_reader is not None:
            try:
                return self._relay_reader(url, accept), "cloudflare_relay"
            except Exception as exc:
                errors.append(f"relay fallback: {exc}")
        raise OfficialSourceUnavailable(f"official fetch failed for {url}; {'; '.join(errors)}")

    def read_feed(self) -> tuple[dict[str, Any], str]:
        content, adapter = self._read(SENATE_FEED_URL, "application/json,text/plain,*/*")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OfficialSourceUnavailable(f"official feed is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(payload, dict) or "parties" not in payload:
            raise OfficialSourceUnavailable("official feed lacks the expected parties collection")
        return payload, adapter

    def read_pdf(self, url: str) -> tuple[bytes, str]:
        content, adapter = self._read(url, "application/pdf,*/*")
        if not content.startswith(b"%PDF-"):
            raise OfficialSourceUnavailable("official document lacks a PDF signature")
        return content, adapter

__all__ = [
    "OfficialSenateSource",
    "OfficialSourceUnavailable",
    "SENATE_FEED_URL",
    "SENATE_LISTING_URL",
]
