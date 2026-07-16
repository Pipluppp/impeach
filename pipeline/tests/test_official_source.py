from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from pipeline.official_source import OfficialSenateSource, OfficialSourceUnavailable


def test_browser_adapter_is_used_after_bounded_http_failures() -> None:
    calls: list[str] = []
    feed = {"parties": []}

    def blocked(url: str, accept: str) -> bytes:
        del accept
        calls.append(f"http:{url}")
        raise HTTPError(url, 403, "Forbidden", {}, None)

    def browser(url: str, accept: str) -> bytes:
        del accept
        calls.append(f"browser:{url}")
        return json.dumps(feed).encode()

    source = OfficialSenateSource(
        http_reader=blocked, browser_reader=browser, attempts=2, sleeper=lambda _: None
    )
    payload, adapter = source.read_feed()
    assert payload == feed
    assert adapter == "browser"
    assert [call.split(":", 1)[0] for call in calls] == ["http", "http", "browser"]


def test_pdf_validation_happens_at_source_interface() -> None:
    source = OfficialSenateSource(
        http_reader=lambda _url, _accept: b"not a pdf",
        browser_reader=lambda _url, _accept: b"unused",
        attempts=1,
    )
    with pytest.raises(OfficialSourceUnavailable, match="PDF signature"):
        source.read_pdf("https://example.invalid/document.pdf")
