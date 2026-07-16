from __future__ import annotations

import json
from pathlib import Path

from pipeline.parse_journal import normalize_for_matching, parse_document


ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "sessions" / "2026-07-14" / "source" / "journal.pdf"


def parsed() -> dict:
    _, payload = parse_document(PDF, journal_number=6, source_revision=1)
    return payload


def test_raw_and_normalized_text_are_separate() -> None:
    payload = parsed()
    trap = next(block for block in payload["blocks"] if "timestamp 51:07" in block["raw_text"])
    assert "D- 1-145-A" in trap["raw_text"]
    assert "d-1-145-a" in trap["normalized_text"]
    assert trap["features"] == ["exhibit_timestamp"]
    assert trap["time_references"][0]["time_domain"] == "exhibit"
    assert trap["time_references"][0]["time_seconds"] == 3067
    assert trap["time_references"][0]["exhibit_id"] == "D-1-145-A"


def test_sampled_headings_speakers_and_pages_are_detected() -> None:
    payload = parsed()
    blocks = payload["blocks"]
    headings = {(block["page"], block["raw_text"]) for block in blocks if block["kind"] == "heading"}
    assert (3, "CALL TO ORDER") in headings
    assert (5, "CROSS-EXAMINATION BY THE COUNSEL FOR THE RESPONDENT {Continuation)") in headings
    assert (62, "SUSPENSION OF TRIAL") in headings
    assert (63, "REDIRECT EXAMINATION OE THE WITNESS") in headings
    assert (92, "ADJOURNMENT OF THE IMPEACHMENT TRIAL") in headings
    greeting = next(block for block in blocks if block.get("speaker_raw") == "Mr. Vinluan" and block["raw_text"].startswith("Good afternoon"))
    assert greeting["page"] == 5
    assert greeting["kind"] == "speaker_utterance"
    assert payload["page_count"] == 92
    assert all(block["page"] >= 1 and block["sequence"] >= 1 for block in blocks)


def test_page_boundary_continuation_is_explicit() -> None:
    payload = parsed()
    continuations = [block for block in payload["blocks"] if "continuation_of" in block]
    assert continuations
    assert all(block["page"] > 1 for block in continuations)


def test_repeated_headings_have_distinct_outline_section_ids() -> None:
    payload = parsed()
    repeated = [
        block["section_id"]
        for block in payload["blocks"]
        if block["kind"] == "heading"
        and block["raw_text"] == "ORDER OF THE PRESIDING OFFICER"
    ]
    assert len(repeated) == 3
    assert len(repeated) == len(set(repeated))


def test_malformed_character_cleanup_only_changes_matching_text() -> None:
    raw = "ORDER OF THE PRESH)ING OFFICER — Wimess"
    normalized = normalize_for_matching(raw)
    assert raw == "ORDER OF THE PRESH)ING OFFICER — Wimess"
    assert normalized == "order of the presiding officer witness"


def test_representative_parser_fixtures() -> None:
    payload = parsed()
    fixture = json.loads(
        (ROOT / "pipeline" / "fixtures" / "parser" / "patterns.json").read_text(encoding="utf-8")
    )
    for case in fixture["cases"]:
        block = next(
            item
            for item in payload["blocks"]
            if item["page"] == case["page"]
            and case["contains"].casefold()
            in f"{item.get('speaker_raw', '')}. {item['raw_text']}".casefold()
        )
        if "expected_kind" in case:
            assert block["kind"] == case["expected_kind"], case["name"]
        if "expected_speaker" in case:
            assert block["speaker_raw"] == case["expected_speaker"], case["name"]
        if case.get("expects_continuation"):
            assert "continuation_of" in block, case["name"]
        if "normalized_contains" in case:
            assert case["normalized_contains"] in block["normalized_text"], case["name"]


def test_markdown_and_blocks_are_deterministic() -> None:
    markdown_one, payload_one = parse_document(PDF, journal_number=6, source_revision=1)
    markdown_two, payload_two = parse_document(PDF, journal_number=6, source_revision=1)
    assert markdown_one == markdown_two
    assert payload_one == payload_two
