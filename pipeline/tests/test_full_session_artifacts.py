from __future__ import annotations

import json
from pathlib import Path

from pipeline.build_public_data import validate_payload


ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "data" / "sessions" / "2026-07-14"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_transcript_is_compact_monotonic_and_boundary_owned() -> None:
    transcript = load(SESSION / "transcript" / "segments.json")
    segments = transcript["segments"]

    assert transcript["purpose"] == "internal_alignment_aid"
    assert transcript["authoritative_public_text"] is False
    assert transcript["segment_count"] == len(segments)
    assert segments
    assert all(segment["time_domain"] == "session_video" for segment in segments)
    assert all(
        current["start"] <= following["start"] and current["end"] >= current["start"]
        for current, following in zip(segments, segments[1:])
    )
    assert len({segment["id"] for segment in segments}) == len(segments)
    assert transcript["runtime"]["chunk_count"] == 12
    assert isinstance(transcript["runtime"]["producer_json_repairs"], list)
    if transcript["configuration"]["engine"] == "whisper.cpp":
        quality = transcript["runtime"]["quality"]
        assert quality["strategy"] == "reset_context_windowed_with_fallback"
        assert quality["max_repeated_phrase_seconds"] < 45


def test_final_alignment_is_honest_and_monotonic() -> None:
    alignment = load(SESSION / "alignment" / "alignments.json")
    entries = alignment["entries"]

    assert alignment["policy"]["exhibit_timestamps_are_session_seeks"] is False
    assert alignment["policy"]["summary_precision"] == "bounded_context_not_verbatim"
    assert alignment["policy"]["speaker_label_is_spoken_query"] is False
    assert alignment["policy"]["uncertain_blocks_may_be_unaligned"] is True
    assert all(current["start"] <= following["start"] for current, following in zip(entries, entries[1:]))
    diagnostics = alignment["diagnostics"]
    assert sum(entry["review_state"] == "manual_reviewed" for entry in entries) == diagnostics[
        "manual_reviewed"
    ]
    assert sum(entry["review_state"] == "auto_accepted" for entry in entries) == diagnostics[
        "auto_accepted"
    ]
    assert diagnostics["abstained_spoken_blocks"] <= diagnostics["eligible_spoken_blocks"]
    assert any(entry["precision"] == "narrative_summary_range" for entry in entries)
    assert any(entry["precision"] == "approximate_dialogue_turn" for entry in entries)


def test_committed_public_payload_matches_alignment_and_schema() -> None:
    public_path = ROOT / "web" / "public" / "data" / "sessions" / "2026-07-14.json"
    payload = load(public_path)
    schema = load(ROOT / "pipeline" / "schemas" / "public-session.schema.json")
    validate_payload(payload, schema)

    blocks = [block for page in payload["pages"] for block in page["blocks"]]
    timed = [block for block in blocks if block["timing"] is not None]
    summary = payload["processing"]["alignment_summary"]
    assert summary["total_blocks"] == len(blocks)
    assert summary["timed_blocks"] == len(timed)
    assert summary["coverage"] == round(len(timed) / len(blocks), 4)
    assert summary["needs_review"] == sum(
        block["timing"]["review_state"] == "needs_review" for block in timed
    )
    assert summary["manual_reviewed"] == sum(
        block["timing"]["review_state"] == "manual_reviewed" for block in timed
    )
    assert summary["unresolved_conflicts"] >= 0
    assert '"normalized_text"' not in public_path.read_text(encoding="utf-8")
