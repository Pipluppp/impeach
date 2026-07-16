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
    assert transcript["segment_count"] == len(segments) == 4710
    assert all(segment["time_domain"] == "session_video" for segment in segments)
    assert all(
        current["start"] <= following["start"] and current["end"] >= current["start"]
        for current, following in zip(segments, segments[1:])
    )
    for boundary in range(1800, 19801, 1800):
        before = [segment for segment in segments if (segment["start"] + segment["end"]) / 2 < boundary]
        after = [segment for segment in segments if (segment["start"] + segment["end"]) / 2 >= boundary]
        assert before[-1]["normalized_text"] != after[0]["normalized_text"]
    assert transcript["runtime"]["producer_json_repairs"]


def test_final_alignment_is_honest_and_monotonic() -> None:
    alignment = load(SESSION / "alignment" / "alignments.json")
    entries = alignment["entries"]

    assert alignment["policy"]["exhibit_timestamps_are_session_seeks"] is False
    assert alignment["policy"]["summary_precision"] == "bounded_context_not_verbatim"
    assert alignment["policy"]["speaker_label_is_spoken_query"] is False
    assert alignment["policy"]["uncertain_blocks_may_be_unaligned"] is True
    assert all(current["start"] <= following["start"] for current, following in zip(entries, entries[1:]))
    assert sum(entry["review_state"] == "manual_reviewed" for entry in entries) == 11
    assert sum(entry["review_state"] == "auto_accepted" for entry in entries) == 598
    assert alignment["diagnostics"]["abstained_spoken_blocks"] == 847
    assert any(entry["precision"] == "narrative_summary_range" for entry in entries)
    assert any(entry["precision"] == "approximate_dialogue_turn" for entry in entries)


def test_committed_public_payload_matches_alignment_and_schema() -> None:
    public_path = ROOT / "web" / "public" / "data" / "sessions" / "2026-07-14.json"
    payload = load(public_path)
    schema = load(ROOT / "pipeline" / "schemas" / "public-session.schema.json")
    validate_payload(payload, schema)

    summary = payload["processing"]["alignment_summary"]
    assert summary == {
        "total_blocks": 2405,
        "timed_blocks": 2105,
        "coverage": 0.8753,
        "needs_review": 1496,
        "manual_reviewed": 11,
        "unresolved_conflicts": 0,
    }
    assert '"normalized_text"' not in public_path.read_text(encoding="utf-8")
