from __future__ import annotations

import pytest

from pipeline.alignment import align
from pipeline.transcribe import normalize_for_matching


def fixture_payloads():
    journal = {
        "blocks": [
            {"id": "b1", "kind": "speaker_utterance", "raw_text": "Yesterday you identified videos, correct?", "speaker_raw": "Mr. A"},
            {"id": "b2", "kind": "narrative_summary", "raw_text": "The trial proceeded."},
            {"id": "b3", "kind": "speaker_utterance", "raw_text": "I will show you timestamp 51:07 onwards of the video.", "speaker_raw": "Mr. A", "time_references": [{"time_domain": "exhibit", "time_seconds": 3067}]},
        ]
    }
    transcript = {
        "session_id": "s1", "youtube_video_id": "video", "runtime": {"audio_seconds": 100},
        "segments": [
            {"id": "asr-1", "start": 10.0, "end": 13.0, "text": "Yesterday you identified videos correct", "normalized_text": "yesterday you identified videos correct"},
            {"id": "asr-2", "start": 20.0, "end": 24.0, "text": "I will show you timestamp 51 07 onwards of the video", "normalized_text": "i will show you timestamp 51 07 onwards of the video"},
        ],
    }
    return journal, transcript


def test_dialogue_and_summary_never_claim_same_precision() -> None:
    journal, transcript = fixture_payloads()
    result = align(journal, transcript, {"overrides": []})
    entries = {item["block_id"]: item for item in result["entries"]}
    assert entries["b1"]["precision"] == "approximate_dialogue_turn"
    assert entries["b2"]["precision"] == "narrative_summary_range"
    assert "word-level" in entries["b1"]["evidence"]["claim_limit"]
    assert entries["b2"]["review_state"] == "needs_review"


def test_exhibit_time_is_preserved_but_never_used_as_session_seek() -> None:
    journal, transcript = fixture_payloads()
    result = align(journal, transcript, {"overrides": []})
    trap = next(item for item in result["entries"] if item["block_id"] == "b3")
    assert trap["time_domain"] == "session_video"
    assert trap["start"] < 100
    assert trap["source_time_references"][0] == {"time_domain": "exhibit", "time_seconds": 3067}
    assert result["policy"]["exhibit_timestamps_are_session_seeks"] is False


def test_manual_override_survives_and_is_validated() -> None:
    journal, transcript = fixture_payloads()
    override = {"overrides": [{"block_id": "b2", "start": 15, "end": 19, "time_domain": "session_video", "note": "reviewed", "reviewer": "fixture"}]}
    result = align(journal, transcript, override)
    entry = next(item for item in result["entries"] if item["block_id"] == "b2")
    assert entry["review_state"] == "manual_reviewed"
    assert (entry["start"], entry["end"]) == (15.0, 19.0)
    bad = {"overrides": [{"block_id": "b2", "start": 3067, "end": 3068, "time_domain": "exhibit", "note": "bad"}]}
    with pytest.raises(ValueError):
        align(journal, transcript, bad)
    unknown = {"overrides": [{"block_id": "missing", "start": 1, "end": 2, "time_domain": "session_video", "note": "bad"}]}
    with pytest.raises(ValueError, match="unknown manual override"):
        align(journal, transcript, unknown)


def test_manual_anchor_bounds_automatic_search() -> None:
    journal, transcript = fixture_payloads()
    override = {
        "overrides": [
            {
                "block_id": "b2", "start": 5, "end": 6,
                "time_domain": "session_video", "note": "reviewed before auto anchor",
            }
        ]
    }
    result = align(journal, transcript, override)

    assert all(entry["block_id"] != "b1" for entry in result["entries"])
    assert result["policy"]["uncertain_blocks_may_be_unaligned"] is True
    assert result["unresolved_conflicts"] == []


def transcript(*segments: tuple[float, float, str]) -> dict:
    return {
        "session_id": "s1", "youtube_video_id": "video", "runtime": {"audio_seconds": 100},
        "segments": [
            {
                "id": f"asr-{index}", "start": start, "end": end, "text": text,
                "normalized_text": normalize_for_matching(text),
            }
            for index, (start, end, text) in enumerate(segments, start=1)
        ],
    }


def test_speaker_label_is_not_searched_as_spoken_audio() -> None:
    journal = {"blocks": [{
        "id": "b1", "kind": "speaker_utterance", "speaker_raw": "Mr. Wronglabel",
        "raw_text": "The cybersecurity division authenticated the documentary evidence.",
    }]}
    result = align(journal, transcript(
        (10, 12, "Mr Wronglabel discussed something entirely unrelated"),
        (20, 24, "The cybersecurity division authenticated the documentary evidence"),
    ))
    entry = next(item for item in result["entries"] if item["block_id"] == "b1")
    assert entry["start"] == 20
    assert result["policy"]["speaker_label_is_spoken_query"] is False


def test_monotonic_path_beats_a_later_local_duplicate() -> None:
    journal = {"blocks": [
        {"id": "b1", "kind": "speaker_utterance", "raw_text": "Alpha committee authenticated the first unique document."},
        {"id": "b2", "kind": "speaker_utterance", "raw_text": "Beta committee verified the second separate record."},
    ]}
    result = align(journal, transcript(
        (10, 14, "Alpha committee authenticated first unique document"),
        (20, 24, "Beta committee verified the second separate record"),
        (30, 34, "Alpha committee authenticated the first unique document"),
    ))
    entries = {item["block_id"]: item for item in result["entries"]}
    assert entries["b1"]["start"] == 10
    assert entries["b2"]["start"] == 20


def test_split_join_asr_words_match_without_changing_source_text() -> None:
    journal = {"blocks": [{
        "id": "b1", "kind": "speaker_utterance",
        "raw_text": "The cyber crime division authenticated the documentary evidence.",
    }]}
    result = align(journal, transcript(
        (10, 14, "The cybercrime division authenticated the documentary evidence"),
    ))
    entry = next(item for item in result["entries"] if item["block_id"] == "b1")
    assert entry["start"] == 10
    assert journal["blocks"][0]["raw_text"] == "The cyber crime division authenticated the documentary evidence."


def test_structural_text_is_context_only_and_generic_reply_abstains() -> None:
    journal = {"blocks": [
        {"id": "b1", "kind": "speaker_utterance", "raw_text": "The first distinctive evidentiary statement begins now."},
        {"id": "heading", "kind": "heading", "raw_text": "CROSS EXAMINATION OF THE WITNESS"},
        {"id": "generic", "kind": "speaker_utterance", "raw_text": "Yes, Your Honor."},
        {"id": "b2", "kind": "speaker_utterance", "raw_text": "The second distinctive evidentiary statement concludes now."},
    ]}
    result = align(journal, transcript(
        (10, 14, "The first distinctive evidentiary statement begins now"),
        (15, 17, "Cross examination of the witness"),
        (18, 20, "Yes your honor"),
        (22, 26, "The second distinctive evidentiary statement concludes now"),
    ))
    entries = {item["block_id"]: item for item in result["entries"]}
    assert entries["heading"]["precision"] == "section_range"
    assert entries["heading"]["evidence"]["method"] == "bounded_monotonic_context"
    assert entries["generic"]["review_state"] == "needs_review"
    assert entries["generic"]["precision"] == "contextual_dialogue_range"
    assert result["policy"]["structural_text_is_spoken_query"] is False
