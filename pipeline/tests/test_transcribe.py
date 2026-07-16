from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.transcribe import (
    Segment,
    TranscriptionError,
    merge_payloads,
    owned_segments,
    parse_whisper_cpp_json,
    parse_whisper_jsonl,
    repair_ffmpeg_whisper_line,
    write_json_atomic,
)


def test_whisper_jsonl_becomes_session_video_time() -> None:
    raw = '\n'.join([
        json.dumps({"start": 0, "end": 1250, "text": " Magandang afternoon. "}),
        json.dumps({"start": 1250, "end": 3000, "text": "Timestamp 51:07"}),
    ])
    segments = parse_whisper_jsonl(raw, global_offset_seconds=600)
    assert [(item.start, item.end) for item in segments] == [(600, 601.25), (601.25, 603)]
    assert segments[0].text == "Magandang afternoon."
    assert segments[1].normalized_text == "timestamp 51 07"


def test_whisper_cpp_json_offsets_become_session_video_time() -> None:
    raw = json.dumps({
        "transcription": [
            {
                "timestamps": {"from": "00:00:00,000", "to": "00:00:01,250"},
                "offsets": {"from": 0, "to": 1250},
                "text": " Magandang afternoon. ",
            },
            {
                "timestamps": {"from": "00:00:01,250", "to": "00:00:03,000"},
                "offsets": {"from": 1250, "to": 3000},
                "text": "Timestamp 51:07",
            },
        ]
    })
    segments = parse_whisper_cpp_json(raw, global_offset_seconds=600)
    assert [(item.start, item.end) for item in segments] == [(600, 601.25), (601.25, 603)]
    assert segments[0].text == "Magandang afternoon."


def test_chunk_overlap_has_single_owner_at_boundary() -> None:
    left = [Segment(98, 102, "left", "left"), Segment(103, 107, "right", "right")]
    right = [Segment(98.5, 101.5, "left repeat", "left repeat"), Segment(102, 106, "right", "right")]
    assert [item.text for item in owned_segments(left, own_start=None, own_end=102.5)] == ["left"]
    assert [item.text for item in owned_segments(right, own_start=102.5, own_end=None)] == ["right"]


def test_merge_is_monotonic_and_reassigns_ids() -> None:
    payloads = [
        {"segments": [{"id": "old", "start": 10.0, "end": 12.0, "text": "one"}]},
        {"segments": [{"id": "old", "start": 12.0, "end": 14.0, "text": "two"}]},
    ]
    merged = merge_payloads(payloads)
    assert [item["id"] for item in merged] == ["asr-000001", "asr-000002"]


def test_malformed_or_reversed_timestamp_fails() -> None:
    with pytest.raises(TranscriptionError):
        parse_whisper_jsonl('{"start": 20, "end": 10, "text": "bad"}', global_offset_seconds=0)


def test_known_ffmpeg_unescaped_quote_defect_is_repaired_and_recorded() -> None:
    line = '{"start":1211095,"end":1214895,"text":""There were threats to my life inside the batasang pambansa.""}'
    repaired = repair_ffmpeg_whisper_line(line)
    assert repaired["text"] == '"There were threats to my life inside the batasang pambansa."'
    repairs: list[int] = []
    segments = parse_whisper_jsonl(line, global_offset_seconds=0, repaired_lines=repairs)
    assert repairs == [1]
    assert segments[0].start == 1211.095


def test_atomic_json_write_retries_transient_windows_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "payload.json"
    real_replace = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated watcher lock")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    write_json_atomic(output, {"state": "complete"}, replace_delay_seconds=0)

    assert attempts == 3
    assert json.loads(output.read_text(encoding="utf-8")) == {"state": "complete"}
