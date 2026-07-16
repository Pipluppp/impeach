from __future__ import annotations

import json
from pathlib import Path

from pipeline.automation import (
    build_public_index,
    merge_chunks,
    pending_sessions,
    processing_plan_outputs,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pending_selects_newer_sessions_and_revisions_without_historical_backfill(
    tmp_path: Path,
) -> None:
    journals = [
        {"session_id": "s1", "session_date": "2026-07-13", "source_url": "old"},
        {"session_id": "s2", "session_date": "2026-07-14", "source_url": "changed"},
        {"session_id": "s3", "session_date": "2026-07-15", "source_url": "new"},
    ]
    matches = [
        {"session_id": item["session_id"], "status": "matched"} for item in journals
    ]
    write_json(tmp_path / "data/discovery/senate-journals.json", {"journals": journals})
    write_json(tmp_path / "data/discovery/matches.json", {"matches": matches})
    write_json(
        tmp_path / "data/sessions/2026-07-14/source/current.json",
        {"metadata_path": "source.json", "pdf_path": "journal.pdf"},
    )
    write_json(
        tmp_path / "data/sessions/2026-07-14/source/source.json",
        {"source_url": "original"},
    )
    assert pending_sessions(tmp_path) == ["2026-07-14", "2026-07-15"]


def test_parallel_chunks_merge_into_full_session_payload(tmp_path: Path) -> None:
    duration = 20.0
    plan = {
        "session_id": "session-1",
        "session_date": "2026-07-15",
        "video_id": "video-1",
        "duration_seconds": duration,
        "chunk_duration_seconds": 10.0,
        "overlap_seconds": 1.0,
    }
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    chunks_dir = tmp_path / "chunks"
    for index, segment in enumerate(((1.0, 9.0), (10.0, 19.0)), 1):
        write_json(chunks_dir / f"chunk-{index:03d}.json", {
            "configuration": {
                "engine": "whisper.cpp", "engine_version": "v1.8.6",
                "model_file": "ggml-base.bin", "model_sha256": "model",
                "input_sha256": "audio",
            },
            "runtime": {
                "elapsed_seconds": 1.0, "peak_rss_bytes": 0,
                "ffmpeg_version": "ffmpeg", "whisper_cpp_version": "v1.8.6",
                "producer_json_repaired_lines": [], "recovered_existing_raw": False,
            },
            "segments": [{
                "id": "local", "start": segment[0], "end": segment[1],
                "time_domain": "session_video", "text": f"part {index}",
                "normalized_text": f"part {index}",
            }],
        })
    payload = merge_chunks(tmp_path, plan_path, chunks_dir)
    assert payload["session_id"] == "session-1"
    assert payload["segment_count"] == 2
    assert [item["id"] for item in payload["segments"]] == ["asr-000001", "asr-000002"]
    assert payload["runtime"]["whisper_cpp_versions"] == ["v1.8.6"]


def test_prepared_plan_can_rehydrate_matrix_outputs(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, {
        "session_id": "session-1",
        "video_id": "video-1",
        "duration_seconds": 20,
        "matrix": {"include": [{"number": "001"}, {"number": "002"}]},
    })
    assert processing_plan_outputs(plan_path) == {
        "session_id": "session-1",
        "video_id": "video-1",
        "duration": 20,
        "chunk_count": 2,
        "matrix": {"include": [{"number": "001"}, {"number": "002"}]},
    }


def test_public_index_selects_latest_generated_session(tmp_path: Path) -> None:
    for date, journal_number in (("2026-07-14", 6), ("2026-07-15", 7)):
        write_json(tmp_path / f"web/public/data/sessions/{date}.json", {
            "session": {"date": date, "journal_number": journal_number, "title": f"Session {journal_number}"}
        })
    payload = build_public_index(tmp_path)
    assert payload["latest"] == "2026-07-15"
    assert payload["sessions"][1]["path"] == "/data/sessions/2026-07-15.json"
