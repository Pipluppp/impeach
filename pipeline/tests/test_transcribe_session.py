from __future__ import annotations

import pytest

from pipeline.transcribe_session import merge_session_chunks, plan_chunks


def test_chunk_plan_has_overlap_but_contiguous_ownership() -> None:
    plans = plan_chunks(duration=3700, chunk_duration=1800, overlap=15)
    assert [(item.core_start, item.core_end) for item in plans] == [
        (0, 1800), (1800, 3600), (3600, 3700)
    ]
    assert [(item.input_start, item.input_end) for item in plans] == [
        (0, 1815), (1785, 3615), (3585, 3700)
    ]


@pytest.mark.parametrize(
    "duration,chunk,overlap",
    [(0, 10, 1), (10, 0, 1), (10, 10, -1), (10, 10, 5)],
)
def test_invalid_chunk_plan_fails(duration: float, chunk: float, overlap: float) -> None:
    with pytest.raises(ValueError):
        plan_chunks(duration, chunk, overlap)


def test_session_merge_preserves_asr_quarantine_totals() -> None:
    plans = plan_chunks(duration=120, chunk_duration=120, overlap=5)
    payload = merge_session_chunks(
        chunk_payloads=[{
            "configuration": {
                "engine": "whisper.cpp",
                "engine_version": "fixture",
                "model_file": "model.bin",
                "model_sha256": "sha256",
                "input_sha256": "audio-sha256",
                "window_seconds": 120,
                "window_overlap_seconds": 5,
                "fallback_window_seconds": 30,
                "max_context_tokens": 0,
            },
            "runtime": {
                "elapsed_seconds": 10,
                "peak_rss_bytes": 0,
                "ffmpeg_version": "fixture",
                "whisper_cpp_version": "fixture",
                "producer_json_repaired_lines": [],
                "quality": {
                    "primary_window_count": 1,
                    "retried_window_count": 1,
                    "quarantined_window_count": 1,
                    "quarantined_seconds": 60,
                    "dropped_low_information_seconds": 75,
                },
            },
            "segments": [{
                "id": "asr-old",
                "start": 0,
                "end": 10,
                "text": "Session resumed.",
                "normalized_text": "session resumed",
            }],
        }],
        plans=plans,
        session_id="fixture-session",
        video_id="fixture-video",
        duration=120,
        input_origin=0,
        chunk_duration=120,
        overlap=5,
        language="auto",
    )

    quality = payload["runtime"]["quality"]
    assert quality["quarantined_window_count"] == 1
    assert quality["quarantined_seconds"] == 60
    assert quality["dropped_low_information_seconds"] == 75
