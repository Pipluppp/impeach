"""Transcribe a full session in deterministic, resumable overlapping chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pipeline.transcribe import (
        TranscriptionError,
        merge_payloads,
        run_chunk,
        write_json_atomic,
    )
except ModuleNotFoundError:  # Direct script execution.
    from transcribe import TranscriptionError, merge_payloads, run_chunk, write_json_atomic


@dataclass(frozen=True)
class ChunkPlan:
    number: int
    core_start: float
    core_end: float
    input_start: float
    input_end: float


def plan_chunks(duration: float, chunk_duration: float, overlap: float) -> list[ChunkPlan]:
    if duration <= 0 or chunk_duration <= 0 or overlap < 0 or overlap >= chunk_duration / 2:
        raise ValueError("invalid duration, chunk duration, or overlap")
    plans = []
    core_start = 0.0
    number = 1
    while core_start < duration:
        core_end = min(duration, core_start + chunk_duration)
        plans.append(
            ChunkPlan(
                number=number,
                core_start=core_start,
                core_end=core_end,
                input_start=max(0.0, core_start - overlap),
                input_end=min(duration, core_end + overlap),
            )
        )
        core_start = core_end
        number += 1
    return plans


def merge_session_chunks(
    *,
    chunk_payloads: list[dict[str, Any]],
    plans: list[ChunkPlan],
    session_id: str,
    video_id: str,
    duration: float,
    input_origin: float,
    chunk_duration: float,
    overlap: float,
    language: str,
) -> dict[str, Any]:
    if len(chunk_payloads) != len(plans) or not chunk_payloads:
        raise TranscriptionError("chunk payload count does not match the session plan")
    segments = merge_payloads(chunk_payloads)
    if not segments or segments[0]["start"] < input_origin:
        raise TranscriptionError("merged transcript falls outside the session time domain")
    if segments[-1]["end"] > input_origin + duration + 30:
        raise TranscriptionError("merged transcript extends beyond the allowed audio tail")
    first_configuration = chunk_payloads[0]["configuration"]
    return {
        "schema_version": 1,
        "session_id": session_id,
        "youtube_video_id": video_id,
        "state": "transcribed",
        "time_domain": "session_video",
        "authoritative_public_text": False,
        "purpose": "internal_alignment_aid",
        "configuration": {
            "engine": first_configuration["engine"],
            "engine_version": first_configuration["engine_version"],
            "model_file": first_configuration["model_file"],
            "model_sha256": first_configuration["model_sha256"],
            "language": language,
            "chunk_duration_seconds": chunk_duration,
            "overlap_seconds": overlap,
            "input_sha256": first_configuration["input_sha256"],
        },
        "runtime": {
            "chunk_count": len(plans),
            "audio_seconds": duration,
            "aggregate_elapsed_seconds": round(
                sum(item["runtime"]["elapsed_seconds"] for item in chunk_payloads), 3
            ),
            "peak_chunk_rss_bytes": max(
                item["runtime"].get("peak_rss_bytes", 0) for item in chunk_payloads
            ),
            "ffmpeg_versions": sorted(
                {item["runtime"]["ffmpeg_version"] for item in chunk_payloads}
            ),
            "whisper_cpp_versions": sorted(
                {
                    item["runtime"]["whisper_cpp_version"]
                    for item in chunk_payloads
                    if item["runtime"].get("whisper_cpp_version")
                }
            ),
            "producer_json_repairs": [
                {"chunk": index, "line_numbers": item["runtime"]["producer_json_repaired_lines"]}
                for index, item in enumerate(chunk_payloads, 1)
                if item["runtime"].get("producer_json_repaired_lines")
            ],
            "recovered_existing_raw_chunks": [
                index
                for index, item in enumerate(chunk_payloads, 1)
                if item["runtime"].get("recovered_existing_raw")
            ],
        },
        "segment_count": len(segments),
        "segments": segments,
    }


def transcribe_session(
    *,
    input_path: Path,
    model_path: Path,
    output_path: Path,
    work_dir: Path,
    session_id: str,
    video_id: str,
    duration: float,
    input_origin: float,
    chunk_duration: float,
    overlap: float,
    language: str,
    queue: float,
) -> dict[str, Any]:
    plans = plan_chunks(duration, chunk_duration, overlap)
    work_dir.mkdir(parents=True, exist_ok=True)
    status_path = work_dir / "status.json"
    chunk_payloads = []
    for plan in plans:
        chunk_path = work_dir / f"chunk-{plan.number:03d}.json"
        try:
            payload = run_chunk(
                input_path=input_path,
                model_path=model_path,
                output_path=chunk_path,
                input_origin_seconds=input_origin,
                chunk_start_seconds=plan.input_start,
                duration_seconds=plan.input_end - plan.input_start,
                language=language,
                queue_seconds=queue,
                own_start=None if plan.number == 1 else input_origin + plan.core_start,
                own_end=None if plan.number == len(plans) else input_origin + plan.core_end,
            )
        except (OSError, TranscriptionError) as exc:
            write_json_atomic(
                status_path,
                {
                    "state": "asr_failed",
                    "resumable": True,
                    "session_id": session_id,
                    "failed_chunk": plan.number,
                    "completed_chunks": len(chunk_payloads),
                    "chunk_count": len(plans),
                    "error": str(exc),
                },
            )
            raise
        chunk_payloads.append(payload)
        write_json_atomic(
            status_path,
            {
                "state": "transcribing" if plan.number < len(plans) else "merging",
                "resumable": True,
                "session_id": session_id,
                "completed_chunks": plan.number,
                "chunk_count": len(plans),
            },
        )
        disposition = "reused" if payload.get("reused") else "completed"
        print(f"chunk {plan.number}/{len(plans)} {disposition}", flush=True)
    payload = merge_session_chunks(
        chunk_payloads=chunk_payloads, plans=plans, session_id=session_id,
        video_id=video_id, duration=duration, input_origin=input_origin,
        chunk_duration=chunk_duration, overlap=overlap, language=language,
    )
    write_json_atomic(output_path, payload)
    write_json_atomic(
        status_path,
        {
            "state": "transcribed",
            "resumable": True,
            "session_id": session_id,
            "completed_chunks": len(plans),
            "chunk_count": len(plans),
            "output": str(output_path).replace("\\", "/"),
        },
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--input-origin", type=float, default=0)
    parser.add_argument("--chunk-duration", type=float, default=1800)
    parser.add_argument("--overlap", type=float, default=15)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--queue", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        payload = transcribe_session(
            input_path=args.input, model_path=args.model, output_path=args.output,
            work_dir=args.work_dir, session_id=args.session_id, video_id=args.video_id,
            duration=args.duration, input_origin=args.input_origin,
            chunk_duration=args.chunk_duration, overlap=args.overlap,
            language=args.language, queue=args.queue,
        )
    except (OSError, TranscriptionError, ValueError) as exc:
        print(json.dumps({"state": "asr_failed", "resumable": True, "error": str(exc)}))
        return 2
    print(json.dumps({"state": payload["state"], "segments": payload["segment_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
