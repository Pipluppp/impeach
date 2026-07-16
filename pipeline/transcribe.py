"""Run resumable FFmpeg/whisper.cpp chunks and merge compact session-time ASR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psutil


ROOT = Path(__file__).resolve().parents[1]
ENGINE_NAME = "ffmpeg-whisper"
ENGINE_VERSION = "0.1.0"
WHISPER_CPP_ENGINE = "whisper.cpp"


class TranscriptionError(RuntimeError):
    """A chunk failed without publishing a partial transcript."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    normalized_text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def normalize_for_matching(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def repair_ffmpeg_whisper_line(line: str) -> dict[str, Any]:
    """Repair FFmpeg whisper JSON when decoded speech contains unescaped quotes."""
    match = re.fullmatch(
        r'\{"start":(?P<start>-?\d+),"end":(?P<end>-?\d+),"text":(?P<text>.*)\}',
        line,
    )
    if not match:
        raise ValueError("line does not match the known FFmpeg whisper object shape")
    literal = match.group("text")
    if len(literal) < 2 or literal[0] != '"' or literal[-1] != '"':
        raise ValueError("text field is not a quoted string")
    inner = literal[1:-1]
    repaired = re.sub(r'(?<!\\)"', r'\\"', inner)
    return {
        "start": int(match.group("start")),
        "end": int(match.group("end")),
        "text": json.loads(f'"{repaired}"'),
    }


def parse_whisper_jsonl(
    text: str, *, global_offset_seconds: float, repaired_lines: list[int] | None = None
) -> list[Segment]:
    segments: list[Segment] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = repair_ffmpeg_whisper_line(line)
                if repaired_lines is not None:
                    repaired_lines.append(line_number)
            start = global_offset_seconds + float(item["start"]) / 1000.0
            end = global_offset_seconds + float(item["end"]) / 1000.0
            raw_text = str(item["text"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(f"invalid whisper JSONL on line {line_number}: {exc}") from exc
        if end < start or (segments and start < segments[-1].start):
            raise TranscriptionError("whisper timestamps are not monotonic")
        if raw_text:
            segments.append(Segment(start, end, raw_text, normalize_for_matching(raw_text)))
    if not segments:
        raise TranscriptionError("whisper produced no non-empty segments")
    return segments


def parse_whisper_cpp_json(text: str, *, global_offset_seconds: float) -> list[Segment]:
    try:
        payload = json.loads(text)
        transcription = payload["transcription"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TranscriptionError(f"invalid whisper.cpp JSON: {exc}") from exc
    segments: list[Segment] = []
    for index, item in enumerate(transcription, 1):
        try:
            offsets = item["offsets"]
            start = global_offset_seconds + float(offsets["from"]) / 1000.0
            end = global_offset_seconds + float(offsets["to"]) / 1000.0
            raw_text = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(f"invalid whisper.cpp segment {index}: {exc}") from exc
        if end < start or (segments and start < segments[-1].start):
            raise TranscriptionError("whisper.cpp timestamps are not monotonic")
        if raw_text:
            segments.append(Segment(start, end, raw_text, normalize_for_matching(raw_text)))
    if not segments:
        raise TranscriptionError("whisper.cpp produced no non-empty segments")
    return segments


def owned_segments(
    segments: Iterable[Segment], *, own_start: float | None, own_end: float | None
) -> list[Segment]:
    result: list[Segment] = []
    for segment in segments:
        midpoint = (segment.start + segment.end) / 2
        if own_start is not None and midpoint < own_start:
            continue
        if own_end is not None and midpoint >= own_end:
            continue
        result.append(segment)
    return result


def compact_segments(segments: Iterable[Segment]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"asr-{index:06d}",
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "time_domain": "session_video",
            "text": segment.text,
            "normalized_text": segment.normalized_text,
        }
        for index, segment in enumerate(segments, 1)
    ]


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    replace_attempts: int = 20,
    replace_delay_seconds: float = 0.1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for attempt in range(replace_attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt + 1 == replace_attempts:
                raise
            time.sleep(replace_delay_seconds)


def ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, timeout=15, check=True
    )
    return completed.stdout.splitlines()[0]


def run_chunk(
    *,
    input_path: Path,
    model_path: Path,
    output_path: Path,
    input_origin_seconds: float,
    chunk_start_seconds: float,
    duration_seconds: float,
    language: str,
    queue_seconds: float,
    own_start: float | None = None,
    own_end: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if duration_seconds <= 0 or chunk_start_seconds < 0:
        raise TranscriptionError("chunk start and duration must be non-negative")
    input_hash = sha256_file(input_path)
    model_hash = sha256_file(model_path)
    configuration = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "model_file": model_path.name,
        "model_sha256": model_hash,
        "language": language,
        "queue_seconds": queue_seconds,
        "use_gpu": False,
        "input_sha256": input_hash,
        "input_origin_seconds": input_origin_seconds,
        "chunk_start_seconds": chunk_start_seconds,
        "duration_seconds": duration_seconds,
        "own_start": own_start,
        "own_end": own_end,
    }
    fingerprint = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if output_path.is_file() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("configuration_fingerprint") == fingerprint:
            existing["reused"] = True
            return existing

    cache_dir = ROOT / ".cache" / "asr-work"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / f"{fingerprint}.jsonl"
    log_path = cache_dir / f"{fingerprint}.log"
    model_arg = model_path.resolve().relative_to(ROOT.resolve()).as_posix()
    raw_arg = raw_path.resolve().relative_to(ROOT.resolve()).as_posix()
    filter_value = (
        "aformat=sample_rates=16000:channel_layouts=mono,"
        f"whisper=model={model_arg}:language={language}:queue={queue_seconds:g}:"
        f"use_gpu=false:destination={raw_arg}:format=json"
    )
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", f"{chunk_start_seconds:g}", "-i", str(input_path),
        "-t", f"{duration_seconds:g}", "-af", filter_value,
        "-f", "null", os.devnull,
    ]
    global_offset = input_origin_seconds + chunk_start_seconds
    all_segments: list[Segment] | None = None
    repaired_lines: list[int] = []
    recovered_existing_raw = False
    elapsed = 0.0
    peak_rss = 0
    if raw_path.is_file() and not force:
        try:
            recovered = parse_whisper_jsonl(
                raw_path.read_text(encoding="utf-8"),
                global_offset_seconds=global_offset,
                repaired_lines=repaired_lines,
            )
            minimum_complete_end = global_offset + duration_seconds - queue_seconds - 2
            if recovered[-1].end >= minimum_complete_end:
                all_segments = recovered
                stat = raw_path.stat()
                elapsed = max(0.0, stat.st_mtime - stat.st_ctime)
                recovered_existing_raw = True
        except TranscriptionError:
            pass
    if all_segments is None:
        raw_path.unlink(missing_ok=True)
        repaired_lines.clear()
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak_rss = max(peak_rss, monitored.memory_info().rss)
                except psutil.Error:
                    pass
                time.sleep(0.1)
            return_code = process.wait()
        elapsed = time.perf_counter() - started
        if return_code != 0 or not raw_path.is_file():
            raise TranscriptionError(
                f"FFmpeg whisper failed with exit {return_code}; see {log_path.relative_to(ROOT)}"
            )
        all_segments = parse_whisper_jsonl(
            raw_path.read_text(encoding="utf-8"),
            global_offset_seconds=global_offset,
            repaired_lines=repaired_lines,
        )
    kept_segments = owned_segments(all_segments, own_start=own_start, own_end=own_end)
    if not kept_segments:
        raise TranscriptionError("chunk ownership removed every ASR segment")
    payload = {
        "schema_version": 1,
        "state": "transcribed",
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "runtime": {
            "elapsed_seconds": round(elapsed, 3),
            "audio_seconds": duration_seconds,
            "realtime_factor": round(elapsed / duration_seconds, 4),
            "peak_rss_bytes": peak_rss,
            "peak_rss_observed": not recovered_existing_raw,
            "recovered_existing_raw": recovered_existing_raw,
            "producer_json_repaired_lines": repaired_lines,
            "ffmpeg_version": ffmpeg_version(),
        },
        "segment_count_before_ownership": len(all_segments),
        "segments": compact_segments(kept_segments),
        "reused": False,
    }
    write_json_atomic(output_path, payload)
    return payload


def run_whisper_cpp_chunk(
    *,
    input_path: Path,
    model_path: Path,
    executable_path: Path,
    output_path: Path,
    input_origin_seconds: float,
    chunk_start_seconds: float,
    duration_seconds: float,
    language: str,
    engine_version: str,
    own_start: float | None = None,
    own_end: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if duration_seconds <= 0 or chunk_start_seconds < 0:
        raise TranscriptionError("chunk start and duration must be non-negative")
    configuration = {
        "engine": WHISPER_CPP_ENGINE,
        "engine_version": engine_version,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "language": language,
        "use_gpu": False,
        "input_sha256": sha256_file(input_path),
        "input_origin_seconds": input_origin_seconds,
        "chunk_start_seconds": chunk_start_seconds,
        "duration_seconds": duration_seconds,
        "own_start": own_start,
        "own_end": own_end,
    }
    fingerprint = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if output_path.is_file() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("configuration_fingerprint") == fingerprint:
            existing["reused"] = True
            return existing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = output_path.with_suffix(".wav")
    raw_prefix = output_path.with_suffix("")
    raw_path = raw_prefix.with_suffix(".json")
    ffmpeg_command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", f"{chunk_start_seconds:g}", "-i", str(input_path),
        "-t", f"{duration_seconds:g}", "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", str(wav_path),
    ]
    whisper_command = [
        str(executable_path), "-m", str(model_path), "-f", str(wav_path),
        "-l", language, "-oj", "-of", str(raw_prefix), "-np", "-ng",
    ]
    started = time.perf_counter()
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, timeout=900)
        completed = subprocess.run(
            whisper_command, check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=21000,
        )
        if not raw_path.is_file():
            raise TranscriptionError(
                f"whisper.cpp produced no JSON; stderr: {completed.stderr[-1000:]}"
            )
        all_segments = parse_whisper_cpp_json(
            raw_path.read_text(encoding="utf-8"),
            global_offset_seconds=input_origin_seconds + chunk_start_seconds,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"")[-1000:]
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise TranscriptionError(f"chunk command failed with exit {exc.returncode}: {stderr}") from exc
    finally:
        wav_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    kept_segments = owned_segments(all_segments, own_start=own_start, own_end=own_end)
    if not kept_segments:
        raise TranscriptionError("chunk ownership removed every ASR segment")
    payload = {
        "schema_version": 1,
        "state": "transcribed",
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "runtime": {
            "elapsed_seconds": round(elapsed, 3),
            "audio_seconds": duration_seconds,
            "realtime_factor": round(elapsed / duration_seconds, 4),
            "peak_rss_bytes": 0,
            "peak_rss_observed": False,
            "recovered_existing_raw": False,
            "producer_json_repaired_lines": [],
            "ffmpeg_version": ffmpeg_version(),
            "whisper_cpp_version": engine_version,
        },
        "segment_count_before_ownership": len(all_segments),
        "segments": compact_segments(kept_segments),
        "reused": False,
    }
    write_json_atomic(output_path, payload)
    return payload


def merge_payloads(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = [segment for payload in payloads for segment in payload["segments"]]
    segments.sort(key=lambda item: (item["start"], item["end"]))
    previous_start = -1.0
    for segment in segments:
        if segment["start"] < previous_start or segment["end"] < segment["start"]:
            raise TranscriptionError("merged transcript is not monotonic")
        previous_start = segment["start"]
    for index, segment in enumerate(segments, 1):
        segment["id"] = f"asr-{index:06d}"
    return segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    chunk = subparsers.add_parser("chunk")
    chunk.add_argument("--input", type=Path, required=True)
    chunk.add_argument("--model", type=Path, required=True)
    chunk.add_argument("--output", type=Path, required=True)
    chunk.add_argument("--input-origin", type=float, default=0)
    chunk.add_argument("--start", type=float, default=0)
    chunk.add_argument("--duration", type=float, required=True)
    chunk.add_argument("--language", default="auto")
    chunk.add_argument("--queue", type=float, default=30.0)
    chunk.add_argument("--own-start", type=float)
    chunk.add_argument("--own-end", type=float)
    chunk.add_argument("--force", action="store_true")
    cpp_chunk = subparsers.add_parser("whisper-cpp-chunk")
    cpp_chunk.add_argument("--input", type=Path, required=True)
    cpp_chunk.add_argument("--model", type=Path, required=True)
    cpp_chunk.add_argument("--executable", type=Path, required=True)
    cpp_chunk.add_argument("--output", type=Path, required=True)
    cpp_chunk.add_argument("--input-origin", type=float, default=0)
    cpp_chunk.add_argument("--start", type=float, default=0)
    cpp_chunk.add_argument("--duration", type=float, required=True)
    cpp_chunk.add_argument("--language", default="auto")
    cpp_chunk.add_argument("--engine-version", required=True)
    cpp_chunk.add_argument("--own-start", type=float)
    cpp_chunk.add_argument("--own-end", type=float)
    cpp_chunk.add_argument("--force", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "chunk":
            payload = run_chunk(
                input_path=args.input, model_path=args.model, output_path=args.output,
                input_origin_seconds=args.input_origin, chunk_start_seconds=args.start,
                duration_seconds=args.duration, language=args.language, queue_seconds=args.queue,
                own_start=args.own_start, own_end=args.own_end, force=args.force,
            )
        elif args.command == "whisper-cpp-chunk":
            payload = run_whisper_cpp_chunk(
                input_path=args.input, model_path=args.model,
                executable_path=args.executable, output_path=args.output,
                input_origin_seconds=args.input_origin, chunk_start_seconds=args.start,
                duration_seconds=args.duration, language=args.language,
                engine_version=args.engine_version, own_start=args.own_start,
                own_end=args.own_end, force=args.force,
            )
        else:
            inputs = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
            payload = {
                "schema_version": 1,
                "state": "transcribed",
                "time_domain": "session_video",
                "source_chunks": [str(path).replace("\\", "/") for path in args.input],
                "segments": merge_payloads(inputs),
            }
            write_json_atomic(args.output, payload)
        print(json.dumps({
            "state": payload["state"],
            "segments": len(payload.get("segments", [])),
            "reused": payload.get("reused", False),
        }, sort_keys=True))
        return 0
    except (OSError, subprocess.SubprocessError, TranscriptionError, json.JSONDecodeError) as exc:
        failure = {"state": "asr_failed", "resumable": True, "error": str(exc)}
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
