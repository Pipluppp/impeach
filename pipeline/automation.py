"""Prepare, merge, and finalize one reproducible synchronized-session run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pipeline.alignment import align
    from pipeline.build_public_data import build_payload, validate_payload
    from pipeline.discovery import register_revision, write_json_if_changed
    from pipeline.official_source import OfficialSenateSource
    from pipeline.parse_journal import parse_document, write_text_if_changed
    from pipeline.transcribe import write_json_atomic
    from pipeline.transcribe_session import merge_session_chunks, plan_chunks
except ModuleNotFoundError:
    from alignment import align
    from build_public_data import build_payload, validate_payload
    from discovery import register_revision, write_json_if_changed
    from official_source import OfficialSenateSource
    from parse_journal import parse_document, write_text_if_changed
    from transcribe import write_json_atomic
    from transcribe_session import merge_session_chunks, plan_chunks


ROOT = Path(__file__).resolve().parents[1]
CHUNK_DURATION_SECONDS = 1800.0
CHUNK_OVERLAP_SECONDS = 15.0


class AutomationError(RuntimeError):
    """The run cannot safely advance while preserving product invariants."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _current_source(source_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    current = load_json(source_dir / "current.json")
    metadata_path = source_dir / current["metadata_path"]
    pdf_path = source_dir / current["pdf_path"]
    return metadata_path, pdf_path, load_json(metadata_path)


def _records(root: Path, session_date: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    journals = load_json(root / "data" / "discovery" / "senate-journals.json")["journals"]
    matches = load_json(root / "data" / "discovery" / "matches.json")["matches"]
    playlist = load_json(root / "data" / "discovery" / "youtube-playlist.json")
    journal = next((item for item in journals if item["session_date"] == session_date), None)
    match = next((item for item in matches if item["session_date"] == session_date), None)
    if not journal or not match:
        raise AutomationError(f"session {session_date} is absent from discovery outputs")
    if match["status"] != "matched" or not match.get("video"):
        raise AutomationError(f"session {session_date} has no unique official-playlist match")
    return journal, match, playlist


def pending_sessions(root: Path) -> list[str]:
    journals = load_json(root / "data" / "discovery" / "senate-journals.json")["journals"]
    matches = load_json(root / "data" / "discovery" / "matches.json")["matches"]
    match_by_id = {item["session_id"]: item for item in matches}
    preserved_dates = [
        journal["session_date"]
        for journal in journals
        if (root / "data" / "sessions" / journal["session_date"] / "source" / "current.json").is_file()
    ]
    latest_preserved = max(preserved_dates, default="")
    pending: list[str] = []
    for journal in journals:
        match = match_by_id.get(journal["session_id"])
        if not match or match["status"] != "matched":
            continue
        source_dir = root / "data" / "sessions" / journal["session_date"] / "source"
        if not (source_dir / "current.json").is_file() and journal["session_date"] > latest_preserved:
            pending.append(journal["session_date"])
            continue
        if not (source_dir / "current.json").is_file():
            continue
        _, _, source = _current_source(source_dir)
        if source.get("source_url") != journal["source_url"]:
            pending.append(journal["session_date"])
    return sorted(set(pending))


def _write_github_outputs(path: Path | None, values: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for key, value in values.items():
            serialized = json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
            handle.write(f"{key}={serialized}\n")


def prepare_session(
    root: Path,
    session_date: str,
    *,
    offline: bool,
    source: OfficialSenateSource | None = None,
    chunk_duration: float = CHUNK_DURATION_SECONDS,
    overlap: float = CHUNK_OVERLAP_SECONDS,
) -> dict[str, Any]:
    journal_record, match, playlist = _records(root, session_date)
    session_dir = root / "data" / "sessions" / session_date
    source_dir = session_dir / "source"
    if not offline:
        content, adapter = (source or OfficialSenateSource()).read_pdf(journal_record["source_url"])
        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        register_revision(source_dir, journal_record, content, retrieved_at)
    elif not (source_dir / "current.json").is_file():
        raise AutomationError("offline preparation requires an already preserved official PDF")
    else:
        adapter = "preserved"

    metadata_path, pdf_path, source_metadata = _current_source(source_dir)
    revision = int(source_metadata["revision"])
    markdown, journal_payload = parse_document(pdf_path, journal_record["journal_number"], revision)
    journal_dir = session_dir / "journal"
    write_text_if_changed(journal_dir / "journal.md", markdown)
    write_text_if_changed(
        journal_dir / "blocks.json",
        json.dumps(journal_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    source_metadata.setdefault("retrieval_method", f"{adapter}_download")
    if source_metadata.get("page_count") != journal_payload["page_count"]:
        source_metadata["page_count"] = journal_payload["page_count"]
    if load_json(metadata_path) != source_metadata:
        write_json_if_changed(metadata_path, source_metadata)

    video = match["video"]
    session_path = session_dir / "session.json"
    existing = load_json(session_path) if session_path.is_file() else {}
    session = {
        "schema_version": 1,
        "session_id": journal_record["session_id"],
        "session_date": session_date,
        "journal_number": journal_record["journal_number"],
        "source_revision": revision,
        "youtube": {
            "playlist_id": playlist.get("playlist_id"),
            "video_id": video["video_id"],
            "title": video["title"],
            "channel": playlist.get("channel", "Senate of the Philippines"),
            "channel_id": playlist.get("channel_id"),
            "watch_url": video["watch_url"],
            "duration_seconds": float(video["duration_seconds"]),
            "match_method": match["match_method"],
            "match_confidence": match["match_confidence"],
        },
        "processing": existing.get("processing", {"state": "prepared"}),
    }
    write_json_if_changed(session_path, session)
    overrides_path = session_dir / "alignment" / "manual-overrides.json"
    if not overrides_path.is_file():
        write_json_if_changed(overrides_path, {
            "schema_version": 1,
            "session_id": session["session_id"],
            "instructions": "Only reviewed session-video ranges belong here; exhibit-video timestamps remain source references.",
            "overrides": [],
        })

    duration = float(video["duration_seconds"])
    plans = plan_chunks(duration, chunk_duration, overlap)
    matrix = {
        "include": [
            {
                "number": f"{plan.number:03d}",
                "start": plan.input_start,
                "duration": plan.input_end - plan.input_start,
                "own_start": plan.core_start,
                "own_end": plan.core_end,
            }
            for plan in plans
        ]
    }
    payload = {
        "schema_version": 1,
        "session_id": session["session_id"],
        "session_date": session_date,
        "video_id": video["video_id"],
        "duration_seconds": duration,
        "source_revision": revision,
        "source_adapter": adapter,
        "chunk_duration_seconds": chunk_duration,
        "overlap_seconds": overlap,
        "matrix": matrix,
    }
    write_json_atomic(root / ".work" / "automation" / session_date / "plan.json", payload)
    return payload


def merge_chunks(root: Path, plan_path: Path, chunks_dir: Path) -> dict[str, Any]:
    plan = load_json(plan_path)
    plans = plan_chunks(
        float(plan["duration_seconds"]),
        float(plan["chunk_duration_seconds"]),
        float(plan["overlap_seconds"]),
    )
    paths = sorted(chunks_dir.glob("chunk-*.json"))
    if len(paths) != len(plans):
        raise AutomationError(f"expected {len(plans)} ASR chunks, found {len(paths)}")
    chunks = [load_json(path) for path in paths]
    payload = merge_session_chunks(
        chunk_payloads=chunks,
        plans=plans,
        session_id=plan["session_id"],
        video_id=plan["video_id"],
        duration=float(plan["duration_seconds"]),
        input_origin=0,
        chunk_duration=float(plan["chunk_duration_seconds"]),
        overlap=float(plan["overlap_seconds"]),
        language="auto",
    )
    output = root / "data" / "sessions" / plan["session_date"] / "transcript" / "segments.json"
    write_json_atomic(output, payload)
    return payload


def processing_plan_outputs(plan_path: Path) -> dict[str, Any]:
    plan = load_json(plan_path)
    matrix = plan["matrix"]
    include = matrix["include"]
    if not include or any("number" not in item for item in include):
        raise AutomationError("prepared processing plan has no valid ASR matrix")
    return {
        "matrix": matrix,
        "session_id": plan["session_id"],
        "video_id": plan["video_id"],
        "duration": plan["duration_seconds"],
        "chunk_count": len(include),
    }


def build_public_index(root: Path) -> dict[str, Any]:
    session_dir = root / "web" / "public" / "data" / "sessions"
    entries = []
    for path in sorted(session_dir.glob("????-??-??.json")):
        payload = load_json(path)
        entries.append({
            "date": payload["session"]["date"],
            "journal_number": payload["session"]["journal_number"],
            "title": payload["session"]["title"],
            "path": f"/data/sessions/{path.name}",
        })
    if not entries:
        raise AutomationError("cannot publish an empty session index")
    payload = {
        "schema_version": 1,
        "latest": max(item["date"] for item in entries),
        "sessions": entries,
    }
    write_json_atomic(session_dir / "index.json", payload)
    return payload


def finalize_session(root: Path, session_date: str) -> dict[str, Any]:
    session_dir = root / "data" / "sessions" / session_date
    session_path = session_dir / "session.json"
    session = load_json(session_path)
    journal = load_json(session_dir / "journal" / "blocks.json")
    transcript = load_json(session_dir / "transcript" / "segments.json")
    overrides = load_json(session_dir / "alignment" / "manual-overrides.json")
    alignment = align(journal, transcript, overrides)
    alignment_path = session_dir / "alignment" / "alignments.json"
    write_json_atomic(alignment_path, alignment)
    _, _, source_metadata = _current_source(session_dir / "source")
    schema = load_json(root / "pipeline" / "schemas" / "public-session.schema.json")
    public_payload = build_payload(
        session=session, source=source_metadata, journal=journal, alignment=alignment
    )
    validate_payload(public_payload, schema)
    public_path = root / "web" / "public" / "data" / "sessions" / f"{session_date}.json"
    write_json_atomic(public_path, public_payload)
    build_public_index(root)

    session.setdefault("processing", {})
    session["processing"].update({
        "state": "automated_review_pending",
        "asr": {
            "engine": transcript["configuration"]["engine"],
            "engine_version": transcript["configuration"]["engine_version"],
            "model": transcript["configuration"]["model_file"],
            "model_sha256": transcript["configuration"]["model_sha256"],
            "segment_count": transcript["segment_count"],
            "time_domain": "session_video",
            "purpose": "internal_alignment_aid",
        },
        "alignment": {
            "name": alignment["aligner"]["name"],
            "version": alignment["aligner"]["version"],
            "timed_blocks": alignment["entry_count"],
            "manual_reviewed": sum(item["review_state"] == "manual_reviewed" for item in alignment["entries"]),
            "needs_review": len(alignment["review_queue"]),
            "unresolved_conflicts": len(alignment["unresolved_conflicts"]),
        },
    })
    write_json_if_changed(session_path, session)
    manifest_path = root / "data" / "manifest.json"
    manifest = load_json(manifest_path)
    item = next(entry for entry in manifest["sessions"] if entry["session_id"] == session["session_id"])
    item.update({
        "current_revision": session["source_revision"],
        "session_data_path": f"sessions/{session_date}/session.json",
        "status": "automated_review_pending",
        "youtube_video_id": session["youtube"]["video_id"],
    })
    write_json_if_changed(manifest_path, manifest)
    return {
        "session_date": session_date,
        "journal_blocks": journal["block_count"],
        "asr_segments": transcript["segment_count"],
        "timed_blocks": alignment["entry_count"],
        "review_queue": len(alignment["review_queue"]),
        "public_path": str(public_path.relative_to(root)).replace("\\", "/"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pending = subparsers.add_parser("pending")
    pending.add_argument("--github-output", type=Path)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--session-date", required=True)
    prepare.add_argument("--offline", action="store_true")
    prepare.add_argument("--github-output", type=Path)
    outputs = subparsers.add_parser("plan-outputs")
    outputs.add_argument("--plan", type=Path, required=True)
    outputs.add_argument("--github-output", type=Path)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--chunks-dir", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--session-date", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "pending":
            dates = pending_sessions(args.root)
            payload: Any = {"dates": dates}
            _write_github_outputs(args.github_output, {"dates": dates, "count": len(dates)})
        elif args.command == "prepare":
            payload = prepare_session(args.root, args.session_date, offline=args.offline)
            _write_github_outputs(args.github_output, {
                "matrix": payload["matrix"], "session_id": payload["session_id"],
                "video_id": payload["video_id"], "duration": payload["duration_seconds"],
                "chunk_count": len(payload["matrix"]["include"]),
            })
        elif args.command == "plan-outputs":
            payload = processing_plan_outputs(args.plan)
            _write_github_outputs(args.github_output, payload)
        elif args.command == "merge":
            payload = {"segments": merge_chunks(args.root, args.plan, args.chunks_dir)["segment_count"]}
        else:
            payload = finalize_session(args.root, args.session_date)
    except (AutomationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"state": "automation_failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
