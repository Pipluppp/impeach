"""Build and validate the compact browser payload from preserved project artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

try:
    from pipeline.transcribe import write_json_atomic
except ModuleNotFoundError:
    from transcribe import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def timing_for(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "start": entry["start"], "end": entry["end"], "time_domain": "session_video",
        "precision": entry["precision"], "confidence": entry["confidence"],
        "review_state": entry["review_state"],
    }


def build_payload(
    *, session: dict[str, Any], source: dict[str, Any], journal: dict[str, Any],
    alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    timings = {entry["block_id"]: entry for entry in alignment.get("entries", [])} if alignment else {}
    pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sections: dict[str, dict[str, Any]] = {}
    for block in journal["blocks"]:
        public_block = {
            "id": block["id"], "sequence": block["sequence"], "kind": block["kind"],
            "text": block["raw_text"], "speaker": block.get("speaker_raw"),
            "section_id": block.get("section_id"), "role": block.get("role"),
            "timing": timing_for(timings.get(block["id"])),
            "source_time_references": block.get("time_references", []),
        }
        pages[block["page"]].append(public_block)
        section_id = block.get("section_id")
        if section_id and section_id not in sections:
            sections[section_id] = {
                "id": section_id, "title": block.get("section_title") or section_id.replace("-", " ").title(),
                "first_block_id": block["id"], "page": block["page"], "timing": public_block["timing"],
            }
        elif section_id and sections[section_id]["timing"] is None and public_block["timing"] is not None:
            sections[section_id]["timing"] = public_block["timing"]
    youtube = session["youtube"]
    total_blocks = sum(len(page_blocks) for page_blocks in pages.values())
    alignment_summary = {
        "total_blocks": total_blocks,
        "timed_blocks": len(timings),
        "coverage": round(len(timings) / total_blocks, 4) if total_blocks else 0.0,
        "needs_review": sum(
            entry.get("review_state") == "needs_review" for entry in timings.values()
        ),
        "manual_reviewed": sum(
            entry.get("review_state") == "manual_reviewed" for entry in timings.values()
        ),
        "unresolved_conflicts": len(alignment.get("unresolved_conflicts", [])) if alignment else 0,
    }
    return {
        "schema_version": "1.1.0",
        "session": {
            "id": session["session_id"], "date": session["session_date"],
            "journal_number": session["journal_number"], "title": youtube["title"],
        },
        "sources": {
            "journal": {
                "url": source["source_url"], "listing_url": source["listing_url"],
                "retrieved_at": source["retrieved_at"], "sha256": source["sha256"],
                "page_count": source["page_count"],
            },
            "video": {
                "id": youtube["video_id"], "url": youtube["watch_url"],
                "playlist_id": youtube["playlist_id"], "channel": youtube["channel"],
                "duration_seconds": youtube["duration_seconds"],
            },
        },
        "processing": {
            "source_revision": session["source_revision"],
            "parser": f"{journal['parser']['name']} {journal['parser']['version']}",
            "aligner": (
                f"{alignment['aligner']['name']} {alignment['aligner']['version']}" if alignment else None
            ),
            "alignment_summary": alignment_summary,
            "alignment_policy": "Direct monotonic ASR matches are approximate seek targets; unmatched dialogue and journal summaries use bounded contextual ranges that remain noninteractive and are never word-level exact.",
        },
        "outline": list(sections.values()),
        "pages": [{"number": number, "blocks": pages[number]} for number in sorted(pages)],
    }


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        details = "; ".join(f"{'/'.join(map(str, error.absolute_path))}: {error.message}" for error in errors[:10])
        raise ValueError(f"public session schema validation failed: {details}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "pipeline" / "schemas" / "public-session.schema.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    alignment = load_json(args.alignment) if args.alignment and args.alignment.is_file() else None
    payload = build_payload(
        session=load_json(args.session), source=load_json(args.source),
        journal=load_json(args.journal), alignment=alignment,
    )
    validate_payload(payload, load_json(args.schema))
    write_json_atomic(args.output, payload)
    print(json.dumps({"pages": len(payload["pages"]), "blocks": sum(len(page["blocks"]) for page in payload["pages"]), "bytes": args.output.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
