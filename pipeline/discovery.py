"""Bounded discovery and revision tracking for official journals and videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from pipeline.official_source import (
        OfficialSenateSource,
        OfficialSourceUnavailable,
        SENATE_LISTING_URL,
    )
except ModuleNotFoundError:
    from official_source import (
        OfficialSenateSource,
        OfficialSourceUnavailable,
        SENATE_LISTING_URL,
    )


ROOT = Path(__file__).resolve().parents[1]
SENATE_FILE_BASE = "https://senate.gov.ph/hq/"
PLAYLIST_ID = "PLY4L49cjE9ow"
PLAYLIST_URL = f"https://www.youtube.com/playlist?list={PLAYLIST_ID}"

JOURNAL_RE = re.compile(
    r"Journal\s+No\.\s*0*(?P<number>\d+)\s*\(\s*"
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})\s*,?\s*(?P<year>\d{4})\s*\)",
    re.IGNORECASE,
)
VIDEO_DATE_RE = re.compile(
    r"\((?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})\)\s*$"
)


class DiscoveryUnavailable(RuntimeError):
    """A bounded live source fetch failed without changing prior output."""


@dataclass(frozen=True)
class JournalIdentity:
    number: int
    session_date: str

    @property
    def session_id(self) -> str:
        return f"impeachment-trial-{self.number:02d}"

    @property
    def logical_source_id(self) -> str:
        return f"journal-{self.number:02d}-{self.session_date}"


def parse_date(month: str, day: str, year: str) -> str:
    parsed = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
    return parsed.date().isoformat()


def parse_journal_identity(label: str, filename: str = "") -> tuple[JournalIdentity, bool]:
    match = JOURNAL_RE.search(label)
    if not match:
        raise ValueError(f"unrecognized journal title: {label!r}")
    identity = JournalIdentity(
        number=int(match.group("number")),
        session_date=parse_date(match.group("month"), match.group("day"), match.group("year")),
    )
    updated = "updated" in f"{label} {filename}".casefold()
    return identity, updated


def parse_video_date(title: str) -> str | None:
    match = VIDEO_DATE_RE.search(title)
    if not match:
        return None
    return parse_date(match.group("month"), match.group("day"), match.group("year"))


def absolute_senate_url(relative_url: str) -> str:
    return SENATE_FILE_BASE + relative_url.lstrip("/")


def extract_journals(feed: dict[str, Any]) -> list[dict[str, Any]]:
    journal_party = next(
        (party for party in feed.get("parties", []) if party.get("slug") == "journal"),
        None,
    )
    if journal_party is None:
        raise ValueError("official feed has no journal party")

    records: list[dict[str, Any]] = []
    for section in journal_party.get("sections", []):
        for document in section.get("documents", []):
            identity, updated = parse_journal_identity(
                document.get("label", ""), document.get("filename", "")
            )
            records.append(
                {
                    "logical_source_id": identity.logical_source_id,
                    "session_id": identity.session_id,
                    "session_date": identity.session_date,
                    "journal_number": identity.number,
                    "display_title": document["label"],
                    "filename": document.get("filename"),
                    "source_url": absolute_senate_url(document["url"]),
                    "listing_url": SENATE_LISTING_URL,
                    "feed_document_id": document.get("id"),
                    "updated_marker": updated,
                }
            )
    return sorted(records, key=lambda item: (item["session_date"], item["journal_number"]))


def normalize_playlist(playlist: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for entry in playlist.get("entries", []):
        session_date = parse_video_date(entry.get("title", ""))
        entries.append(
            {
                "video_id": entry.get("id"),
                "title": entry.get("title"),
                "duration_seconds": entry.get("duration"),
                "session_date": session_date,
                "watch_url": f"https://www.youtube.com/watch?v={entry.get('id')}",
            }
        )
    return {
        "playlist_id": playlist.get("id", PLAYLIST_ID),
        "title": playlist.get("title"),
        "modified_date": playlist.get("modified_date"),
        "channel": playlist.get("channel", "Senate of the Philippines"),
        "channel_id": playlist.get("channel_id", "UCPJWi44tskVVMNyKzYrPoGg"),
        "entries": sorted(entries, key=lambda item: (item["session_date"] or "", item["video_id"] or "")),
    }


def match_journals_to_videos(
    journals: list[dict[str, Any]], playlist: dict[str, Any]
) -> list[dict[str, Any]]:
    matches = []
    for journal in journals:
        candidates = [
            entry
            for entry in playlist["entries"]
            if entry["session_date"] == journal["session_date"]
        ]
        if len(candidates) == 1:
            status = "matched"
            confidence = 1.0
            video = candidates[0]
        else:
            status = "review_required"
            confidence = 0.0
            video = None
        matches.append(
            {
                "session_id": journal["session_id"],
                "session_date": journal["session_date"],
                "journal_number": journal["journal_number"],
                "status": status,
                "match_method": "official_playlist_title_date_unique",
                "match_confidence": confidence,
                "video": video,
                "candidate_video_ids": [item["video_id"] for item in candidates],
            }
        )
    return matches


def fetch_playlist() -> dict[str, Any]:
    command = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--no-warnings",
        "--retries",
        "2",
        "--extractor-retries",
        "2",
        "--retry-sleep",
        "linear=1::2",
        "--sleep-requests",
        "0.25",
        PLAYLIST_URL,
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=60
        )
        return json.loads(completed.stdout)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise DiscoveryUnavailable(f"playlist discovery failed: {exc}") from exc


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json_if_changed(path: Path, data: Any) -> bool:
    content = canonical_json(data)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def update_manifest(
    manifest: dict[str, Any],
    journals: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    existing = {item["session_id"]: dict(item) for item in manifest.get("sessions", [])}
    match_by_id = {item["session_id"]: item for item in matches}
    for journal in journals:
        session_id = journal["session_id"]
        item = existing.get(session_id, {})
        item.update(
            {
                "session_id": session_id,
                "session_date": journal["session_date"],
                "journal_number": journal["journal_number"],
                "source_url": journal["source_url"],
            }
        )
        item.setdefault("current_revision", 0)
        match = match_by_id[session_id]
        # Discovery may refine an unresolved item, but it must never move an
        # already processed civic record backwards in the state machine.
        if item.get("status") in {None, "discovered", "review_required"}:
            item["status"] = "discovered" if match["status"] == "matched" else "review_required"
        if match["video"]:
            item["youtube_video_id"] = match["video"]["video_id"]
        existing[session_id] = item
    return {
        "schema_version": manifest.get("schema_version", 1),
        "sessions": sorted(
            existing.values(), key=lambda item: (item["session_date"], item["journal_number"])
        ),
    }


def known_revisions(source_dir: Path) -> list[dict[str, Any]]:
    paths = []
    root_metadata = source_dir / "source.json"
    if root_metadata.is_file():
        paths.append(root_metadata)
    revision_dir = source_dir / "revisions"
    if revision_dir.is_dir():
        paths.extend(revision_dir.glob("*/source.json"))
    return sorted((json.loads(path.read_text(encoding="utf-8")) for path in paths), key=lambda x: x["revision"])


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def register_revision(
    source_dir: Path,
    journal: dict[str, Any],
    pdf_content: bytes,
    retrieved_at: str,
) -> tuple[int, bool]:
    if not pdf_content.startswith(b"%PDF-"):
        raise ValueError("downloaded source lacks a PDF signature")
    digest = sha256_bytes(pdf_content)
    revisions = known_revisions(source_dir)
    for metadata in revisions:
        if metadata["sha256"] == digest:
            return metadata["revision"], False

    revision = max((item["revision"] for item in revisions), default=0) + 1
    target_dir = source_dir if revision == 1 and not revisions else source_dir / "revisions" / str(revision)
    if target_dir == source_dir:
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "logical_source_id": journal["logical_source_id"],
        "revision": revision,
        "display_title": journal["display_title"],
        "source_url": journal["source_url"],
        "listing_url": journal["listing_url"],
        "retrieved_at": retrieved_at,
        "content_type": "application/pdf",
        "byte_length": len(pdf_content),
        "sha256": digest,
        "supersedes_revision": revision - 1 or None,
    }
    (target_dir / "journal.pdf").write_bytes(pdf_content)
    write_json_if_changed(target_dir / "source.json", metadata)
    (target_dir / "source.sha256").write_text(f"{digest}  journal.pdf\n", encoding="utf-8")
    write_json_if_changed(
        source_dir / "current.json",
        {
            "logical_source_id": journal["logical_source_id"],
            "current_revision": revision,
            "metadata_path": str((target_dir / "source.json").relative_to(source_dir)).replace("\\", "/"),
            "pdf_path": str((target_dir / "journal.pdf").relative_to(source_dir)).replace("\\", "/"),
        },
    )
    return revision, True


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> int:
    try:
        senate_feed = (
            load_json(args.senate_feed)
            if args.senate_feed
            else OfficialSenateSource().read_feed()[0]
        )
        playlist_feed = load_json(args.playlist_feed) if args.playlist_feed else fetch_playlist()
    except (DiscoveryUnavailable, OfficialSourceUnavailable) as exc:
        print(f"DISCOVERY_UNAVAILABLE: {exc}", file=sys.stderr)
        return 2

    journals = extract_journals(senate_feed)
    playlist = normalize_playlist(playlist_feed)
    matches = match_journals_to_videos(journals, playlist)
    manifest = load_json(args.manifest) if args.manifest.is_file() else {"schema_version": 1, "sessions": []}
    next_manifest = update_manifest(manifest, journals, matches)

    summary = {
        "journals": len(journals),
        "videos": len(playlist["entries"]),
        "matched": sum(item["status"] == "matched" for item in matches),
        "review_required": sum(item["status"] == "review_required" for item in matches),
    }
    if args.dry_run:
        print(canonical_json(summary), end="")
        return 0

    changed = [
        write_json_if_changed(args.output_dir / "senate-journals.json", {"schema_version": 1, "journals": journals}),
        write_json_if_changed(args.output_dir / "youtube-playlist.json", {"schema_version": 1, **playlist}),
        write_json_if_changed(args.output_dir / "matches.json", {"schema_version": 1, "matches": matches}),
        write_json_if_changed(args.manifest, next_manifest),
    ]
    summary["changed_files"] = sum(changed)
    print(canonical_json(summary), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--senate-feed", type=Path, help="captured official feed JSON")
    parser.add_argument("--playlist-feed", type=Path, help="captured yt-dlp playlist JSON")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "discovery")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
