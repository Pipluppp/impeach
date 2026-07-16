from __future__ import annotations

import json
from pathlib import Path
from pipeline.discovery import (
    extract_journals,
    match_journals_to_videos,
    normalize_playlist,
    parse_journal_identity,
    register_revision,
    update_manifest,
    write_json_if_changed,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "pipeline" / "fixtures" / "discovery"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_title_parsing_and_updated_marker() -> None:
    identity, updated = parse_journal_identity(
        "VPSD Impeachment Trial - Journal No. 3 (July 7, 2026)",
        "VPSD Journal 03 July 07 2026 (Updated).pdf",
    )
    assert identity.number == 3
    assert identity.session_date == "2026-07-07"
    assert identity.logical_source_id == "journal-03-2026-07-07"
    assert updated is True


def test_feed_and_playlist_match_uniquely_by_date() -> None:
    journals = extract_journals(load("senate_published_journals.json"))
    playlist = normalize_playlist(load("youtube_playlist.json"))
    matches = match_journals_to_videos(journals, playlist)
    assert len(journals) == 6
    assert sum(item["status"] == "matched" for item in matches) == 5
    target = next(item for item in matches if item["session_date"] == "2026-07-14")
    assert target["video"]["video_id"] == "GrQeE6SB1YY"
    assert target["match_confidence"] == 1.0
    first = next(item for item in matches if item["journal_number"] == 1)
    assert first["status"] == "review_required"


def test_manifest_update_is_deterministic_and_deduplicated(tmp_path: Path) -> None:
    journals = extract_journals(load("senate_published_journals.json"))
    playlist = normalize_playlist(load("youtube_playlist.json"))
    matches = match_journals_to_videos(journals, playlist)
    initial = {"schema_version": 1, "sessions": []}
    first = update_manifest(initial, journals, matches)
    second = update_manifest(first, journals, matches)
    assert first == second
    assert len(first["sessions"]) == 6
    path = tmp_path / "manifest.json"
    assert write_json_if_changed(path, first) is True
    assert write_json_if_changed(path, second) is False


def test_changed_pdf_creates_revision_without_overwriting_root(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    original = b"%PDF-1.4\noriginal"
    journal = extract_journals(load("senate_published_journals.json"))[-1]
    revision, changed = register_revision(source_dir, journal, original, "2026-07-15T00:00:00Z")
    assert (revision, changed) == (1, True)
    assert (source_dir / "journal.pdf").read_bytes() == original

    replacement = b"%PDF-1.4\nupdated"
    revision, changed = register_revision(source_dir, journal, replacement, "2026-07-16T00:00:00Z")
    assert (revision, changed) == (2, True)
    assert (source_dir / "journal.pdf").read_bytes() == original
    assert (source_dir / "revisions" / "2" / "journal.pdf").read_bytes() == replacement
    assert register_revision(source_dir, journal, replacement, "2026-07-17T00:00:00Z") == (2, False)
