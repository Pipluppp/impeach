from __future__ import annotations

import json
from pathlib import Path

from pipeline.source_intake import plan_intake


def write_current_source(root: Path, date: str, source_url: str) -> None:
    source_dir = root / "data" / "sessions" / date / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "current.json").write_text(
        json.dumps({"metadata_path": "source.json", "pdf_path": "journal.pdf"}),
        encoding="utf-8",
    )
    (source_dir / "source.json").write_text(
        json.dumps({"revision": 1, "source_url": source_url, "sha256": "digest"}),
        encoding="utf-8",
    )


def records() -> tuple[list[dict], list[dict]]:
    journals = [
        {
            "session_id": f"session-{number}",
            "session_date": date,
            "journal_number": number,
            "source_url": f"https://senate.gov.ph/{number}.pdf",
        }
        for number, date in ((1, "2026-07-13"), (2, "2026-07-14"), (3, "2026-07-15"))
    ]
    matches = [
        {
            "session_id": item["session_id"],
            "status": "matched" if item["journal_number"] != 3 else "review_required",
            "video": {"video_id": f"video-{item['journal_number']}"}
            if item["journal_number"] != 3
            else None,
        }
        for item in journals
    ]
    return journals, matches


def test_plan_is_forward_only_and_preserves_unmatched_new_journal(tmp_path: Path) -> None:
    journals, matches = records()
    write_current_source(tmp_path, "2026-07-14", journals[1]["source_url"])
    plan = plan_intake(tmp_path, journals, matches)
    assert [(item.session_date, item.reason, item.video_id) for item in plan] == [
        ("2026-07-15", "new_journal", None)
    ]


def test_plan_rechecks_latest_source_for_same_url_revision(tmp_path: Path) -> None:
    journals, matches = records()
    write_current_source(tmp_path, "2026-07-15", journals[2]["source_url"])
    plan = plan_intake(tmp_path, journals, matches)
    assert [(item.session_date, item.reason) for item in plan] == [
        ("2026-07-15", "verify_latest_revision")
    ]


def test_pending_source_branch_is_revisited_until_video_appears(tmp_path: Path) -> None:
    journals, matches = records()
    write_current_source(tmp_path, "2026-07-14", journals[1]["source_url"])
    plan = plan_intake(
        tmp_path,
        journals,
        matches,
        pending_branch_dates={"2026-07-15"},
    )
    assert plan[0].reason == "pending_video_or_processing"


def test_requested_historical_session_bypasses_forward_only_gate(tmp_path: Path) -> None:
    journals, matches = records()
    write_current_source(tmp_path, "2026-07-14", journals[1]["source_url"])
    plan = plan_intake(
        tmp_path,
        journals,
        matches,
        requested_dates={"2026-07-13"},
    )
    assert [(item.session_date, item.reason, item.video_id) for item in plan] == [
        ("2026-07-13", "requested_session", "video-1"),
        ("2026-07-15", "new_journal", None),
    ]
