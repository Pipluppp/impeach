"""Acquire official source records locally and hand preserved inputs to GitHub Actions.

The public interface is intentionally small: ``inspect`` reports what a daily run
would do, while ``run`` preserves each new or revised journal on a review branch
and dispatches the hosted processing workflow once a playlist match exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pipeline.discovery import (
        DiscoveryUnavailable,
        extract_journals,
        fetch_playlist,
        match_journals_to_videos,
        normalize_playlist,
        register_revision,
        update_manifest,
        write_json_if_changed,
    )
    from pipeline.official_source import OfficialSenateSource, OfficialSourceUnavailable
except ModuleNotFoundError:
    from discovery import (
        DiscoveryUnavailable,
        extract_journals,
        fetch_playlist,
        match_journals_to_videos,
        normalize_playlist,
        register_revision,
        update_manifest,
        write_json_if_changed,
    )
    from official_source import OfficialSenateSource, OfficialSourceUnavailable


ROOT = Path(__file__).resolve().parents[1]
BRANCH_RE = re.compile(r"(?:origin/)?automation/session-(?P<date>\d{4}-\d{2}-\d{2})$")


class SourceIntakeError(RuntimeError):
    """The local-to-hosted source handoff cannot advance safely."""


@dataclass(frozen=True)
class IntakeCandidate:
    session_date: str
    session_id: str
    journal_number: int
    branch: str
    reason: str
    source_url: str
    video_id: str | None

    @property
    def matched(self) -> bool:
        return self.video_id is not None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _current_source_metadata(root: Path, session_date: str) -> dict[str, Any] | None:
    source_dir = root / "data" / "sessions" / session_date / "source"
    current_path = source_dir / "current.json"
    if not current_path.is_file():
        return None
    current = _load_json(current_path)
    return _load_json(source_dir / current["metadata_path"])


def plan_intake(
    root: Path,
    journals: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    *,
    pending_branch_dates: set[str] | None = None,
) -> list[IntakeCandidate]:
    """Select forward-only sources, revisions, and already-staged pending sessions."""

    pending_branch_dates = pending_branch_dates or set()
    match_by_id = {item["session_id"]: item for item in matches}
    preserved_dates = [
        journal["session_date"]
        for journal in journals
        if _current_source_metadata(root, journal["session_date"]) is not None
    ]
    latest_preserved = max(preserved_dates, default="")
    latest_published = max((item["session_date"] for item in journals), default="")
    candidates: list[IntakeCandidate] = []
    for journal in journals:
        date = journal["session_date"]
        source = _current_source_metadata(root, date)
        if date in pending_branch_dates and source is None:
            reason = "pending_video_or_processing"
        elif source is None and date > latest_preserved:
            reason = "new_journal"
        elif source is not None and source.get("source_url") != journal["source_url"]:
            reason = "source_url_changed"
        elif source is not None and date == latest_published:
            # Re-read only the newest official journal so same-URL corrections are detected.
            reason = "verify_latest_revision"
        else:
            continue
        match = match_by_id.get(journal["session_id"], {})
        video = match.get("video") if match.get("status") == "matched" else None
        candidates.append(
            IntakeCandidate(
                session_date=date,
                session_id=journal["session_id"],
                journal_number=int(journal["journal_number"]),
                branch=f"automation/session-{date}",
                reason=reason,
                source_url=journal["source_url"],
                video_id=video.get("video_id") if video else None,
            )
        )
    return sorted(candidates, key=lambda item: item.session_date)


class GitRepository:
    """Adapter for the small set of Git operations required by source intake."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, *args: str, check: bool = True) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=check,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def require_clean(self) -> None:
        if self.run("status", "--porcelain"):
            raise SourceIntakeError("source-agent checkout has uncommitted changes")

    def sync_main(self) -> None:
        self.require_clean()
        self.run("fetch", "--prune", "origin")
        self.run("switch", "main")
        self.run("merge", "--ff-only", "origin/main")

    def pending_branch_dates(self) -> set[str]:
        output = self.run(
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/remotes/origin/automation/session-*",
        )
        dates: set[str] = set()
        for line in output.splitlines():
            match = BRANCH_RE.fullmatch(line.strip())
            if match:
                dates.add(match.group("date"))
        return dates

    def _ref_exists(self, ref: str) -> bool:
        return subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", ref],
            cwd=self.root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0

    def switch_session_branch(self, branch: str) -> None:
        remote_ref = f"refs/remotes/origin/{branch}"
        local_ref = f"refs/heads/{branch}"
        local_exists = self._ref_exists(local_ref)
        remote_exists = self._ref_exists(remote_ref)
        if local_exists:
            self.run("switch", branch)
            if remote_exists:
                self.run("merge", "--ff-only", f"origin/{branch}")
        elif remote_exists:
            self.run("switch", "--track", "-c", branch, f"origin/{branch}")
        else:
            self.run("switch", "-c", branch, "origin/main")
        # Pending source branches may live for days while waiting for a video.
        self.run("merge", "--no-edit", "origin/main")

    def commit_and_push(self, branch: str, session_date: str) -> bool:
        self.run("add", "data/discovery", "data/manifest.json", f"data/sessions/{session_date}/source")
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=self.root
        ).returncode
        if staged == 0:
            return False
        self.run("config", "user.name", "Senate source intake")
        self.run("config", "user.email", "source-intake@users.noreply.github.com")
        self.run("commit", "-m", f"Preserve Senate source for {session_date}")
        self.run("push", "--set-upstream", "origin", branch)
        return True


class GitHubActions:
    """Adapter that deduplicates and dispatches hosted processing."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _run(self, *args: str) -> str:
        completed = subprocess.run(
            ["gh", *args], cwd=self.root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    def dispatch_if_idle(self, candidate: IntakeCandidate) -> bool:
        runs = json.loads(
            self._run(
                "run", "list", "--workflow", "process.yml", "--branch", candidate.branch,
                "--limit", "10", "--json", "status"
            )
            or "[]"
        )
        if any(item.get("status") in {"queued", "in_progress", "waiting", "pending"} for item in runs):
            return False
        self._run(
            "workflow", "run", "process.yml", "--ref", candidate.branch,
            "-f", f"session_date={candidate.session_date}", "-f", "publish_pr=true"
        )
        return True


def _write_discovery(
    root: Path,
    journals: list[dict[str, Any]],
    playlist: dict[str, Any],
    matches: list[dict[str, Any]],
    *,
    candidate: IntakeCandidate,
    revision: int,
) -> None:
    manifest_path = root / "data" / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {"schema_version": 1, "sessions": []}
    next_manifest = update_manifest(manifest, journals, matches)
    for item in next_manifest["sessions"]:
        if item["session_id"] == candidate.session_id:
            item["current_revision"] = revision
            if item.get("status") not in {"pdf_parsed", "aligned", "published", "prototype_ready_review_pending"}:
                item["status"] = "source_preserved"
            break
    output = root / "data" / "discovery"
    write_json_if_changed(output / "senate-journals.json", {"schema_version": 1, "journals": journals})
    write_json_if_changed(output / "youtube-playlist.json", {"schema_version": 1, **playlist})
    write_json_if_changed(output / "matches.json", {"schema_version": 1, "matches": matches})
    write_json_if_changed(manifest_path, next_manifest)


def _read_live_inputs(root: Path, source: OfficialSenateSource) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], str]:
    feed, adapter = source.read_feed()
    journals = extract_journals(feed)
    try:
        playlist = normalize_playlist(fetch_playlist())
    except DiscoveryUnavailable:
        existing = root / "data" / "discovery" / "youtube-playlist.json"
        if not existing.is_file():
            raise
        playlist = _load_json(existing)
    matches = match_journals_to_videos(journals, playlist)
    return journals, playlist, matches, adapter


def execute(root: Path, *, dry_run: bool, dispatch: bool) -> dict[str, Any]:
    repo = GitRepository(root)
    if not dry_run:
        repo.sync_main()
    source = OfficialSenateSource()
    journals, playlist, matches, adapter = _read_live_inputs(root, source)
    pending_dates = repo.pending_branch_dates()
    candidates = plan_intake(root, journals, matches, pending_branch_dates=pending_dates)
    report: dict[str, Any] = {
        "feed_adapter": adapter,
        "journals": len(journals),
        "playlist_videos": len(playlist.get("entries", [])),
        "candidates": [asdict(item) for item in candidates],
        "preserved": [],
        "dispatched": [],
    }
    if dry_run:
        return report

    journal_by_date = {item["session_date"]: item for item in journals}
    actions = GitHubActions(root)
    for candidate in candidates:
        journal = journal_by_date[candidate.session_date]
        pdf_content, pdf_adapter = source.read_pdf(candidate.source_url)
        current = _current_source_metadata(root, candidate.session_date)
        if (
            candidate.reason == "verify_latest_revision"
            and current is not None
            and current.get("sha256") == hashlib.sha256(pdf_content).hexdigest()
        ):
            report["preserved"].append(
                {
                    "session_date": candidate.session_date,
                    "revision": current["revision"],
                    "changed": False,
                    "pushed": False,
                    "adapter": pdf_adapter,
                }
            )
            continue
        repo.switch_session_branch(candidate.branch)
        source_dir = root / "data" / "sessions" / candidate.session_date / "source"
        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        revision, changed = register_revision(source_dir, journal, pdf_content, retrieved_at)
        _write_discovery(
            root, journals, playlist, matches, candidate=candidate, revision=revision
        )
        pushed = repo.commit_and_push(candidate.branch, candidate.session_date)
        report["preserved"].append(
            {
                "session_date": candidate.session_date,
                "revision": revision,
                "changed": changed,
                "pushed": pushed,
                "adapter": pdf_adapter,
            }
        )
        already_processed = (
            root
            / "data"
            / "sessions"
            / candidate.session_date
            / "transcript"
            / "segments.json"
        ).is_file()
        if (
            dispatch
            and candidate.matched
            and not already_processed
            and actions.dispatch_if_idle(candidate)
        ):
            report["dispatched"].append(candidate.session_date)
    repo.run("switch", "main")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-dispatch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = execute(
            args.root.resolve(),
            dry_run=args.command == "inspect",
            dispatch=not args.no_dispatch,
        )
    except (SourceIntakeError, OfficialSourceUnavailable, DiscoveryUnavailable, subprocess.SubprocessError) as exc:
        print(f"SOURCE_INTAKE_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
