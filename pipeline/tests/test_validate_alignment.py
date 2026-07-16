from __future__ import annotations

import json
from pathlib import Path

from pipeline.validate_alignment import evaluate


ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "data" / "sessions" / "2026-07-14"


def test_preserved_review_anchors_cover_all_vertical_slice_moments() -> None:
    journal = json.loads((SESSION / "journal" / "blocks.json").read_text(encoding="utf-8"))
    overrides = json.loads(
        (SESSION / "alignment" / "manual-overrides.json").read_text(encoding="utf-8")
    )
    block_lookup = {block["id"]: block for block in journal["blocks"]}
    entries = []
    for override in overrides["overrides"]:
        entries.append(
            {
                **override,
                "review_state": "manual_reviewed",
                "source_time_references": block_lookup[override["block_id"]].get(
                    "time_references", []
                ),
            }
        )
    alignment = {
        "session_id": "impeachment-trial-06",
        "entries": entries,
    }
    references = json.loads(
        (
            ROOT / "pipeline" / "fixtures" / "vertical-slice" / "moments.json"
        ).read_text(encoding="utf-8")
    )

    result = evaluate(journal, alignment, references)

    assert result["passed"] == 6
    assert result["pass_rate"] == 1.0
    trap = next(item for item in result["results"] if item["id"] == "exhibit-timestamp-trap")
    assert trap["block_id"] == "j06-p005-b025"
    assert trap["exhibit_time_preserved"] is True
    assert trap["exhibit_time_not_used_as_seek"] is True
