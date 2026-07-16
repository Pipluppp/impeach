from __future__ import annotations

import json
from pathlib import Path

from pipeline.build_public_data import build_payload, validate_payload


ROOT = Path(__file__).resolve().parents[2]
SESSION_ROOT = ROOT / "data" / "sessions" / "2026-07-14"


def test_real_public_payload_is_schema_valid_and_excludes_matching_text() -> None:
    journal = json.loads((SESSION_ROOT / "journal" / "blocks.json").read_text(encoding="utf-8"))
    payload = build_payload(
        session=json.loads((SESSION_ROOT / "session.json").read_text(encoding="utf-8")),
        source=json.loads((SESSION_ROOT / "source" / "source.json").read_text(encoding="utf-8")),
        journal=journal,
        alignment=None,
    )
    schema = json.loads(
        (ROOT / "pipeline" / "schemas" / "public-session.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate_payload(payload, schema)
    serialized = json.dumps(payload)
    assert '"normalized_text"' not in serialized
    assert len(payload["pages"]) == 92
    assert sum(len(page["blocks"]) for page in payload["pages"]) == 2405
    assert len(payload["outline"]) == sum(
        block["kind"] == "heading" for block in journal["blocks"]
    )
    assert payload["processing"]["alignment_summary"] == {
        "total_blocks": 2405,
        "timed_blocks": 0,
        "coverage": 0.0,
        "needs_review": 0,
        "manual_reviewed": 0,
        "unresolved_conflicts": 0,
    }
    trap = next(block for page in payload["pages"] for block in page["blocks"] if "timestamp 51:07" in block["text"])
    assert trap["source_time_references"][0]["time_domain"] == "exhibit"
    assert trap["timing"] is None
