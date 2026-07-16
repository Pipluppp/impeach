"""Evaluate prototype alignments against preserved navigation-reference moments."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from pipeline.transcribe import normalize_for_matching, write_json_atomic
except ModuleNotFoundError:
    from transcribe import normalize_for_matching, write_json_atomic


def block_match_score(excerpt: str, block: dict[str, Any]) -> float:
    query = normalize_for_matching(excerpt)
    candidate = normalize_for_matching(
        " ".join(filter(None, [block.get("speaker_raw"), block["raw_text"]]))
    )
    query_tokens = set(query.split())
    coverage = len(query_tokens & set(candidate.split())) / max(1, len(query_tokens))
    return 0.75 * coverage + 0.25 * SequenceMatcher(None, query, candidate).ratio()


def match_reference_block(moment: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [block for block in blocks if block["page"] == moment["journal_page"]]
    if not candidates:
        raise ValueError(f"no blocks on reference page {moment['journal_page']}")
    return max(candidates, key=lambda block: block_match_score(moment["journal_excerpt"], block))


def distance_to_range(value: float, start: float, end: float) -> float:
    if start <= value <= end:
        return 0.0
    return min(abs(value - start), abs(value - end))


def evaluate(
    journal: dict[str, Any], alignment: dict[str, Any], references: dict[str, Any]
) -> dict[str, Any]:
    entries = {entry["block_id"]: entry for entry in alignment["entries"]}
    results: list[dict[str, Any]] = []
    for moment in references["moments"]:
        block = match_reference_block(moment, journal["blocks"])
        entry = entries.get(block["id"])
        expected = moment["session_range"]
        tolerance = float(moment["tolerance_seconds"])
        result: dict[str, Any] = {
            "id": moment["id"],
            "block_id": block["id"],
            "block_kind": block["kind"],
            "block_match_score": round(block_match_score(moment["journal_excerpt"], block), 4),
            "expected_range": expected,
            "tolerance_seconds": tolerance,
            "aligned": entry is not None,
        }
        if entry:
            overlaps = entry["start"] <= expected["end"] and entry["end"] >= expected["start"]
            navigation_error = distance_to_range(expected["start"], entry["start"], entry["end"])
            result.update(
                {
                    "actual_range": {"start": entry["start"], "end": entry["end"]},
                    "precision": entry["precision"],
                    "review_state": entry["review_state"],
                    "range_overlap": overlaps,
                    "navigation_error_seconds": round(navigation_error, 3),
                    "pass": overlaps or navigation_error <= tolerance,
                }
            )
            if moment["id"] == "exhibit-timestamp-trap":
                exhibit_refs = [
                    item
                    for item in entry.get("source_time_references", [])
                    if item.get("time_domain") == "exhibit"
                ]
                result["exhibit_time_preserved"] = bool(exhibit_refs)
                result["exhibit_time_not_used_as_seek"] = all(
                    not (entry["start"] <= item["time_seconds"] <= entry["end"])
                    for item in exhibit_refs
                )
                result["pass"] = bool(
                    result["pass"]
                    and result["exhibit_time_preserved"]
                    and result["exhibit_time_not_used_as_seek"]
                )
        else:
            result["pass"] = False
        results.append(result)

    passed = sum(bool(result["pass"]) for result in results)
    return {
        "schema_version": 1,
        "session_id": alignment["session_id"],
        "reference_method": references["reference_method"],
        "metric": "reference range overlap or distance-to-range within stated tolerance",
        "sample_count": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = evaluate(
        json.loads(args.journal.read_text(encoding="utf-8")),
        json.loads(args.alignment.read_text(encoding="utf-8")),
        json.loads(args.references.read_text(encoding="utf-8")),
    )
    write_json_atomic(args.output, payload)
    print(json.dumps({"passed": payload["passed"], "samples": payload["sample_count"]}))
    return 0 if payload["passed"] == payload["sample_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
