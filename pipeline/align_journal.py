"""CLI adapter for the journal-to-ASR alignment module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from pipeline.alignment import align
    from pipeline.transcribe import write_json_atomic
except ModuleNotFoundError:  # Direct script execution.
    from alignment import align
    from transcribe import write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = align(
        json.loads(args.journal.read_text(encoding="utf-8")),
        json.loads(args.transcript.read_text(encoding="utf-8")),
        json.loads(args.overrides.read_text(encoding="utf-8")),
    )
    write_json_atomic(args.output, payload)
    print(json.dumps({
        "entries": payload["entry_count"],
        "reviews": len(payload["review_queue"]),
        "diagnostics": payload["diagnostics"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["align", "main"]
