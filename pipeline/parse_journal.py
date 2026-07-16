"""Extract a Senate journal into deterministic page-aware Markdown and typed blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz


ROOT = Path(__file__).resolve().parents[1]
PARSER_NAME = "pymupdf-blocks"
PARSER_VERSION = "0.3.0"

DATE_HEADER_RE = re.compile(r"^(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY),\s+[A-Z]+\s+\d{1,2},?\s*\d{4}$")
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
CONTINUATION_RE = re.compile(r"^[{([]\s*continuation\s*[)}\]]$", re.IGNORECASE)
SPEAKER_RE = re.compile(
    r"^(?P<speaker>"
    r"The Presiding Officer|The Chair|"
    r"(?:Senator-Judge|Senator|Senate President|Representative|Presiding Officer|Justice)\s+[^.]{1,90}|"
    r"(?:Mr|Ms|Mrs|Atty|Attys|Sen|Rep)\s*[.,]\s*[^.]{1,80}"
    r")\.\s*(?P<utterance>.*)$",
    re.IGNORECASE,
)
EXHIBIT_RE = re.compile(
    r"\bExhibit\s+[“\"']?(?P<id>[A-Z0-9]+(?:\s*-\s*[A-Z0-9]+)+|[A-Z0-9]+)",
    re.IGNORECASE,
)
CLOCK_RE = re.compile(r"\b(?P<hour>\d{1,2}):(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?\s*(?P<ampm>[ap]\.?m\.?)?", re.IGNORECASE)

NORMALIZATION_REPLACEMENTS = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "PRESH)ING": "PRESIDING",
    "P ANGELIN AN": "PANGILINAN",
    "Wimess": "Witness",
    "wimess": "witness",
}

PROCEDURAL_SECTIONS = (
    "CALL TO ORDER",
    "PROCLAMATION",
    "ROLL CALL",
    "APPROVAL OF THE JOURNAL",
    "CALLING OF THE CASE",
    "APPEARANCES",
    "SUSPENSION",
    "RESUMPTION",
    "ADJOURNMENT",
    "ORDER OF THE PRESIDING OFFICER",
)


@dataclass
class Fragment:
    page: int
    source_indexes: list[int]
    source_text: str
    raw_text: str
    bbox: list[float]
    role: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def join_pdf_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def is_page_header(text: str) -> bool:
    return bool(DATE_HEADER_RE.fullmatch(text.replace("  ", " ").strip()))


def is_heading(text: str) -> bool:
    if is_page_header(text) or PAGE_NUMBER_RE.fullmatch(text) or len(text) > 180:
        return False
    heading_text = re.sub(
        r"\s+[\{\(\[]\s*continuation.*$", "", text, flags=re.IGNORECASE
    )
    letters = [char for char in heading_text if char.isalpha()]
    if len(letters) < 4:
        return False
    return sum(char.isupper() for char in letters) / len(letters) >= 0.92


def speaker_match(text: str) -> re.Match[str] | None:
    """Return a dialogue match only when text remains after the speaker label."""
    match = SPEAKER_RE.match(text)
    if not match or not match.group("utterance").strip():
        return None
    return match


def split_artifact_block(page: int, index: int, text: str, bbox: Iterable[float]) -> list[Fragment]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and all(is_page_header(line) or PAGE_NUMBER_RE.fullmatch(line) for line in lines):
        return [
            Fragment(
                page=page,
                source_indexes=[index],
                source_text=line,
                raw_text=line,
                bbox=list(map(float, bbox)),
                role="page_header" if is_page_header(line) else "page_number",
            )
            for line in lines
        ]
    raw = join_pdf_lines(text)
    role = "page_header" if is_page_header(raw) else "page_number" if PAGE_NUMBER_RE.fullmatch(raw) else None
    return [
        Fragment(
            page=page,
            source_indexes=[index],
            source_text=text.rstrip("\n"),
            raw_text=raw,
            bbox=list(map(float, bbox)),
            role=role,
        )
    ]


def structural(fragment: Fragment) -> bool:
    return bool(
        fragment.role
        or is_heading(fragment.raw_text)
        or speaker_match(fragment.raw_text)
        or fragment.raw_text.startswith("[")
    )


def should_merge(previous: Fragment, current: Fragment) -> bool:
    if previous.page != current.page or structural(previous) or structural(current):
        return False
    if CONTINUATION_RE.fullmatch(current.raw_text):
        return is_heading(previous.raw_text)
    if not previous.raw_text:
        return False
    return (
        previous.raw_text.endswith((",", ";", ":", "—", "–"))
        or previous.raw_text[-1] not in ".?!)]”’"
        or (current.raw_text and current.raw_text[0].islower())
        or current.raw_text.startswith(("‘", "’", "“", '"'))
    )


def merge_pair(previous: Fragment, current: Fragment) -> Fragment:
    bbox = [
        min(previous.bbox[0], current.bbox[0]),
        min(previous.bbox[1], current.bbox[1]),
        max(previous.bbox[2], current.bbox[2]),
        max(previous.bbox[3], current.bbox[3]),
    ]
    return Fragment(
        page=previous.page,
        source_indexes=previous.source_indexes + current.source_indexes,
        source_text=f"{previous.source_text}\n{current.source_text}",
        raw_text=f"{previous.raw_text} {current.raw_text}".strip(),
        bbox=bbox,
        role=previous.role,
    )


def extract_fragments(document: fitz.Document) -> list[Fragment]:
    output: list[Fragment] = []
    for page_number, page in enumerate(document, start=1):
        page_fragments: list[Fragment] = []
        for index, block in enumerate(page.get_text("blocks", sort=True)):
            if not block[4].strip():
                continue
            for fragment in split_artifact_block(page_number, index, block[4], block[:4]):
                if page_fragments and CONTINUATION_RE.fullmatch(fragment.raw_text) and is_heading(page_fragments[-1].raw_text):
                    page_fragments[-1] = merge_pair(page_fragments[-1], fragment)
                elif page_fragments and should_merge(page_fragments[-1], fragment):
                    page_fragments[-1] = merge_pair(page_fragments[-1], fragment)
                else:
                    page_fragments.append(fragment)
        output.extend(page_fragments)
    return output


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.casefold()).strip("-")
    return slug[:80] or "section"


def normalize_for_matching(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    for source, replacement in NORMALIZATION_REPLACEMENTS.items():
        value = value.replace(source, replacement)
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "-", value)
    value = value.casefold()
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def parse_speaker(text: str) -> tuple[str | None, str]:
    match = speaker_match(text)
    if not match:
        return None, text
    return match.group("speaker"), match.group("utterance")


def time_references(text: str) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    exhibit = EXHIBIT_RE.search(text)
    exhibit_context = bool(re.search(r"\b(?:timestamp|video|exhibit)\b", text, re.IGNORECASE))
    for match in CLOCK_RE.finditer(text):
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        second = int(match.group("second") or 0)
        ampm = match.group("ampm")
        if ampm:
            normalized_hour = hour % 12 + (12 if ampm.lower().startswith("p") else 0)
            references.append(
                {
                    "time_domain": "wall_clock",
                    "display": match.group(0),
                    "time_seconds_since_midnight": normalized_hour * 3600 + minute * 60 + second,
                }
            )
        elif exhibit_context:
            item: dict[str, Any] = {
                "time_domain": "exhibit",
                "display": match.group(0),
                "time_seconds": hour * 3600 + minute * 60 + second if match.group("second") else hour * 60 + minute,
            }
            if exhibit:
                item["exhibit_id"] = re.sub(r"\s+", "", exhibit.group("id"))
            references.append(item)
    return references


def classify(raw_text: str, role: str | None, section_title: str | None) -> str:
    if role:
        return "other"
    if is_heading(raw_text):
        return "heading"
    if speaker_match(raw_text):
        return "speaker_utterance"
    if len(re.sub(r"[\W_]", "", normalize_for_matching(raw_text), flags=re.UNICODE)) <= 1:
        return "other"
    if re.fullmatch(r"\[\s*Video presentation\s*\]", raw_text, re.IGNORECASE):
        return "video_presentation"
    if CLOCK_RE.search(raw_text) and re.search(r"\b(?:At|It was)\b", raw_text):
        return "session_clock_anchor"
    if section_title and "PRAYER" in section_title:
        return "prepared_statement"
    if section_title and any(name in section_title for name in PROCEDURAL_SECTIONS):
        return "procedural_event"
    return "narrative_summary"


def build_blocks(
    fragments: list[Fragment], journal_number: int, source_revision: int
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    page_counts: dict[int, int] = {}
    section_title: str | None = None
    section_id: str | None = None
    section_counts: dict[str, int] = {}
    previous_substantive: dict[str, Any] | None = None

    for sequence, fragment in enumerate(fragments, start=1):
        page_counts[fragment.page] = page_counts.get(fragment.page, 0) + 1
        local_number = page_counts[fragment.page]
        block_id = f"j{journal_number:02d}-p{fragment.page:03d}-b{local_number:03d}"
        kind = classify(fragment.raw_text, fragment.role, section_title)
        if kind == "heading":
            section_title = fragment.raw_text
            section_base = slugify(section_title)
            section_counts[section_base] = section_counts.get(section_base, 0) + 1
            section_id = (
                section_base
                if section_counts[section_base] == 1
                else f"{section_base}-{section_counts[section_base]}"
            )
        speaker, body = parse_speaker(fragment.raw_text)
        block: dict[str, Any] = {
            "id": block_id,
            "sequence": sequence,
            "page": fragment.page,
            "section_id": section_id,
            "section_title": section_title,
            "kind": kind,
            "raw_text": body if speaker else fragment.raw_text,
            "normalized_text": normalize_for_matching(body if speaker else fragment.raw_text),
            "source_text": fragment.source_text,
            "source_revision": source_revision,
            "source_block_indexes": fragment.source_indexes,
            "bbox": fragment.bbox,
        }
        if fragment.role:
            block["role"] = fragment.role
        if speaker:
            block["speaker_raw"] = speaker
            block["speaker_id"] = slugify(speaker)
        references = time_references(fragment.raw_text)
        if references:
            block["time_references"] = references
            if any(item["time_domain"] == "exhibit" for item in references):
                block["features"] = ["exhibit_timestamp"]

        if (
            previous_substantive
            and previous_substantive["page"] == fragment.page - 1
            and kind not in {"heading", "other"}
            and fragment.raw_text
            and fragment.raw_text[0].islower()
            and previous_substantive["raw_text"]
            and previous_substantive["raw_text"][-1] not in ".?!)]”’"
        ):
            block["continuation_of"] = previous_substantive["id"]

        if kind != "other":
            previous_substantive = block
        blocks.append(block)
    return blocks


def block_markdown(block: dict[str, Any]) -> str:
    text = block["raw_text"]
    if block["kind"] == "heading":
        return f"## {text}"
    if block.get("speaker_raw"):
        return f"**{block['speaker_raw']}.** {text}".rstrip()
    if block["kind"] == "video_presentation":
        return f"_{text}_"
    return text


def build_markdown(blocks: list[dict[str, Any]], source_sha256: str) -> str:
    lines = [
        "<!-- Generated from the official Senate journal PDF. -->",
        f"<!-- source_sha256: {source_sha256} -->",
        f"<!-- parser: {PARSER_NAME} {PARSER_VERSION} -->",
        "",
    ]
    current_page = None
    for block in blocks:
        if block["page"] != current_page:
            current_page = block["page"]
            lines.extend([f"<!-- page: {current_page} -->", ""])
        lines.extend([block_markdown(block), ""])
    return "\n".join(lines).rstrip() + "\n"


def parse_document(
    pdf_path: Path, journal_number: int, source_revision: int
) -> tuple[str, dict[str, Any]]:
    source_sha = sha256_file(pdf_path)
    with fitz.open(pdf_path) as document:
        fragments = extract_fragments(document)
        page_count = document.page_count
        image_counts = [len(page.get_images(full=True)) for page in document]
        text_character_count = sum(len(page.get_text()) for page in document)
    pages_with_images = sum(count > 0 for count in image_counts)
    blocks = build_blocks(fragments, journal_number, source_revision)
    payload = {
        "schema_version": 1,
        "parser": {"name": PARSER_NAME, "version": PARSER_VERSION, "pymupdf_version": fitz.__version__},
        "source_sha256": source_sha,
        "source_revision": source_revision,
        "page_count": page_count,
        "source_characteristics": {
            "image_count": sum(image_counts),
            "pages_with_images": pages_with_images,
            "text_character_count": text_character_count,
            "likely_image_based_with_ocr": bool(
                page_count and pages_with_images / page_count >= 0.8 and text_character_count
            ),
        },
        "block_count": len(blocks),
        "blocks": blocks,
    }
    return build_markdown(blocks, source_sha), payload


def write_text_if_changed(path: Path, content: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--journal-number", type=int, required=True)
    parser.add_argument("--source-revision", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    markdown, payload = parse_document(args.pdf, args.journal_number, args.source_revision)
    changed = [
        write_text_if_changed(args.output_dir / "journal.md", markdown),
        write_text_if_changed(
            args.output_dir / "blocks.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        ),
    ]
    print(
        json.dumps(
            {
                "pages": payload["page_count"],
                "blocks": payload["block_count"],
                "changed_files": sum(changed),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
