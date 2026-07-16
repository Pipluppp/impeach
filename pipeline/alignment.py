"""Deep journal-to-ASR alignment module.

Public interface: ``align(journal, transcript, overrides=None) -> dict``.

The caller supplies preserved data and receives a deterministic result. Structural eligibility,
candidate retrieval, fuzzy scoring, monotonic path selection, confidence, contextual ranges,
manual constraints, and diagnostics remain implementation details of this module.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

try:
    from pipeline.transcribe import normalize_for_matching
except ModuleNotFoundError:  # Direct script execution through the CLI adapter.
    from transcribe import normalize_for_matching


ALIGNER_NAME = "monotonic-journal-asr"
ALIGNER_VERSION = "0.4.0"

SPOKEN_KINDS = {"speaker_utterance", "prepared_statement"}
SUMMARY_KINDS = {"narrative_summary", "procedural_event", "session_clock_anchor"}

PROPOSAL_SCORE = 0.54
AUTO_SCORE = 0.72
AUTO_COVERAGE = 0.62
AUTO_MARGIN = 0.055
MAX_CANDIDATES_PER_BLOCK = 6
MAX_SEEDS = 14
MAX_WINDOW_SEGMENTS = 20
MAX_DIRECT_CONTEXT_SECONDS = 45.0
MAX_SUMMARY_CONTEXT_SECONDS = 180.0

# These remove only extremely common courtroom glue from candidate seeding. Final scoring still
# sees every token. Filipino entries matter because an English-only list makes "po" and "ang"
# appear falsely distinctive in this code-switched record.
SEED_STOPWORDS = {
    "a", "an", "and", "ang", "ano", "are", "as", "at", "ay", "ba", "be", "but", "by",
    "dahil", "din", "doon", "for", "from", "had", "has", "have", "he", "her", "hindi",
    "his", "honor", "i", "if", "ikaw", "in", "is", "it", "ito", "iyon", "ka", "kami",
    "kasi", "kay", "ko", "kung", "lang", "may", "mga", "mo", "na", "naman", "ng",
    "ni", "nila", "ninyo", "no", "not", "of", "on", "opo", "or", "our", "pa", "para",
    "po", "sa", "she", "sir", "so", "that", "the", "their", "there", "they", "this",
    "to", "was", "we", "were", "with", "yes", "you", "your", "yung",
}


@dataclass(frozen=True)
class Candidate:
    start_index: int
    end_index: int  # exclusive
    score: float
    coverage: float
    exact_coverage: float
    distinctive_tokens: int


@dataclass
class PathNode:
    block_index: int
    candidate: Candidate
    total: float
    previous: int | None


class _FenwickBest:
    """Prefix maximum used by monotonic path selection."""

    def __init__(self, size: int):
        self.values: list[tuple[float, int | None]] = [(0.0, None)] * (size + 2)

    def update(self, position: int, value: tuple[float, int]) -> None:
        while position < len(self.values):
            if value[0] > self.values[position][0]:
                self.values[position] = value
            position += position & -position

    def query(self, position: int) -> tuple[float, int | None]:
        best = (0.0, None)
        while position > 0:
            if self.values[position][0] > best[0]:
                best = self.values[position]
            position -= position & -position
        return best


class _TranscriptIndex:
    def __init__(self, segments: list[dict[str, Any]]):
        self.segments = segments
        self.starts = [float(segment["start"]) for segment in segments]
        self.tokens = [segment["normalized_text"].split() for segment in segments]
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, tokens in enumerate(self.tokens):
            for token in set(tokens):
                self.postings[token].append(index)

    def at_or_after(self, seconds: float) -> int:
        return min(len(self.segments), bisect.bisect_left(self.starts, seconds))

    def at_or_before(self, seconds: float) -> int:
        return max(-1, bisect.bisect_right(self.starts, seconds) - 1)

    def idf(self, token: str) -> float:
        return math.log((len(self.segments) + 1) / (len(self.postings.get(token, ())) + 1)) + 1.0

    def informative(self, tokens: list[str]) -> list[str]:
        maximum_frequency = max(5, math.ceil(len(self.segments) * 0.025))
        return [
            token for token in tokens
            if len(token) > 2
            and token not in SEED_STOPWORDS
            and len(self.postings.get(token, ())) <= maximum_frequency
        ]

    def candidates(
        self, query: str, *, minimum_index: int, maximum_index: int
    ) -> list[Candidate]:
        query_tokens = normalize_for_matching(query).split()
        if len(query_tokens) < 2 or minimum_index > maximum_index:
            return []
        informative = self.informative(query_tokens)
        if not informative:
            return []

        rare = sorted(set(informative), key=lambda token: (len(self.postings.get(token, ())), token))[:12]
        votes: Counter[int] = Counter()
        for token in rare:
            for index in self.postings.get(token, ()):
                if minimum_index <= index <= maximum_index:
                    votes[index] += self.idf(token)
        if not votes:
            return []

        seeds = [index for index, _ in votes.most_common(MAX_SEEDS)]
        query_size = len(query_tokens)
        maximum_tokens = max(24, math.ceil(query_size * 2.0))
        minimum_tokens = max(1, math.floor(query_size * 0.35))
        visited: set[tuple[int, int]] = set()
        found: list[Candidate] = []

        for seed in seeds:
            for lead in range(0, 4):
                start = seed - lead
                if start < minimum_index:
                    continue
                candidate_tokens: list[str] = []
                for end in range(start + 1, min(maximum_index + 2, start + MAX_WINDOW_SEGMENTS + 1)):
                    candidate_tokens.extend(self.tokens[end - 1])
                    if len(candidate_tokens) < minimum_tokens:
                        continue
                    if len(candidate_tokens) > maximum_tokens:
                        break
                    key = (start, end)
                    if key in visited:
                        continue
                    visited.add(key)
                    score, coverage, exact = self._score(query_tokens, candidate_tokens)
                    if score < PROPOSAL_SCORE or coverage < 0.46:
                        continue
                    found.append(Candidate(
                        start_index=start,
                        end_index=end,
                        score=score,
                        coverage=coverage,
                        exact_coverage=exact,
                        distinctive_tokens=len(set(informative)),
                    ))

        found.sort(key=lambda item: (item.score, item.coverage, item.exact_coverage), reverse=True)
        distinct: list[Candidate] = []
        for candidate in found:
            if any(
                max(0, min(candidate.end_index, kept.end_index) - max(candidate.start_index, kept.start_index))
                / max(1, min(candidate.end_index - candidate.start_index, kept.end_index - kept.start_index))
                >= 0.6
                for kept in distinct
            ):
                continue
            distinct.append(candidate)
            if len(distinct) == MAX_CANDIDATES_PER_BLOCK:
                break
        return distinct

    def _score(
        self, query_tokens: list[str], candidate_tokens: list[str]
    ) -> tuple[float, float, float]:
        candidate_counts = Counter(candidate_tokens)
        query_counts = Counter(query_tokens)
        total_weight = sum(self.idf(token) * count for token, count in query_counts.items()) or 1.0
        exact_weight = sum(
            self.idf(token) * min(count, candidate_counts[token])
            for token, count in query_counts.items()
        )
        exact_coverage = exact_weight / total_weight

        # Soft token recall tolerates ASR spelling and word-boundary errors while preserving the
        # journal text. It is intentionally query-focused because ASR segments often contain an
        # adjacent short turn.
        soft_weight = 0.0
        candidate_unique = set(candidate_tokens)
        for token, count in query_counts.items():
            similarity = 1.0 if token in candidate_unique else 0.0
            if similarity == 0.0 and len(token) >= 4:
                comparable = (
                    other for other in candidate_unique
                    if abs(len(other) - len(token)) <= max(2, len(token) // 3)
                )
                similarity = max(
                    (fuzz.ratio(token, other) / 100.0 for other in comparable),
                    default=0.0,
                )
                if similarity < 0.72:
                    similarity = 0.0
            soft_weight += self.idf(token) * count * similarity
        coverage = soft_weight / total_weight

        query_text = " ".join(query_tokens)
        candidate_text = " ".join(candidate_tokens)
        ordered = fuzz.ratio(query_text, candidate_text) / 100.0
        compact = fuzz.ratio("".join(query_tokens), "".join(candidate_tokens)) / 100.0
        length_balance = min(len(query_tokens), len(candidate_tokens)) / max(len(query_tokens), len(candidate_tokens))
        score = (
            0.42 * coverage
            + 0.23 * exact_coverage
            + 0.20 * ordered
            + 0.15 * compact
        ) * (0.84 + 0.16 * math.sqrt(length_balance))
        return score, coverage, exact_coverage


def _candidate_weight(candidate: Candidate) -> float:
    specificity = min(0.6, candidate.distinctive_tokens * 0.08)
    return max(0.01, (candidate.score - PROPOSAL_SCORE) * 5.0 + candidate.coverage + specificity)


def _select_monotonic_path(
    candidate_groups: list[tuple[int, list[Candidate]]], transcript_size: int
) -> list[tuple[int, Candidate, float]]:
    tree = _FenwickBest(transcript_size + 2)
    nodes: list[PathNode] = []
    group_lookup = {block_index: candidates for block_index, candidates in candidate_groups}
    for block_index, candidates in candidate_groups:
        pending_updates: list[tuple[int, tuple[float, int]]] = []
        for candidate in candidates:
            previous_total, previous_node = tree.query(candidate.start_index + 1)
            node = PathNode(
                block_index=block_index,
                candidate=candidate,
                total=previous_total + _candidate_weight(candidate),
                previous=previous_node,
            )
            node_index = len(nodes)
            nodes.append(node)
            pending_updates.append((candidate.end_index + 1, (node.total, node_index)))
        for position, value in pending_updates:
            tree.update(position, value)

    _, best_index = tree.query(transcript_size + 1)
    selected: list[tuple[int, Candidate, float]] = []
    while best_index is not None:
        node = nodes[best_index]
        alternatives = group_lookup[node.block_index]
        other_scores = [
            item.score for item in alternatives
            if abs(item.start_index - node.candidate.start_index) > 3
        ]
        margin = node.candidate.score - max(other_scores, default=0.0)
        selected.append((node.block_index, node.candidate, margin))
        best_index = node.previous
    selected.reverse()
    return selected


def _manual_entries(
    blocks: list[dict[str, Any]], overrides: dict[str, Any], duration: float
) -> dict[str, dict[str, Any]]:
    valid_ids = {block["id"] for block in blocks}
    entries: dict[str, dict[str, Any]] = {}
    for override in overrides.get("overrides", []):
        block_id = override.get("block_id")
        if block_id not in valid_ids:
            raise ValueError(f"unknown manual override block {block_id}")
        start = float(override["start"])
        end = float(override["end"])
        if override.get("time_domain") != "session_video" or not (0 <= start <= end <= duration):
            raise ValueError(f"invalid manual override for {block_id}")
        entries[block_id] = {
            "block_id": block_id,
            "start": start,
            "end": end,
            "time_domain": "session_video",
            "precision": override.get("precision", "manual_reviewed_range"),
            "confidence": 1.0,
            "review_state": "manual_reviewed",
            "evidence": {
                "method": "manual_override",
                "reviewer": override.get("reviewer"),
                "note": override["note"],
                "claim_limit": "Human-reviewed range; no word-level claim.",
            },
        }
    return entries


def _direct_entry(
    block: dict[str, Any], candidate: Candidate, margin: float, transcript: _TranscriptIndex
) -> dict[str, Any]:
    first = transcript.segments[candidate.start_index]
    last = transcript.segments[candidate.end_index - 1]
    accepted = (
        candidate.score >= AUTO_SCORE
        and candidate.coverage >= AUTO_COVERAGE
        and candidate.distinctive_tokens >= 3
        and margin >= AUTO_MARGIN
    )
    confidence = min(0.98, max(0.0, (candidate.score - 0.48) / 0.44))
    return {
        "block_id": block["id"],
        "start": first["start"],
        "end": last["end"],
        "time_domain": "session_video",
        "precision": "approximate_dialogue_turn",
        "confidence": round(confidence, 3),
        "review_state": "auto_accepted" if accepted else "needs_review",
        "evidence": {
            "method": "monotonic_fuzzy_asr_path",
            "match_score": round(candidate.score, 4),
            "token_coverage": round(candidate.coverage, 4),
            "exact_token_coverage": round(candidate.exact_coverage, 4),
            "uniqueness_margin": round(margin, 4),
            "distinctive_tokens": candidate.distinctive_tokens,
            "asr_segment_ids": [
                segment["id"]
                for segment in transcript.segments[candidate.start_index:candidate.end_index]
            ],
            "claim_limit": "ASR-assisted turn range; not a word-level timestamp.",
        },
    }


def _add_contextual_ranges(
    blocks: list[dict[str, Any]], entries: dict[str, dict[str, Any]]
) -> None:
    positions = [index for index, block in enumerate(blocks) if block["id"] in entries]
    for index, block in enumerate(blocks):
        if block["id"] in entries or block["kind"] == "other":
            continue
        before = next((position for position in reversed(positions) if position < index), None)
        after = next((position for position in positions if position > index), None)
        if before is None or after is None:
            continue
        previous = entries[blocks[before]["id"]]
        following = entries[blocks[after]["id"]]
        start = float(previous["end"])
        end = float(following["start"])
        if end < start:
            continue
        gap = end - start
        if block["kind"] in SPOKEN_KINDS:
            if gap > MAX_DIRECT_CONTEXT_SECONDS:
                continue
            precision = "contextual_dialogue_range"
            claim = "Unmatched dialogue bounded by neighboring monotonic matches; not a direct lexical match."
        elif block["kind"] in SUMMARY_KINDS:
            if gap > MAX_SUMMARY_CONTEXT_SECONDS:
                continue
            precision = "narrative_summary_range"
            claim = "Journal summary bounded by neighboring monotonic dialogue; not verbatim speech."
        elif block["kind"] == "heading":
            if gap > MAX_SUMMARY_CONTEXT_SECONDS:
                continue
            precision = "section_range"
            claim = "Section context only; the printed heading is not treated as spoken audio."
        else:
            continue
        entries[block["id"]] = {
            "block_id": block["id"],
            "start": round(start, 3),
            "end": round(end, 3),
            "time_domain": "session_video",
            "precision": precision,
            "confidence": round(max(0.08, 0.46 - gap / 500), 3),
            "review_state": "needs_review",
            "evidence": {"method": "bounded_monotonic_context", "claim_limit": claim},
        }


def _automatic_entries(
    blocks: list[dict[str, Any]], transcript: _TranscriptIndex,
    manual: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    positions = {block["id"]: index for index, block in enumerate(blocks)}
    ordered_manual = sorted((positions[block_id], entry) for block_id, entry in manual.items())
    for (_, previous), (_, current) in zip(ordered_manual, ordered_manual[1:]):
        if current["start"] < previous["start"]:
            raise ValueError("manual override ranges are not monotonic in journal order")

    entries = dict(manual)
    interval_bounds: list[tuple[int, int, int, int]] = []
    previous_block = -1
    previous_segment = 0
    for block_position, entry in ordered_manual:
        next_segment = transcript.at_or_before(float(entry["start"]))
        interval_bounds.append((previous_block + 1, block_position, previous_segment, next_segment))
        previous_block = block_position
        previous_segment = transcript.at_or_after(float(entry["end"]))
    interval_bounds.append((previous_block + 1, len(blocks), previous_segment, len(transcript.segments) - 1))

    candidate_blocks = 0
    selected_blocks = 0
    for block_start, block_end, segment_start, segment_end in interval_bounds:
        if block_start >= block_end or segment_start > segment_end:
            continue
        groups: list[tuple[int, list[Candidate]]] = []
        for block_index in range(block_start, block_end):
            block = blocks[block_index]
            if block["kind"] not in SPOKEN_KINDS:
                continue
            candidates = transcript.candidates(
                block.get("normalized_text", block["raw_text"]),
                minimum_index=segment_start,
                maximum_index=segment_end,
            )
            if candidates:
                candidate_blocks += 1
                groups.append((block_index, candidates))
        if not groups:
            continue
        for block_index, candidate, margin in _select_monotonic_path(groups, len(transcript.segments)):
            block = blocks[block_index]
            entries[block["id"]] = _direct_entry(block, candidate, margin, transcript)
            selected_blocks += 1

    return entries, {"candidate_blocks": candidate_blocks, "selected_direct_blocks": selected_blocks}


def align(
    journal: dict[str, Any], transcript_payload: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Align authoritative journal blocks to timestamped ASR.

    Invariants: output follows journal order, session and exhibit time domains stay distinct,
    structural text is never searched as speech, and uncertain content may be left unaligned.
    """
    blocks = journal["blocks"]
    transcript = _TranscriptIndex(transcript_payload["segments"])
    duration = float(transcript_payload["runtime"]["audio_seconds"])
    manual = _manual_entries(blocks, overrides or {"overrides": []}, duration)
    entries, diagnostics = _automatic_entries(blocks, transcript, manual)
    _add_contextual_ranges(blocks, entries)

    ordered: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    last_start = -1.0
    for block in blocks:
        entry = entries.get(block["id"])
        if entry is None:
            continue
        if float(entry["start"]) < last_start:
            conflicts.append({"block_id": block["id"], "reason": "non_monotonic_output"})
            continue
        refs = block.get("time_references")
        if refs:
            entry["source_time_references"] = refs
        ordered.append(entry)
        last_start = float(entry["start"])

    direct_eligible = sum(block["kind"] in SPOKEN_KINDS for block in blocks)
    auto_accepted = sum(entry["review_state"] == "auto_accepted" for entry in ordered)
    manual_reviewed = sum(entry["review_state"] == "manual_reviewed" for entry in ordered)
    direct_matches = sum(entry["precision"] == "approximate_dialogue_turn" for entry in ordered)
    contextual = sum(entry["precision"] != "approximate_dialogue_turn" for entry in ordered)
    diagnostics.update({
        "eligible_spoken_blocks": direct_eligible,
        "direct_matches": direct_matches,
        "auto_accepted": auto_accepted,
        "manual_reviewed": manual_reviewed,
        "contextual_ranges": contextual,
        "abstained_spoken_blocks": max(0, direct_eligible - direct_matches),
    })
    review_queue = [
        {
            "block_id": entry["block_id"],
            "reason": "low_confidence_or_contextual_range",
            "precision": entry["precision"],
            "confidence": entry["confidence"],
        }
        for entry in ordered if entry["review_state"] == "needs_review"
    ]
    return {
        "schema_version": 1,
        "session_id": transcript_payload["session_id"],
        "youtube_video_id": transcript_payload["youtube_video_id"],
        "time_domain": "session_video",
        "aligner": {"name": ALIGNER_NAME, "version": ALIGNER_VERSION},
        "policy": {
            "authoritative_text": "official_senate_journal",
            "asr_role": "internal_alignment_aid",
            "dialogue_precision": "approximate_turn_range_not_word_exact",
            "summary_precision": "bounded_context_not_verbatim",
            "structural_text_is_spoken_query": False,
            "speaker_label_is_spoken_query": False,
            "exhibit_timestamps_are_session_seeks": False,
            "uncertain_blocks_may_be_unaligned": True,
            "automatic_score_threshold": AUTO_SCORE,
            "automatic_uniqueness_margin": AUTO_MARGIN,
        },
        "entry_count": len(ordered),
        "entries": ordered,
        "review_queue": review_queue,
        "unresolved_conflicts": conflicts,
        "diagnostics": diagnostics,
    }


__all__ = ["align"]
