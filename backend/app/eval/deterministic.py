"""Deterministic, reproducible metrics computed without any model calls.

These form the backbone of the evaluation: they are free, instant, and cannot
drift the way a judge model can. Each returns a 0-1 score plus the supporting
detail needed to justify it in the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import facts as facts_mod
from .spec import PromptSpec

# Markdown/heading noise stripped before comparing headings.
_HEADING_CLEAN = re.compile(r"^[\s#*_>`\-\u2022\d.):]+|[\s#*_`:]+$")
_MD_EMPHASIS = re.compile(r"[*_`]+")


@dataclass
class MetricResult:
    scores: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.scores.get(key, default)


def _candidate_headings(text: str) -> list[tuple[int, str]]:
    """Lines that plausibly act as headings, with their line index."""
    out: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines()):
        raw = line.strip()
        if not raw or len(raw) > 160:
            continue
        looks_like_heading = (
            raw.startswith("#")
            or re.match(r"^\**\s*\d{1,2}[.)]\s+\S", raw)
            or (raw.startswith("**") and raw.rstrip().endswith("**"))
            or (raw.endswith(":") and len(raw) < 90)
            or (raw.isupper() and len(raw) > 3)
        )
        if not looks_like_heading:
            continue
        cleaned = _MD_EMPHASIS.sub("", _HEADING_CLEAN.sub("", raw)).strip()
        if cleaned:
            out.append((i, cleaned.lower()))
    return out


def score_structure(text: str, spec: PromptSpec) -> tuple[float, dict[str, Any]]:
    """Fraction of required sections present, plus whether order was kept.

    A section counts as present when a heading-like line carries enough of its
    distinctive title words, or (fallback) when those words appear together
    anywhere in the body.
    """
    required = spec.required_sections
    if not required:
        return 1.0, {"required": 0, "found": 0, "order_ok": True, "missing": []}

    headings = _candidate_headings(text)
    body = text.lower()

    found_positions: dict[int, int] = {}
    missing: list[str] = []

    for section in required:
        keywords = section.keywords
        if not keywords:
            continue
        # Require a majority of the title's content words, at least one.
        need = max(1, round(len(keywords) * 0.5))

        best_line: int | None = None
        for line_no, heading in headings:
            hits = sum(1 for kw in keywords if kw in heading)
            if hits >= need:
                best_line = line_no
                break

        if best_line is None:
            hits = sum(1 for kw in keywords if kw in body)
            if hits >= max(2, need):
                # Present in prose but not as a heading: counts, ordering unknown.
                best_line = -1

        if best_line is None:
            missing.append(f"{section.number}. {section.title}")
        else:
            found_positions[section.number] = best_line

    found = len(found_positions)
    coverage = found / len(required)

    ordered = [
        found_positions[s.number]
        for s in required
        if s.number in found_positions and found_positions[s.number] >= 0
    ]
    order_ok = all(a <= b for a, b in zip(ordered, ordered[1:]))

    return coverage, {
        "required": len(required),
        "found": found,
        "order_ok": order_ok,
        "missing": missing,
    }


def score_length(word_count: int, spec: PromptSpec, overflow_zero_multiple: float) -> tuple[float, dict[str, Any]]:
    """1.0 at or under the cap, decaying linearly to 0 at cap * multiple.

    Also penalizes grossly short answers, which are usually truncation or a
    refusal to do the work rather than admirable concision.
    """
    limit = spec.word_limit
    if not limit:
        return 1.0, {"word_limit": None, "word_count": word_count, "ratio": None}

    ratio = word_count / limit
    if word_count <= limit:
        # Under a quarter of the requested length is not a real report.
        score = 1.0 if ratio >= 0.25 else max(0.0, ratio / 0.25)
    else:
        span = max(overflow_zero_multiple - 1.0, 1e-6)
        score = max(0.0, 1.0 - (ratio - 1.0) / span)

    return round(score, 4), {
        "word_limit": limit,
        "word_count": word_count,
        "ratio": round(ratio, 3),
        "over_limit": word_count > limit,
    }


def score_required_tokens(text: str, spec: PromptSpec) -> tuple[float, dict[str, Any]]:
    """Whether markers the prompt explicitly demanded actually appear."""
    required = spec.required_tokens
    if not required:
        return 1.0, {"required": [], "present": [], "missing": []}

    lowered = text.lower()
    present, missing = [], []
    for token in required:
        needle = token.rstrip(":").lower()
        (present if needle in lowered else missing).append(token)

    return len(present) / len(required), {
        "required": required,
        "present": present,
        "missing": missing,
    }


def score_format(text: str, reasoning_leak: bool, truncated: bool) -> tuple[float, dict[str, Any]]:
    """Penalize leaked reasoning scaffolding, truncation and empty output."""
    problems: list[str] = []
    score = 1.0

    if reasoning_leak:
        problems.append("inline <think> reasoning leaked into the answer")
        score -= 0.5
    if truncated:
        problems.append("response hit the token ceiling and was cut off")
        score -= 0.3
    if not text.strip():
        problems.append("empty response")
        score = 0.0
    elif len(text.split()) < 50:
        problems.append("response too short to be a report")
        score -= 0.3

    if re.search(r"\b(as an ai|i cannot|i'm unable to)\b", text, re.IGNORECASE):
        problems.append("meta-commentary or refusal language")
        score -= 0.2

    return max(0.0, round(score, 4)), {"problems": problems}


def evaluate(
    text: str,
    spec: PromptSpec,
    *,
    reasoning_leak: bool = False,
    truncated: bool = False,
    match_tol: float = 0.01,
    contradiction_window: float = 0.25,
    overflow_zero_multiple: float = 3.0,
) -> MetricResult:
    """Run every deterministic metric for one response."""
    grounding = facts_mod.classify(
        text,
        spec.key_facts_raw,
        spec.text,
        match_tol=match_tol,
        contradiction_window=contradiction_window,
    )

    recall = facts_mod.fact_recall(grounding)
    word_count = len(text.split())

    structure, structure_detail = score_structure(text, spec)
    length, length_detail = score_length(word_count, spec, overflow_zero_multiple)
    tokens, tokens_detail = score_required_tokens(text, spec)
    fmt, fmt_detail = score_format(text, reasoning_leak, truncated)

    # Any contradiction of a supplied fact is severe; three or more zeroes it.
    contradiction_free = max(0.0, 1.0 - grounding.contradictions / 3.0)

    scores = {
        "fact_recall": round(recall, 4),
        "numeric_grounding": round(grounding.grounding_rate, 4),
        "contradiction_free": round(contradiction_free, 4),
        "structural_compliance": round(structure, 4),
        "length_compliance": length,
        "required_tokens": round(tokens, 4),
        "format_clean": fmt,
    }

    detail = {
        "facts": {
            "found": grounding.facts_found,
            "missing": grounding.facts_missing,
            "total": len(grounding.facts_found) + len(grounding.facts_missing),
        },
        "numbers": {
            "total": grounding.total,
            "supported": grounding.supported,
            "derived": grounding.derived,
            "unsupported": grounding.unsupported,
            "contradicting": grounding.contradictions,
            "items": [c.to_dict() for c in grounding.classified],
        },
        "structure": structure_detail,
        "length": length_detail,
        "required_tokens": tokens_detail,
        "format": fmt_detail,
        "word_count": word_count,
    }

    return MetricResult(scores=scores, detail=detail)


def deterministic_composite(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted blend of the deterministic scores only (judge excluded)."""
    usable = {k: w for k, w in weights.items() if k in scores}
    total = sum(usable.values())
    if total <= 0:
        return 0.0
    return round(sum(scores[k] * w for k, w in usable.items()) / total, 4)
