"""Derive the grading specification for a prompt from the prompt text itself.

Every prompt in this benchmark carries its own contract:

  * a ``[KEY FACTS: ...]`` block  -> the reference facts the answer must stick to
  * ``Maximum length: N words``   -> the length constraint
  * a numbered list after ``with the following structure:`` -> required sections
  * literal markers such as ``VERIFY_SOURCE`` / ``INFERRED:`` -> required tokens

Parsing the contract out of the prompt keeps the graders in sync with the
prompts automatically, instead of relying on hand-maintained fixtures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TECHNIQUE_PATTERNS: list[tuple[str, str]] = [
    (r"chain[- ]of[- ]thought", "CoT"),
    (r"tree[- ]of[- ]thought", "ToT"),
    (r"role\s+prompting", "Role"),
    (r"few[- ]shot", "Few-Shot"),
    (r"react\s+prompting", "ReAct"),
]

# Markers the prompt may demand verbatim in the output.
TOKEN_REQUIREMENTS: list[tuple[str, str]] = [
    ("VERIFY_SOURCE", r"VERIFY_SOURCE"),
    ("INFERRED:", r"INFERRED:?"),
    ("evidence", r"'evidence'|\"evidence\""),
    ("uncertainties", r"'uncertainties'|\"uncertainties\""),
]

_STRUCTURE_MARKER = re.compile(r"following\s+structure\s*:", re.IGNORECASE)
_SECTION_LINE = re.compile(r"^\s*(\d{1,2})\.\s+(.+?)\s*$")
_WORD_LIMIT = re.compile(r"maximum\s+length\s*:\s*([\d,]+)\s*words", re.IGNORECASE)
_KEY_FACTS = re.compile(r"\[KEY FACTS:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
_STEP_LINE = re.compile(r"^\s*(?:step\s*)?(\d{1,2})\)\s*(.+?)\s*$", re.IGNORECASE)
_PATH_LINE = re.compile(r"^\s*(?:PATH|BRANCH)\s+([A-Z])\)\s*(.+?)\s*$", re.IGNORECASE)
_SUB_BULLET = re.compile(r"^\s+[-*\u2022]\s+")


@dataclass
class RequiredSection:
    number: int
    title: str

    @property
    def keywords(self) -> list[str]:
        """Content words from the title, used for fuzzy heading matching."""
        base = re.sub(r"\(.*?\)", " ", self.title)
        words = re.findall(r"[A-Za-z][A-Za-z/\-]+", base)
        stop = {
            "and", "the", "with", "for", "of", "to", "a", "an", "in", "on",
            "its", "their", "by", "from", "or", "as", "at",
        }
        return [w.lower() for w in words if w.lower() not in stop]


@dataclass
class PromptSpec:
    prompt_id: str          # e.g. "Q1/prompt 1a"
    query_id: str           # "Q1" | "Q2"
    technique: str          # CoT | ToT | Role | Few-Shot | ReAct
    variant: str            # "1a", "2c", ...
    path: Path
    text: str
    key_facts_raw: str
    required_sections: list[RequiredSection] = field(default_factory=list)
    word_limit: int | None = None
    required_tokens: list[str] = field(default_factory=list)
    analytical_steps: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.variant.upper()} {self.technique}"

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "query_id": self.query_id,
            "technique": self.technique,
            "variant": self.variant,
            "word_limit": self.word_limit,
            "required_sections": [
                {"number": s.number, "title": s.title} for s in self.required_sections
            ],
            "required_tokens": self.required_tokens,
            "analytical_steps": self.analytical_steps,
            "key_facts_raw": self.key_facts_raw,
        }


def _detect_technique(text: str) -> str:
    head = text[:400].lower()
    for pattern, name in TECHNIQUE_PATTERNS:
        if re.search(pattern, head):
            return name
    for pattern, name in TECHNIQUE_PATTERNS:
        if re.search(pattern, text.lower()):
            return name
    return "Unknown"


def _detect_variant(text: str, filename: str) -> str:
    m = re.search(r"\b(\d)([A-Ea-e])\s*\.", text[:200])
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    m = re.search(r"(\d)\s*([a-e])", filename.lower())
    return f"{m.group(1)}{m.group(2)}" if m else filename


def _parse_required_sections(text: str) -> list[RequiredSection]:
    """Take the contiguous ``N. Title`` block following the structure marker.

    Scoped to that marker so the ``N)`` analytical steps earlier in the prompt
    are not mistaken for report sections.
    """
    marker = _STRUCTURE_MARKER.search(text)
    if not marker:
        return []

    sections: list[RequiredSection] = []
    expected = 1
    for line in text[marker.end():].splitlines():
        if not line.strip():
            # Blank lines inside the list are tolerated only before it starts.
            if sections:
                break
            continue
        # Indented sub-bullets elaborate the section above them (the Few-Shot
        # prompts do this under "Texas Event Analysis"); they are not sections.
        if _SUB_BULLET.match(line):
            continue
        m = _SECTION_LINE.match(line)
        if not m:
            if sections:
                break
            continue
        number = int(m.group(1))
        if number != expected:
            break
        sections.append(RequiredSection(number=number, title=m.group(2)))
        expected += 1
    return sections


def _parse_analytical_steps(text: str) -> list[str]:
    marker = _STRUCTURE_MARKER.search(text)
    scope = text[: marker.start()] if marker else text
    steps: list[str] = []
    for line in scope.splitlines():
        m = _STEP_LINE.match(line)
        if m:
            steps.append(m.group(2).strip())
            continue
        m = _PATH_LINE.match(line)
        if m:
            steps.append(f"PATH {m.group(1).upper()}: {m.group(2).strip()}")
    return steps


def _parse_required_tokens(text: str) -> list[str]:
    found = []
    for label, pattern in TOKEN_REQUIREMENTS:
        if re.search(pattern, text):
            found.append(label)
    return found


def parse_prompt_file(path: Path, prompts_root: Path) -> PromptSpec:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(prompts_root)
    prompt_id = f"{rel.parent.as_posix()}/{path.stem}"

    key_facts = _KEY_FACTS.search(text)
    limit = _WORD_LIMIT.search(text)

    return PromptSpec(
        prompt_id=prompt_id,
        query_id=rel.parent.as_posix(),
        technique=_detect_technique(text),
        variant=_detect_variant(text, path.stem),
        path=path,
        text=text,
        key_facts_raw=key_facts.group(1).strip() if key_facts else "",
        required_sections=_parse_required_sections(text),
        word_limit=int(limit.group(1).replace(",", "")) if limit else None,
        required_tokens=_parse_required_tokens(text),
        analytical_steps=_parse_analytical_steps(text),
    )


def load_prompt_specs(prompts_root: str | Path) -> list[PromptSpec]:
    """Load every prompt under ``prompts_root``, sorted by query then variant."""
    root = Path(prompts_root)
    specs = [
        parse_prompt_file(p, root)
        for p in sorted(root.rglob("*.txt"))
        if p.is_file()
    ]
    return sorted(specs, key=lambda s: (s.query_id, s.variant))


def specs_by_id(prompts_root: str | Path) -> dict[str, PromptSpec]:
    return {s.prompt_id: s for s in load_prompt_specs(prompts_root)}
