"""Numeric grounding: extract quantities and classify them against the prompt.

The prompts all demand "Base analysis ONLY on provided facts", which makes
numeric grounding the single most informative automatic metric available. For
each number in a response we decide whether it is:

  ``supported``    - matches a KEY FACT for this query
  ``prompt``       - appears elsewhere in the prompt (e.g. the Few-Shot
                     examples), so it was provided, not invented
  ``derived``      - equals a simple arithmetic combination of provided
                     figures, and we record the formula that produces it
  ``contradicting``- close to a provided figure in the same dimension but
                     materially different, i.e. the fact was misstated
  ``unsupported``  - none of the above; an invented quantity

Units are normalized so ``52,000MW`` and ``52 GW`` compare equal, and ranges
like ``$80-130B`` match any value inside the interval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

Dimension = Literal[
    "power", "energy", "money", "percent", "duration",
    "count", "voltage", "temperature", "dimensionless",
]

Classification = Literal["supported", "prompt", "derived", "contradicting", "unsupported"]

# Multiplier tables, all resolving to a canonical base unit per dimension.
_POWER = {"kw": 0.001, "mw": 1.0, "gw": 1000.0, "tw": 1_000_000.0}
_ENERGY = {"kwh": 0.001, "mwh": 1.0, "gwh": 1000.0, "twh": 1_000_000.0}
_MAGNITUDE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "million": 1e6, "mn": 1e6,
    "b": 1e9, "billion": 1e9, "bn": 1e9,
    "t": 1e12, "trillion": 1e12,
}
_DURATION_HOURS = {
    "min": 1 / 60, "mins": 1 / 60, "minute": 1 / 60, "minutes": 1 / 60,
    "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
    "day": 24.0, "days": 24.0,
    "wk": 168.0, "wks": 168.0, "week": 168.0, "weeks": 168.0,
    "month": 730.0, "months": 730.0,
    "year": 8760.0, "years": 8760.0,
}
# Count nouns grouped so paraphrases compare equal.
_SUBJECTS = {
    "deaths": "deaths", "death": "deaths", "fatalities": "deaths",
    "fatality": "deaths", "lives": "deaths", "died": "deaths",
    "people": "people", "person": "people", "persons": "people",
    "residents": "people", "population": "people", "individuals": "people",
    "homes": "premises", "home": "premises", "households": "premises",
    "household": "premises", "customers": "premises", "customer": "premises",
    "premises": "premises", "meters": "premises", "businesses": "premises",
    "towers": "towers", "tower": "towers",
    "states": "states", "state": "states",
    "lines": "lines", "line": "lines",
    "plants": "plants", "plant": "plants", "units": "plants",
}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"

_UNIT_WORDS = (
    "%|percent|percentage points?|pct"
    r"|MWh|GWh|kWh|TWh"
    r"|MW|GW|kW|TW"
    r"|kV|MV"
    r"|°F|°C"
    r"|million|billion|trillion|thousand|bn|mn"
    r"|minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|years?"
    r"|deaths?|fatalities|fatality|lives|people|persons?|residents|population"
    r"|homes?|households?|customers?|premises|meters|businesses"
    r"|towers?|states?|lines?|plants?|units?"
    # Single-letter magnitudes ("4.5M", "$130B") come last so that longer
    # units such as MW, MWh and "million" win the alternation first.
    r"|[MBKT](?![A-Za-z])"
)

# "$80-130B", "6hrs-2wks", "80 to 130 billion"
_RANGE_RE = re.compile(
    rf"(?P<cur>[$€£])?\s*(?P<lo>{_NUM})\s*(?P<lounit>{_UNIT_WORDS})?"
    rf"\s*(?:-|–|—|to)\s*"
    rf"(?P<cur2>[$€£])?\s*(?P<hi>{_NUM})\s*(?P<hiunit>{_UNIT_WORDS})?",
    re.IGNORECASE,
)

_SINGLE_RE = re.compile(
    rf"(?P<cur>[$€£])?\s*(?P<sign>[-+])?\s*(?P<num>{_NUM})\s*(?P<plus>\+)?"
    rf"\s*-?\s*(?P<unit>{_UNIT_WORDS})?",
    re.IGNORECASE,
)

# A bare integer this small, with no unit, is list numbering or an ordinal.
_YEAR_RE = re.compile(r"^(1[89]\d{2}|20\d{2})$")

# Models routinely emit typographic spaces and hyphens ("$80\u2011130\u202fB").
# Normalizing these to ASCII is a 1:1 character mapping, so highlight offsets
# computed on the normalized text still line up with the original.
_CHAR_NORMALIZATION = str.maketrans(
    {
        "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u202f": " ", "\u2005": " ",
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-",
    }
)

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "hundred": 100,
}

# Spelled-out quantities, but only when a unit follows ("four days"), so that
# ordinary prose like "one of the causes" is not read as a number.
_WORD_NUM_RE = re.compile(
    rf"\b(?P<word>{'|'.join(_WORD_NUMBERS)})\s*-?\s*(?P<unit>{_UNIT_WORDS})\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """ASCII-fold typographic spaces and dashes without changing length."""
    return text.translate(_CHAR_NORMALIZATION) if text else text


@dataclass
class Quantity:
    """One numeric mention, normalized to a canonical unit."""

    raw: str
    value: float                  # canonical magnitude
    dimension: Dimension
    subject: str | None = None    # for counts: deaths / people / premises ...
    hi: float | None = None       # set when the mention is a range
    start: int = 0
    end: int = 0
    negative: bool = False

    @property
    def is_range(self) -> bool:
        return self.hi is not None

    def covers(self, value: float, tol: float) -> bool:
        """Does this quantity (possibly a range) admit ``value``?"""
        if self.is_range:
            lo, hi = min(self.value, self.hi), max(self.value, self.hi)
            pad = max(abs(lo), abs(hi)) * tol
            return (lo - pad) <= value <= (hi + pad)
        return _close(self.value, value, tol)

    def subject_compatible(self, other: "Quantity") -> bool:
        if self.subject and other.subject:
            return self.subject == other.subject
        return True

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "value": self.value,
            "dimension": self.dimension,
            "subject": self.subject,
            "hi": self.hi,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class ClassifiedQuantity:
    quantity: Quantity
    classification: Classification
    evidence: str | None = None   # matched fact text or derivation formula

    def to_dict(self) -> dict:
        d = self.quantity.to_dict()
        d["classification"] = self.classification
        d["evidence"] = self.evidence
        return d


@dataclass
class GroundingResult:
    classified: list[ClassifiedQuantity] = field(default_factory=list)
    facts_found: list[str] = field(default_factory=list)
    facts_missing: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.classified)

    def count(self, *kinds: str) -> int:
        return sum(1 for c in self.classified if c.classification in kinds)

    @property
    def supported(self) -> int:
        return self.count("supported", "prompt")

    @property
    def derived(self) -> int:
        return self.count("derived")

    @property
    def unsupported(self) -> int:
        return self.count("unsupported")

    @property
    def contradictions(self) -> int:
        return self.count("contradicting")

    @property
    def grounding_rate(self) -> float:
        """Share of quantities that are provided or legitimately derived."""
        if self.total == 0:
            return 1.0
        return (self.supported + self.derived) / self.total


def _close(a: float, b: float, tol: float) -> bool:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= tol


def _canonical_unit(unit: str | None, currency: str | None) -> tuple[Dimension, float, str | None]:
    """Map a raw unit token to (dimension, multiplier, subject)."""
    u = (unit or "").strip().lower().rstrip(".")

    if currency:
        return "money", _MAGNITUDE.get(u, 1.0), None

    if not u:
        return "dimensionless", 1.0, None
    if u in ("%", "percent", "pct", "percentage point", "percentage points"):
        return "percent", 1.0, None
    if u in _POWER:
        return "power", _POWER[u], None
    if u in _ENERGY:
        return "energy", _ENERGY[u], None
    if u in ("kv", "mv"):
        return "voltage", 1.0 if u == "kv" else 1000.0, None
    if u in ("°f", "°c"):
        return "temperature", 1.0, None
    if u in _DURATION_HOURS:
        return "duration", _DURATION_HOURS[u], None
    if u in _MAGNITUDE:
        # Bare magnitude ("4.5M") with no noun: an unlabelled count.
        return "count", _MAGNITUDE[u], None
    if u in _SUBJECTS:
        return "count", 1.0, _SUBJECTS[u]

    return "dimensionless", 1.0, None


def _parse_value(text: str) -> float:
    return float(text.replace(",", ""))


def _magnitude_from_context(text: str, unit_end: int) -> tuple[float, str | None]:
    """Pick up a magnitude word and/or count noun following a number.

    Handles ``4.5 million homes`` and ``$80 billion`` where the multiplier and
    the noun are separate tokens.
    """
    tail = text[unit_end : unit_end + 32].lower()
    mult = 1.0
    subject = None
    m = re.match(r"\s*(million|billion|trillion|thousand|bn|mn)\b", tail)
    if m:
        mult = _MAGNITUDE[m.group(1)]
        tail = tail[m.end() :]
    m = re.match(r"[\s-]*([a-z]+)", tail)
    if m and m.group(1) in _SUBJECTS:
        subject = _SUBJECTS[m.group(1)]
    return mult, subject


def extract_quantities(text: str) -> list[Quantity]:
    """Pull every meaningful quantity out of ``text``."""
    if not text:
        return []

    text = normalize_text(text)
    quantities: list[Quantity] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < ce and e > cs for cs, ce in consumed)

    # Ranges first so "80-130B" is not read as two separate numbers.
    for m in _RANGE_RE.finditer(text):
        lo_raw, hi_raw = m.group("lo"), m.group("hi")
        unit = m.group("hiunit") or m.group("lounit")
        currency = m.group("cur") or m.group("cur2")
        if not unit and not currency:
            continue  # bare "3-5" is not a quantity we can reason about
        dim, mult, subject = _canonical_unit(unit, currency)
        extra, ctx_subject = _magnitude_from_context(text, m.end())
        if dim == "money" and mult == 1.0:
            mult = extra
        lo = _parse_value(lo_raw) * mult
        hi = _parse_value(hi_raw) * mult
        if dim == "dimensionless":
            continue
        quantities.append(
            Quantity(
                raw=m.group(0).strip(),
                value=lo,
                hi=hi,
                dimension=dim,
                subject=subject or ctx_subject,
                start=m.start(),
                end=m.end(),
            )
        )
        consumed.append((m.start(), m.end()))

    for m in _SINGLE_RE.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        num_raw = m.group("num")
        unit = m.group("unit")
        currency = m.group("cur")
        dim, mult, subject = _canonical_unit(unit, currency)

        extra, ctx_subject = _magnitude_from_context(text, m.end())
        if dim in ("money", "count") and mult == 1.0 and extra != 1.0:
            mult = extra
        elif dim == "dimensionless" and extra != 1.0:
            dim, mult = "count", extra
        subject = subject or ctx_subject

        value = _parse_value(num_raw)

        # Drop list numbering, ordinals and bare years.
        if dim == "dimensionless" and not subject:
            continue
        if dim == "count" and mult == 1.0 and not subject and _YEAR_RE.match(num_raw):
            continue

        quantities.append(
            Quantity(
                raw=m.group(0).strip(),
                value=value * mult,
                dimension=dim,
                subject=subject,
                start=m.start(),
                end=m.end(),
                negative=m.group("sign") == "-",
            )
        )
        consumed.append((m.start(), m.end()))

    # Spelled-out quantities ("four days", "nine minutes") last, so a digit
    # form at the same position always wins.
    for m in _WORD_NUM_RE.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        dim, mult, subject = _canonical_unit(m.group("unit"), None)
        if dim == "dimensionless":
            continue
        _, ctx_subject = _magnitude_from_context(text, m.end())
        quantities.append(
            Quantity(
                raw=m.group(0).strip(),
                value=_WORD_NUMBERS[m.group("word").lower()] * mult,
                dimension=dim,
                subject=subject or ctx_subject,
                start=m.start(),
                end=m.end(),
            )
        )
        consumed.append((m.start(), m.end()))

    return sorted(quantities, key=lambda q: q.start)


def _derivations(refs: list[Quantity]) -> list[tuple[float, Dimension, str]]:
    """Simple arithmetic combinations of provided figures.

    Covers the operations these prompts actually invite: summing or
    differencing like quantities, applying a percentage, inverting a
    percentage to recover a total, and expressing one figure as a share of
    another.
    """
    out: list[tuple[float, Dimension, str]] = []
    scalars = [r for r in refs if not r.is_range]

    for a in scalars:
        for b in scalars:
            if a is b:
                continue
            av, bv = a.value, b.value

            if a.dimension == b.dimension and a.dimension != "percent":
                out.append((av + bv, a.dimension, f"{a.raw} + {b.raw}"))
                if av > bv:
                    out.append((av - bv, a.dimension, f"{a.raw} - {b.raw}"))
                if bv:
                    out.append((av / bv * 100, "percent", f"{a.raw} / {b.raw} x 100"))
                    out.append((av / bv, "dimensionless", f"{a.raw} / {b.raw}"))

            if b.dimension == "percent" and bv:
                out.append((av * bv / 100, a.dimension, f"{b.raw} of {a.raw}"))
                out.append((av / (bv / 100), a.dimension, f"{a.raw} / {b.raw}"))

            if a.dimension != "duration" and b.dimension == "duration" and bv:
                out.append((av / bv, "dimensionless", f"{a.raw} per {b.raw}"))
    return out


def build_reference(key_facts: str, prompt_text: str) -> tuple[list[Quantity], list[Quantity]]:
    """Quantities from the KEY FACTS block, and from the whole prompt.

    The second list matters for the Few-Shot prompts, which supply comparative
    examples (Puerto Rico, South Australia). Citing those is following
    instructions, not hallucinating, so they count as provided.
    """
    return extract_quantities(key_facts), extract_quantities(prompt_text)


def classify(
    text: str,
    key_facts: str,
    prompt_text: str,
    *,
    match_tol: float = 0.01,
    contradiction_window: float = 0.25,
) -> GroundingResult:
    """Classify every quantity in ``text`` against the prompt's figures."""
    fact_refs, prompt_refs = build_reference(key_facts, prompt_text)
    derived = _derivations(fact_refs)
    found = extract_quantities(text)

    result = GroundingResult()
    matched_fact_ids: set[int] = set()

    for q in found:
        cls: Classification = "unsupported"
        evidence: str | None = None

        # 1. Matches a KEY FACT.
        for idx, ref in enumerate(fact_refs):
            if ref.dimension != q.dimension or not ref.subject_compatible(q):
                continue
            if ref.covers(q.value, match_tol) or (
                q.is_range and q.covers(ref.value, match_tol)
            ):
                cls, evidence = "supported", ref.raw
                matched_fact_ids.add(idx)
                break

        # 2. Provided elsewhere in the prompt (few-shot examples, role framing).
        if cls == "unsupported":
            for ref in prompt_refs:
                if ref.dimension != q.dimension or not ref.subject_compatible(q):
                    continue
                if ref.covers(q.value, match_tol):
                    cls, evidence = "prompt", f"provided in prompt: {ref.raw}"
                    break

        # 3. Derivable from the KEY FACTS.
        if cls == "unsupported":
            for value, dim, formula in derived:
                if dim != q.dimension:
                    continue
                if q.covers(value, match_tol):
                    cls, evidence = "derived", formula
                    break

        # 4. Near-miss on a provided figure => the fact was misstated.
        if cls == "unsupported":
            best: tuple[float, Quantity] | None = None
            for ref in fact_refs:
                if ref.dimension != q.dimension or not ref.subject_compatible(q):
                    continue
                if ref.is_range or q.is_range:
                    continue
                scale = max(abs(ref.value), abs(q.value), 1e-9)
                rel = abs(ref.value - q.value) / scale
                if rel <= contradiction_window and (best is None or rel < best[0]):
                    best = (rel, ref)
            if best is not None:
                cls = "contradicting"
                evidence = (
                    f"differs from provided figure {best[1].raw}"
                    f" by {best[0] * 100:.1f}%"
                )

        result.classified.append(ClassifiedQuantity(q, cls, evidence))

    for idx, ref in enumerate(fact_refs):
        (result.facts_found if idx in matched_fact_ids else result.facts_missing).append(
            ref.raw
        )

    return result


def fact_recall(result: GroundingResult) -> float:
    total = len(result.facts_found) + len(result.facts_missing)
    if total == 0:
        return 1.0
    return len(result.facts_found) / total
