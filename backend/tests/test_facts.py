"""Checks for the numeric grounding engine.

Run: .venv/bin/python backend/tests/test_facts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.facts import classify, extract_quantities, fact_recall

FACTS_Q1 = (
    "52,000MW peak offline (65% of grid capacity), 4.5M homes lost power, "
    "natural gas wellheads/pipelines froze, lasted 4+ days, "
    "economic loss $80-130B, 246 deaths"
)
PROMPT_Q1 = "Texas 2021 Winter Storm Blackout [KEY FACTS: " + FACTS_Q1 + "]. Maximum length: 800 words."

failures: list[str] = []


def check(label: str, got, want) -> None:
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r} want {want!r}")
    else:
        print(f"  ok   {label} = {got!r}")


def cls_of(text: str, raw_startswith: str):
    r = classify(text, FACTS_Q1, PROMPT_Q1)
    for c in r.classified:
        if c.quantity.raw.replace(" ", "").startswith(raw_startswith.replace(" ", "")):
            return c.classification
    return f"<not extracted from {text!r}>"


print("\n== unit normalization ==")
q = extract_quantities("52,000MW offline")[0]
check("52,000MW -> 52000 power", (q.value, q.dimension), (52000.0, "power"))
q = extract_quantities("52 GW offline")[0]
check("52 GW -> 52000 power", (q.value, q.dimension), (52000.0, "power"))
q = extract_quantities("4.5M homes")[0]
check("4.5M homes", (q.value, q.dimension, q.subject), (4_500_000.0, "count", "premises"))
q = extract_quantities("4.5 million households")[0]
check("4.5 million households", (q.value, q.subject), (4_500_000.0, "premises"))
q = extract_quantities("$90B damage")[0]
check("$90B", (q.value, q.dimension), (90e9, "money"))
q = extract_quantities("$6 billion")[0]
check("$6 billion", (q.value, q.dimension), (6e9, "money"))
q = extract_quantities("246 deaths")[0]
check("246 deaths", (q.value, q.subject), (246.0, "deaths"))
q = extract_quantities("a 9-minute cascade")[0]
check("9-minute -> hours", (round(q.value, 4), q.dimension), (0.15, "duration"))
q = extract_quantities("345kV lines")[0]
check("345kV", (q.value, q.dimension), (345.0, "voltage"))
q = extract_quantities("65% of capacity")[0]
check("65%", (q.value, q.dimension), (65.0, "percent"))

print("\n== ranges ==")
qs = extract_quantities("economic loss $80-130B")
check("$80-130B is one range", len(qs), 1)
check("range bounds", (qs[0].value, qs[0].hi), (80e9, 130e9))
check("$100B inside range", qs[0].covers(100e9, 0.02), True)
check("$200B outside range", qs[0].covers(200e9, 0.02), False)

print("\n== noise is ignored ==")
check("bare list number", extract_quantities("1. Executive Summary"), [])
check("bare year", extract_quantities("in 2021 the grid"), [])
check("section refs", extract_quantities("see 3 and 4 below"), [])

print("\n== classification ==")
check("exact fact", cls_of("The grid lost 52,000MW of capacity.", "52,000MW"), "supported")
check("unit-converted fact", cls_of("The grid lost 52 GW.", "52 GW"), "supported")
check("range midpoint", cls_of("Losses reached $100B.", "$100B"), "supported")
check("paraphrased subject", cls_of("4.5M customers lost power.", "4.5M"), "supported")
check("misstated fact", cls_of("The grid lost 53,000MW.", "53,000MW"), "contradicting")
check("invented figure", cls_of("Wind added 25,000MW of shortfall.", "25,000MW"), "unsupported")
check(
    "derived total capacity",
    cls_of("Total capacity was therefore 80,000MW.", "80,000MW"),
    "derived",
)
check(
    "derived percentage",
    cls_of("That is 5.3% of the 4.5M homes.", "5.3%"),
    "unsupported",
)

print("\n== few-shot example figures count as provided ==")
# $90B is deliberately not asserted here: it falls inside the Texas
# $80-130B range, so matching it as a KEY FACT is also correct.
FEWSHOT_PROMPT = (
    "EXAMPLE 1 - Puerto Rico 2017: Impact: 1.5M customers, 11 months recovery. "
    "NOW ANALYZE: Texas [KEY FACTS: " + FACTS_Q1 + "]."
)
r = classify(
    "Unlike Puerto Rico's 1.5M customers and 11 months recovery, Texas lost 52,000MW.",
    FACTS_Q1,
    FEWSHOT_PROMPT,
)
by_raw = {c.quantity.raw.replace(" ", ""): c.classification for c in r.classified}
check("example 1.5M is provided", by_raw.get("1.5M"), "prompt")
check("example 11 months is provided", by_raw.get("11months"), "prompt")
check("target fact still supported", by_raw.get("52,000MW"), "supported")
check(
    "few-shot figures are not counted as hallucinations",
    r.unsupported,
    0,
)

print("\n== recall and rates ==")
good = classify(
    "52,000MW offline, 65% of capacity, 4.5M homes, 4+ days, $80-130B, 246 deaths.",
    FACTS_Q1,
    PROMPT_Q1,
)
check("recall on full restatement", round(fact_recall(good), 3), 1.0)
check("no contradictions", good.contradictions, 0)
check("grounding rate 1.0", round(good.grounding_rate, 3), 1.0)

bad = classify("The outage took out 12,345MW and killed 999 people.", FACTS_Q1, PROMPT_Q1)
check("low recall", fact_recall(bad) < 0.2, True)
check("has unsupported", bad.unsupported >= 1, True)

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all fact-engine checks passed")
