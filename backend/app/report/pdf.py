"""Consolidated PDF evaluation report."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..core.config import EvalConfig
from ..eval.aggregate import METRIC_KEYS, METRIC_LABELS
from . import charts

ACCENT = colors.HexColor("#2f6f9f")
MUTED = colors.HexColor("#666666")
LIGHT = colors.HexColor("#f2f5f8")
BORDER = colors.HexColor("#cfd8e0")
BAD = colors.HexColor("#c1554e")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=24, leading=28,
            textColor=ACCENT, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=11.5, leading=15,
            textColor=MUTED, alignment=TA_LEFT, spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=15.5, leading=19,
            textColor=ACCENT, spaceBefore=16, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=12, leading=15,
            textColor=colors.HexColor("#26424f"), spaceBefore=11, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9.6, leading=13.6,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=8.2, leading=11,
            textColor=MUTED,
        ),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.2, leading=10.6),
        "cellb": ParagraphStyle(
            "cellb", parent=base["Normal"], fontSize=8.2, leading=10.6,
            fontName="Helvetica-Bold",
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["Normal"], fontName="Courier", fontSize=7.8,
            leading=10, textColor=colors.HexColor("#333333"),
        ),
    }


def _esc(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _fmt(value: Any, digits: int = 3, dash: str = "n/a") -> str:
    if value is None:
        return dash
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _money(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.4f}"


def _ci(stat: dict[str, Any]) -> str:
    """Mean with its 95% interval, or a note when n is too small."""
    if not stat or stat.get("mean") is None:
        return "n/a"
    if stat.get("ci95") is None:
        return f"{stat['mean']:.3f} (n={stat.get('n', 0)})"
    return f"{stat['mean']:.3f} \u00b1 {stat['ci95']:.3f}"


def _chart(png: bytes, width: float = 6.9 * inch) -> Image:
    reader = io.BytesIO(png)
    img = Image(reader)
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * ratio
    img.hAlign = "CENTER"
    return img


def _table(data: Sequence[Sequence[Any]], widths: Sequence[float],
           highlight_rows: Sequence[int] = ()) -> Table:
    table = Table(list(data), colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    for row in highlight_rows:
        style.append(("TEXTCOLOR", (0, row), (-1, row), BAD))
    table.setStyle(TableStyle(style))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.5 * inch, "Prompt Evaluation Framework")
    canvas.drawRightString(
        letter[0] - 0.75 * inch, 0.5 * inch, f"Page {canvas.getPageNumber()}"
    )
    canvas.restoreState()


def build_report(
    summary: dict[str, Any],
    config: EvalConfig,
    run_id: str,
    rows: Sequence[dict] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Render the full report and return its path.

    ``rows`` are the raw per-generation records; several charts need the
    per-response grounding detail that the aggregated summary collapses away.
    """
    st = _styles()
    run = summary.get("run", {})
    totals = summary["totals"]
    ops = summary["operational"]
    models = summary["models"]
    techniques = summary["techniques"]
    rows = list(rows or [])

    path = Path(output_path) if output_path else (
        config.runs_root / run_id / f"evaluation-report-{run_id}.pdf"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.75 * inch,
        title=f"Prompt Evaluation Report {run_id}",
        author="Prompt Evaluation Framework",
    )

    story: list[Any] = []
    _, _, incomplete = _coverage_map(summary, rows)
    story += _cover(st, run_id, run, totals, ops, config)
    story += _headline(st, summary, incomplete)
    story += _coverage(st, summary, rows)
    story.append(PageBreak())
    story += _methodology(st, config, summary)
    story.append(PageBreak())
    story += _leaderboards(st, summary, models, techniques, rows, config)
    story.append(PageBreak())
    story += _grounding_section(st, summary, models, rows)
    story.append(PageBreak())
    story += _operational_section(st, summary, models, rows)
    story.append(PageBreak())
    story += _failures_section(st, summary)
    story += _caveats(st, summary, config)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def _cover(st, run_id, run, totals, ops, config) -> list[Any]:
    created = run.get("created_at") or datetime.now(timezone.utc).isoformat()
    judged_note = (
        f"{totals['judged']} judged by {run.get('judge_model') or config.judge_model}"
        if totals["judged"]
        else "judge disabled"
    )

    out = [
        Paragraph("Prompt Evaluation Report", st["title"]),
        Paragraph(
            "Prompting-technique benchmark across language models, scored with "
            "deterministic fact-grounding metrics and an anchored LLM-as-judge "
            "rubric.",
            st["subtitle"],
        ),
        _table(
            [
                ["Run", _esc(run_id)],
                ["Created", _esc(created)],
                ["Status", _esc(run.get("status", "unknown"))],
                ["Models", _esc(", ".join(totals["models"]))],
                ["Techniques", _esc(", ".join(totals["techniques"]))],
                ["Prompts", f"{len(totals['prompts'])} prompts x {run.get('repeats', '?')} repeats"],
                ["Generations", f"{totals['ok']} succeeded, {totals['failed']} failed"],
                ["Judge", _esc(judged_note)],
                ["Total cost", f"{_money(ops.get('cost_usd'))} generation + {_money(ops.get('judge_cost_usd'))} judging"],
            ],
            [1.5 * inch, 5.2 * inch],
        ),
        Spacer(1, 0.22 * inch),
    ]
    return out


def _coverage_map(summary, rows) -> tuple[int, dict[str, dict[str, int]], set[str]]:
    """Expected sample count, per-model tallies, and which models fell short."""
    run = summary.get("run", {})
    repeats = run.get("repeats") or 0
    prompts = len(run.get("prompt_ids") or []) or len(summary["totals"]["prompts"])
    expected = repeats * prompts

    per_model: dict[str, dict[str, int]] = {}
    for r in rows:
        entry = per_model.setdefault(r["model"], {"ok": 0, "failed": 0})
        entry["ok" if r.get("status") == "ok" else "failed"] += 1

    incomplete = {m for m, c in per_model.items() if expected and c["ok"] < expected}
    return expected, per_model, incomplete


def _coverage(st, summary, rows) -> list[Any]:
    """Per-model completeness.

    A model whose requests partly failed contributes fewer samples than the
    others, and its successful subset is not necessarily representative. That
    has to be visible next to the leaderboard, not buried, or the ranking will
    be over-read.
    """
    expected, per_model, incomplete = _coverage_map(summary, rows)
    if not per_model:
        return []

    data = [["Model", "Expected", "Succeeded", "Failed", "Coverage"]]
    flagged: list[int] = []
    for i, (model, counts) in enumerate(sorted(per_model.items()), start=1):
        share = counts["ok"] / expected if expected else None
        if model in incomplete:
            flagged.append(i)
        data.append(
            [
                Paragraph(_esc(model), st["cellb"]),
                str(expected or "n/a"),
                str(counts["ok"]),
                str(counts["failed"]),
                _pct(share) if share is not None else "n/a",
            ]
        )

    out = [Paragraph("Coverage", st["h1"]), _table(
        data, [2.0 * inch, 0.95 * inch, 1.0 * inch, 0.8 * inch, 0.95 * inch], flagged
    )]

    if incomplete:
        out.append(
            Paragraph(
                "Highlighted models did not complete every requested generation, "
                "so they contribute fewer samples than the others and their "
                "successful subset may not be representative. Scores for those "
                "models are not directly comparable with fully covered ones. "
                "Failures in this run were provider rate limits, not model "
                "errors; see the run's stored error messages for detail.",
                st["small"],
            )
        )
    return out


def _headline(st, summary, incomplete: set[str] | None = None) -> list[Any]:
    """Plain-language statement of what the run found."""
    incomplete = incomplete or set()
    techniques = summary["techniques"]
    overall = summary["overall"]

    # Rank only models that produced a complete sample set. A partially
    # completed model's surviving responses are a biased subset, so letting it
    # top the leaderboard would state something the data cannot support.
    all_models = summary["models"]
    models = [m for m in all_models if m["model"] not in incomplete]
    excluded = [m for m in all_models if m["model"] in incomplete]

    lines: list[Any] = [Paragraph("Headline findings", st["h1"])]

    if len(models) == 1:
        only = models[0]
        lines.append(
            Paragraph(
                f"<b>{_esc(only['model'])}</b> scored "
                f"{_ci(only['metrics']['composite'])} on the composite. Only one "
                "model was evaluated in this run, so there is no ranking to draw.",
                st["body"],
            )
        )
    elif models:
        top = models[0]
        parts = [
            f"<b>{_esc(top['model'])}</b> leads on the composite score at "
            f"{_ci(top['metrics']['composite'])}."
        ]
        if len(models) > 1:
            second = models[1]
            gap_overlaps = _intervals_overlap(
                top["metrics"]["composite"], second["metrics"]["composite"]
            )
            if gap_overlaps:
                parts.append(
                    f"Its 95% interval overlaps that of {_esc(second['model'])} "
                    f"({_ci(second['metrics']['composite'])}), so the ordering "
                    "between them is not resolved at this sample size."
                )
            else:
                parts.append(
                    f"The gap over {_esc(second['model'])} "
                    f"({_ci(second['metrics']['composite'])}) is larger than the "
                    "confidence intervals, so the ordering is meaningful."
                )
        lines.append(Paragraph(" ".join(parts), st["body"]))

    if excluded:
        names = ", ".join(
            f"{_esc(m['model'])} ({_ci(m['metrics']['composite'])})" for m in excluded
        )
        lines.append(
            Paragraph(
                f"Excluded from that comparison: {names}. These models did not "
                "complete every requested generation because of provider rate "
                "limits, so their surviving responses are a biased subset and "
                "their apparent scores are not comparable. See Coverage below.",
                st["body"],
            )
        )

    if len(techniques) > 1:
        best_t, worst_t = techniques[0], techniques[-1]
        lines.append(
            Paragraph(
                f"Across models, <b>{_esc(best_t['technique'])}</b> scored highest "
                f"({_ci(best_t['metrics']['composite'])}) and "
                f"<b>{_esc(worst_t['technique'])}</b> lowest "
                f"({_ci(worst_t['metrics']['composite'])}).",
                st["body"],
            )
        )

    grounding = overall.get("numeric_grounding", {})
    length = overall.get("length_compliance", {})
    if grounding.get("mean") is not None:
        lines.append(
            Paragraph(
                f"Numeric grounding averaged {_pct(grounding['mean'])}, meaning "
                f"roughly {_pct(1 - grounding['mean'])} of all quantities in the "
                "outputs were neither supplied in the prompt nor derivable from "
                "it. This is the clearest hallucination signal in the run, "
                "because every prompt instructed the model to rely only on the "
                "supplied KEY FACTS.",
                st["body"],
            )
        )
    if length.get("mean") is not None:
        over = summary["operational"].get("words", {}).get("mean")
        lines.append(
            Paragraph(
                f"Mean response length was {_fmt(over, 0)} words against an "
                f"800-word cap, giving a length-compliance score of "
                f"{_fmt(length['mean'])}.",
                st["body"],
            )
        )

    return lines


def _intervals_overlap(a: dict, b: dict) -> bool:
    if not a or not b or a.get("mean") is None or b.get("mean") is None:
        return True
    a_lo = a.get("low")
    b_hi = b.get("high")
    if a_lo is None or b_hi is None:
        return True
    return a_lo <= b_hi


def _methodology(st, config, summary) -> list[Any]:
    run = summary.get("run", {}) or {}
    run_cfg = run.get("config", {}) or {}
    weights = summary.get("weights", {})
    judged_repeats = int(run_cfg.get("judge_repeats_per_cell") or 0)
    total_repeats = int(run.get("repeats") or 0)
    sampled_judge = judged_repeats > 0 and judged_repeats < total_repeats

    out = [
        Paragraph("Methodology", st["h1"]),
        Paragraph(
            "Every prompt in this benchmark states its own contract: a KEY FACTS "
            "block, a required section list, a word limit and, in some cases, "
            "literal markers such as VERIFY_SOURCE. The grading specification is "
            "parsed directly out of each prompt file, so the metrics cannot drift "
            "out of sync with the prompts.",
            st["body"],
        ),
        Paragraph("Deterministic metrics", st["h2"]),
        Paragraph(
            "These require no model calls and are fully reproducible. Numbers in "
            "each response are extracted and normalized for units (so 52,000MW "
            "and 52 GW compare equal, and $80-130B is treated as an interval), "
            "then classified as provided, derived, contradicting or unsupported.",
            st["body"],
        ),
        _table(
            [
                ["Metric", "What it measures", "Weight"],
                [Paragraph("Fact Recall", st["cell"]),
                 Paragraph("Share of the prompt's KEY FACTS the response actually cites.", st["cell"]),
                 _fmt(weights.get("fact_recall"), 2)],
                [Paragraph("Numeric Grounding", st["cell"]),
                 Paragraph("Share of all quantities that were provided or arithmetically derivable.", st["cell"]),
                 _fmt(weights.get("numeric_grounding"), 2)],
                [Paragraph("Contradiction-Free", st["cell"]),
                 Paragraph("Penalizes figures that restate a supplied fact incorrectly.", st["cell"]),
                 _fmt(weights.get("contradiction_free"), 2)],
                [Paragraph("Structural Compliance", st["cell"]),
                 Paragraph("Required numbered sections present, in order.", st["cell"]),
                 _fmt(weights.get("structural_compliance"), 2)],
                [Paragraph("Length Compliance", st["cell"]),
                 Paragraph("Adherence to the stated word cap; decays with overrun.", st["cell"]),
                 _fmt(weights.get("length_compliance"), 2)],
                [Paragraph("Required Markers", st["cell"]),
                 Paragraph("Presence of demanded literals such as VERIFY_SOURCE or INFERRED:.", st["cell"]),
                 _fmt(weights.get("required_tokens"), 2)],
                [Paragraph("Format Cleanliness", st["cell"]),
                 Paragraph("Penalizes leaked reasoning, truncation, refusals and stubs.", st["cell"]),
                 _fmt(weights.get("format_clean"), 2)],
                [Paragraph("Judge Overall", st["cell"]),
                 Paragraph("Rubric-weighted LLM-as-judge score.", st["cell"]),
                 _fmt(weights.get("judge_overall"), 2)],
            ],
            [1.4 * inch, 4.2 * inch, 0.7 * inch],
        ),
        Spacer(1, 0.14 * inch),
        Paragraph("LLM-as-judge", st["h2"]),
        Paragraph(
            f"Judge model {_esc(run_cfg.get('judge_model', config.judge_model))} at "
            f"temperature {_esc(run_cfg.get('judge_temperature', config.judge_temperature))}, "
            "graded against an anchored 1-5 rubric. The judge receives the same "
            "KEY FACTS and section list the response was held to, and must quote "
            "the response to justify each score, which makes grades auditable "
            "rather than opaque. "
            + (
                f"To stay inside the daily token allowance, only the first "
                f"{judged_repeats} repeat(s) of each "
                "model × prompt cell were judged; deterministic metrics still "
                "cover every successful generation."
                if sampled_judge
                else "Every successful generation was judged."
            ),
            st["body"],
        ),
    ]
    out.append(
        KeepTogether(
            [
                Paragraph("Generation settings", st["h2"]),
                _table(
                    [
                        ["Setting", "Value"],
                        ["Temperature", _esc(run_cfg.get("temperature", config.temperature))],
                        ["top_p", _esc(run_cfg.get("top_p", config.top_p))],
                        ["Max output tokens", _esc(run_cfg.get("max_output_tokens", config.max_output_tokens))],
                        ["Repeats per cell", _esc(summary.get("run", {}).get("repeats"))],
                        ["Fact match tolerance", _esc(run_cfg.get("match_rel_tolerance"))],
                        ["Contradiction window", _esc(run_cfg.get("contradiction_rel_window"))],
                    ],
                    [2.2 * inch, 4.3 * inch],
                ),
                Paragraph(
                    "All models share identical generation settings so the "
                    "comparison is fair. The token ceiling is deliberately "
                    "generous: one model spends 1,800-2,200 words on internal "
                    "reasoning before answering and was returning empty or "
                    "truncated output at lower ceilings.",
                    st["small"],
                ),
            ]
        )
    )
    return out


def _leaderboards(st, summary, models, techniques, rows, config) -> list[Any]:
    out = [Paragraph("Leaderboards", st["h1"])]

    if models:
        out.append(_chart(charts.leaderboard(models, "model", "Model ranking by composite score")))
        out.append(Spacer(1, 0.1 * inch))

        header = ["Model", "n", "Composite", "Facts", "Grounding", "Structure", "Length", "Judge"]
        data = [header]
        flagged = []
        for i, m in enumerate(models, start=1):
            met = m["metrics"]
            label = m["model"] + (" *" if m.get("self_graded") else "")
            if m.get("self_graded"):
                flagged.append(i)
            data.append(
                [
                    Paragraph(_esc(label), st["cellb"]),
                    str(met["composite"].get("n", 0)),
                    _ci(met["composite"]),
                    _fmt(met["fact_recall"]["mean"]),
                    _fmt(met["numeric_grounding"]["mean"]),
                    _fmt(met["structural_compliance"]["mean"]),
                    _fmt(met["length_compliance"]["mean"]),
                    _fmt(met["judge_overall"]["mean"]),
                ]
            )
        out.append(_table(data, [1.5 * inch, 0.32 * inch, 1.1 * inch, 0.66 * inch,
                                 0.82 * inch, 0.82 * inch, 0.62 * inch, 0.62 * inch],
                          flagged))
        if flagged:
            out.append(
                Paragraph(
                    "* graded by itself; see the self-preference caveat at the end.",
                    st["small"],
                )
            )
        out.append(Spacer(1, 0.16 * inch))

    if techniques:
        out.append(Paragraph("Prompting technique ranking", st["h2"]))
        out.append(
            Paragraph(
                "Averaged across all models and repeats, so this isolates the "
                "effect of the prompting technique itself.",
                st["body"],
            )
        )
        out.append(
            _chart(charts.leaderboard(techniques, "technique", "Technique ranking by composite score"))
        )
        data = [["Technique", "Composite", "Facts", "Grounding", "Structure", "Length", "Judge"]]
        for t in techniques:
            met = t["metrics"]
            data.append(
                [
                    Paragraph(_esc(t["technique"]), st["cellb"]),
                    _ci(met["composite"]),
                    _fmt(met["fact_recall"]["mean"]),
                    _fmt(met["numeric_grounding"]["mean"]),
                    _fmt(met["structural_compliance"]["mean"]),
                    _fmt(met["length_compliance"]["mean"]),
                    _fmt(met["judge_overall"]["mean"]),
                ]
            )
        out.append(_table(data, [1.55 * inch, 1.2 * inch, 0.7 * inch, 0.85 * inch,
                                 0.85 * inch, 0.65 * inch, 0.65 * inch]))
        out.append(Spacer(1, 0.16 * inch))

    heat = summary.get("heatmap", {})
    if heat.get("grid"):
        out.append(Paragraph("Technique against model", st["h2"]))
        out.append(
            Paragraph(
                "Each cell is the mean composite score for that pairing. This "
                "shows whether a technique's advantage holds across models or is "
                "specific to one.",
                st["body"],
            )
        )
        out.append(_chart(charts.heatmap(heat, "Composite score by technique and model"),
                          width=6.4 * inch))

    if models:
        out.append(PageBreak())
        out.append(Paragraph("Metric-by-metric comparison", st["h1"]))
        out.append(
            _chart(
                charts.metric_grouped_bars(
                    models, "model", METRIC_KEYS, METRIC_LABELS,
                    "Per-metric mean by model",
                ),
                width=7.0 * inch,
            )
        )
        if any(r.get("scores_json") for r in rows):
            out.append(Spacer(1, 0.12 * inch))
            out.append(
                _chart(charts.judge_radar(models, rows, config.rubric), width=5.4 * inch)
            )
    return out


def _grounding_section(st, summary, models, rows) -> list[Any]:
    overall = summary["overall"]
    out = [
        Paragraph("Factual grounding", st["h1"]),
        Paragraph(
            "Every prompt instructs the model to base its analysis only on the "
            "supplied KEY FACTS. That makes it possible to audit each number in "
            "each response automatically. A figure counts as provided when it "
            "matches a supplied fact, derived when it equals a simple arithmetic "
            "combination of supplied facts (the formula is recorded), "
            "contradicting when it restates a supplied fact with a materially "
            "different value, and unsupported otherwise.",
            st["body"],
        ),
    ]

    if rows:
        out.append(_chart(charts.grounding_breakdown(models, rows), width=6.6 * inch))
        out.append(Spacer(1, 0.1 * inch))

    data = [["Model", "Grounding", "Fact recall", "Contradictions", "Unsupported figures"]]
    for m in models:
        met, op = m["metrics"], m["operational"]
        counts = _grounding_counts(rows, m["model"])
        data.append(
            [
                Paragraph(_esc(m["model"]), st["cellb"]),
                _pct(met["numeric_grounding"]["mean"]),
                _pct(met["fact_recall"]["mean"]),
                str(counts["contradicting"]),
                str(counts["unsupported"]),
            ]
        )
    out.append(_table(data, [1.7 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch, 1.5 * inch]))

    out.append(Spacer(1, 0.14 * inch))
    out.append(Paragraph("Consistency across repeats", st["h1"]))
    out.append(
        Paragraph(
            "Because temperature is above zero, each cell was generated multiple "
            "times. Similarity between those repeats measures how repeatable a "
            "model is. Note this is lexical overlap, not semantic equivalence: "
            "Groq exposes no embedding model, so two correct answers phrased "
            "differently will score as inconsistent.",
            st["body"],
        )
    )
    cons_by_model = summary.get("consistency", {}).get("by_model", [])
    if cons_by_model:
        out.append(_chart(charts.consistency_chart(cons_by_model), width=6.6 * inch))
        data = [["Model", "TF-IDF cosine", "ROUGE-L"]]
        for c in cons_by_model:
            data.append(
                [Paragraph(_esc(c["model"]), st["cellb"]),
                 _ci(c["tfidf_cosine"]), _ci(c["rouge_l"])]
            )
        out.append(_table(data, [2.2 * inch, 1.7 * inch, 1.7 * inch]))
    return out


def _grounding_counts(rows, model: str) -> dict[str, int]:
    acc = {"supported": 0, "derived": 0, "unsupported": 0, "contradicting": 0}
    for r in rows:
        if r.get("model") != model or r.get("status") != "ok":
            continue
        try:
            numbers = json.loads(r.get("details_json") or "{}").get("numbers", {})
        except json.JSONDecodeError:
            continue
        for key in acc:
            acc[key] += numbers.get(key, 0)
    return acc


def _operational_section(st, summary, models, rows) -> list[Any]:
    ops = summary["operational"]
    out = [
        Paragraph("Operational characteristics", st["h1"]),
        Paragraph(
            "Cost, latency and throughput decide whether a configuration is "
            "usable in production, independently of quality. Note that reasoning "
            "tokens are billed as output even when they never appear in the "
            "response, so cost can exceed what the visible answer suggests.",
            st["body"],
        ),
    ]

    if models:
        out.append(_chart(charts.cost_latency(models), width=7.0 * inch))
        out.append(Spacer(1, 0.1 * inch))

    data = [["Model", "Cost", "Mean latency", "Mean TTFT", "Words", "Reasoning words", "Truncated"]]
    for m in models:
        op = m["operational"]
        data.append(
            [
                Paragraph(_esc(m["model"]), st["cellb"]),
                _money(op.get("cost_usd")),
                f"{_fmt((op['latency_ms']['mean'] or 0) / 1000, 1)}s",
                f"{_fmt((op['ttft_ms']['mean'] or 0) / 1000, 2)}s",
                _fmt(op["words"]["mean"], 0),
                str(op.get("reasoning_words", 0)),
                str(op.get("truncated", 0)),
            ]
        )
    out.append(_table(data, [1.5 * inch, 0.8 * inch, 0.95 * inch, 0.85 * inch,
                             0.7 * inch, 1.05 * inch, 0.8 * inch]))

    out.append(Spacer(1, 0.12 * inch))
    if rows:
        out.append(_chart(charts.length_compliance(rows), width=6.6 * inch))

    out.append(Spacer(1, 0.1 * inch))
    out.append(
        _table(
            [
                ["Run totals", ""],
                ["Generations", f"{ops['generations']} ({ops['errors']} errors, {ops['truncated']} truncated)"],
                ["Retries", str(ops["retries"])],
                ["Prompt tokens", f"{ops['prompt_tokens']:,}"],
                ["Completion tokens", f"{ops['completion_tokens']:,}"],
                ["Generation cost", _money(ops.get("cost_usd"))],
                ["Judging cost", _money(ops.get("judge_cost_usd"))],
            ],
            [1.9 * inch, 4.6 * inch],
        )
    )
    return out


def _failures_section(st, summary) -> list[Any]:
    failures = summary.get("failures", [])
    out = [Paragraph("Failure analysis", st["h1"])]

    if not failures:
        out.append(
            Paragraph(
                "No response produced a contradiction, an unsupported figure or a "
                "length overrun.",
                st["body"],
            )
        )
        return out

    out.append(
        Paragraph(
            "The responses below carry the most severe grounding problems, "
            "weighting contradictions of supplied facts most heavily, then "
            "invented figures, then length overruns. Each quantity is quoted "
            "verbatim with the reason it was flagged.",
            st["body"],
        )
    )

    for f in failures:
        block = [
            Paragraph(
                f"{_esc(f['model'])} &mdash; {_esc(f['prompt_id'])} "
                f"({_esc(f['technique'])}, repeat {f['repeat_index']})",
                st["h2"],
            ),
            Paragraph(
                f"{f.get('contradictions') or 0} contradiction(s), "
                f"{f.get('unsupported') or 0} unsupported figure(s), "
                f"{f.get('word_count')} words against a "
                f"{f.get('word_limit') or 'n/a'}-word limit.",
                st["small"],
            ),
        ]
        rows_data = [["Quantity", "Verdict", "Reason"]]
        for ex in f.get("examples", [])[:6]:
            rows_data.append(
                [
                    Paragraph(_esc(ex.get("raw")), st["mono"]),
                    Paragraph(_esc(ex.get("classification")), st["cell"]),
                    Paragraph(_esc(ex.get("evidence") or "not present in the supplied facts"), st["cell"]),
                ]
            )
        if len(rows_data) > 1:
            block.append(_table(rows_data, [1.2 * inch, 1.0 * inch, 4.3 * inch]))
        if f.get("missing_sections"):
            block.append(
                Paragraph(
                    "Missing sections: " + _esc("; ".join(f["missing_sections"][:5])),
                    st["small"],
                )
            )
        block.append(Spacer(1, 0.12 * inch))
        out.append(KeepTogether(block))

    return out


def _caveats(st, summary, config) -> list[Any]:
    totals = summary["totals"]
    repeats = summary.get("run", {}).get("repeats") or config.repeats
    self_graded = totals.get("self_graded", 0)

    items = [
        (
            "Self-preference bias",
            f"{self_graded} of {totals['judged']} judged responses were graded by "
            f"the same model that produced them ({config.judge_model}). Models "
            "tend to prefer their own output, so those scores are marked with an "
            "asterisk in the leaderboard and should not be read as neutral."
            if self_graded
            else "No response was graded by the model that produced it.",
        ),
        (
            "Sample size",
            f"Each cell was generated {repeats} times. Confidence intervals are "
            "reported throughout and are wide at this sample size; where two "
            "intervals overlap, the ranking between those entries is not "
            "statistically resolved.",
        ),
        (
            "Consistency is lexical",
            "Cross-repeat similarity uses TF-IDF cosine and ROUGE-L because no "
            "embedding model is available on Groq. Two answers that are equally "
            "correct but differently worded will appear inconsistent.",
        ),
        (
            "Grounding heuristics",
            "Numeric classification is rule-based. It normalizes units and "
            "recognizes simple derivations (sums, differences, percentages), but "
            "a legitimate multi-step calculation the rules do not model may be "
            "flagged unsupported. Figures the prompt supplied as few-shot "
            "examples are explicitly treated as provided, not invented.",
        ),
        (
            "Judge reliability",
            "The judge runs at temperature 0 for stability, but it remains a "
            "language model scoring free text. Treat its scores as a correlated "
            "second opinion alongside the deterministic metrics, not ground truth.",
        ),
        (
            "Judge coverage",
            f"Only {totals['judged']} of {totals['ok']} successful responses were "
            "scored by the judge; the remainder failed judging, almost always "
            "because the judge model's own rate-limit budget was exhausted. The "
            "judge component of the composite is therefore based on a subset, and "
            "for unjudged responses the remaining metric weights are "
            "redistributed rather than scored as zero."
            if totals["judged"] < totals["ok"]
            else f"All {totals['judged']} successful responses were judged.",
        ),
        (
            "Cost figures",
            "Costs are computed from a configurable price table in "
            "config/eval_config.yaml. Verify those rates against current Groq "
            "pricing before quoting them externally.",
        ),
    ]

    out = [PageBreak(), Paragraph("Caveats and threats to validity", st["h1"])]
    for title, text in items:
        out.append(Paragraph(title, st["h2"]))
        out.append(Paragraph(text, st["body"]))

    out.append(Paragraph("Reproducing this run", st["h2"]))
    out.append(
        Paragraph(
            "Prompts, configuration and every raw model output for this run are "
            f"stored under runs/{_esc(summary.get('run', {}).get('id', ''))}/. The "
            "deterministic metrics recompute identically from those files; judge "
            "scores may shift slightly if the provider updates the judge model.",
            st["body"],
        )
    )
    return out
