"""Aggregate per-generation rows into the views the UI and report consume.

Repeat runs are summarized with mean, standard deviation and a 95% confidence
interval. With n=3 the interval is wide on purpose: it is the honest way to
show that small differences between models are not resolvable at this sample
size, which stops the leaderboard from being over-read.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

# Two-sided 95% t critical values by degrees of freedom (n-1).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000,
}

METRIC_KEYS = [
    "fact_recall",
    "numeric_grounding",
    "contradiction_free",
    "structural_compliance",
    "length_compliance",
    "required_tokens",
    "format_clean",
    "judge_overall",
]

METRIC_LABELS = {
    "fact_recall": "Fact Recall",
    "numeric_grounding": "Numeric Grounding",
    "contradiction_free": "Contradiction-Free",
    "structural_compliance": "Structural Compliance",
    "length_compliance": "Length Compliance",
    "required_tokens": "Required Markers",
    "format_clean": "Format Cleanliness",
    "judge_overall": "Judge Overall",
}


def _t_critical(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in _T95:
        return _T95[df]
    for key in sorted(_T95):
        if df <= key:
            return _T95[key]
    return 1.96


@dataclass
class Stat:
    n: int = 0
    mean: float | None = None
    stdev: float | None = None
    ci95: float | None = None   # half-width

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "stdev": self.stdev,
            "ci95": self.ci95,
            "low": None if self.mean is None or self.ci95 is None else round(max(0.0, self.mean - self.ci95), 4),
            "high": None if self.mean is None or self.ci95 is None else round(self.mean + self.ci95, 4),
        }


def summarize(values: Iterable[float | None]) -> Stat:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return Stat()
    if len(clean) == 1:
        return Stat(n=1, mean=round(clean[0], 4), stdev=0.0, ci95=None)

    m = mean(clean)
    sd = stdev(clean)
    half = _t_critical(len(clean) - 1) * sd / math.sqrt(len(clean))
    return Stat(n=len(clean), mean=round(m, 4), stdev=round(sd, 4), ci95=round(half, 4))


def composite_score(row: dict[str, Any], weights: dict[str, float]) -> float | None:
    """Weighted 0-1 composite for one generation.

    Weights of metrics that are missing on this row (typically judge_overall
    when judging is disabled or failed) are redistributed across the rest, so
    a missing judge lowers confidence rather than silently scoring zero.
    """
    usable = {
        key: weights[key]
        for key in weights
        if row.get(key) is not None
    }
    total = sum(usable.values())
    if total <= 0:
        return None
    return round(sum(float(row[k]) * w for k, w in usable.items()) / total, 4)


def attach_composites(rows: Sequence[dict], weights: dict[str, float]) -> list[dict]:
    out = []
    for row in rows:
        enriched = dict(row)
        enriched["composite"] = composite_score(row, weights)
        out.append(enriched)
    return out


def _group_stats(rows: Sequence[dict], keys: Sequence[str]) -> dict[str, Any]:
    stats = {k: summarize([r.get(k) for r in rows]).to_dict() for k in keys}
    stats["composite"] = summarize([r.get("composite") for r in rows]).to_dict()
    return stats


def _operational(rows: Sequence[dict]) -> dict[str, Any]:
    def total(key: str) -> float:
        return round(sum(float(r.get(key) or 0) for r in rows), 6)

    latencies = [r.get("latency_ms") for r in rows if r.get("latency_ms")]
    ttfts = [r.get("ttft_ms") for r in rows if r.get("ttft_ms")]
    costs = [r.get("cost_usd") for r in rows if r.get("cost_usd") is not None]

    return {
        "generations": len(rows),
        "errors": sum(1 for r in rows if r.get("status") != "ok"),
        "truncated": sum(1 for r in rows if r.get("truncated")),
        "retries": int(total("retries")),
        "prompt_tokens": int(total("prompt_tokens")),
        "completion_tokens": int(total("completion_tokens")),
        "reasoning_words": int(total("reasoning_words")),
        "cost_usd": round(sum(float(c) for c in costs), 6) if costs else None,
        "judge_cost_usd": round(
            sum(float(r.get("judge_cost_usd") or 0) for r in rows), 6
        ),
        "latency_ms": summarize(latencies).to_dict(),
        "ttft_ms": summarize(ttfts).to_dict(),
        "words": summarize([r.get("word_count") for r in rows]).to_dict(),
    }


def by_model(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["model"]].append(r)

    out = []
    for model, group in grouped.items():
        entry = {
            "model": model,
            "metrics": _group_stats(group, METRIC_KEYS),
            "operational": _operational(group),
            "self_graded": any(r.get("self_graded") for r in group),
        }
        entry["score"] = entry["metrics"]["composite"]["mean"]
        out.append(entry)
    return sorted(out, key=lambda e: (e["score"] is None, -(e["score"] or 0)))


def by_technique(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["technique"]].append(r)

    out = []
    for technique, group in grouped.items():
        entry = {
            "technique": technique,
            "metrics": _group_stats(group, METRIC_KEYS),
            "operational": _operational(group),
        }
        entry["score"] = entry["metrics"]["composite"]["mean"]
        out.append(entry)
    return sorted(out, key=lambda e: (e["score"] is None, -(e["score"] or 0)))


def by_query(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["query_id"]].append(r)
    return [
        {
            "query_id": q,
            "metrics": _group_stats(g, METRIC_KEYS),
            "score": _group_stats(g, METRIC_KEYS)["composite"]["mean"],
        }
        for q, g in sorted(grouped.items())
    ]


def heatmap(rows: Sequence[dict]) -> dict[str, Any]:
    """technique x model composite means, for the dashboard heatmap."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    techniques: list[str] = []
    models: list[str] = []

    for r in rows:
        t, m = r["technique"], r["model"]
        if t not in techniques:
            techniques.append(t)
        if m not in models:
            models.append(m)
        if r.get("composite") is not None:
            cells[(t, m)].append(float(r["composite"]))

    techniques.sort()
    models.sort()
    grid = [
        [
            round(mean(cells[(t, m)]), 4) if cells[(t, m)] else None
            for m in models
        ]
        for t in techniques
    ]
    return {"techniques": techniques, "models": models, "grid": grid}


def prompt_cells(rows: Sequence[dict]) -> list[dict]:
    """One entry per (prompt, model) cell with its repeats summarized."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["prompt_id"], r["model"])].append(r)

    out = []
    for (prompt_id, model), group in sorted(grouped.items()):
        first = group[0]
        out.append(
            {
                "prompt_id": prompt_id,
                "model": model,
                "technique": first["technique"],
                "query_id": first["query_id"],
                "repeats": len(group),
                "metrics": _group_stats(group, METRIC_KEYS),
                "composite": _group_stats(group, METRIC_KEYS)["composite"],
                "generation_ids": [r["id"] for r in group],
            }
        )
    return out


def failure_examples(rows: Sequence[dict], limit: int = 8) -> list[dict]:
    """Worst offenders, prioritizing contradictions then invented figures."""
    scored = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        detail = _load_details(r)
        numbers = detail.get("numbers", {})
        severity = (
            (r.get("contradictions") or 0) * 10
            + (numbers.get("unsupported") or 0)
            + (0 if (r.get("length_compliance") is None) else (1 - r["length_compliance"]) * 5)
        )
        if severity <= 0:
            continue
        scored.append((severity, r, detail))

    scored.sort(key=lambda t: -t[0])
    out = []
    for severity, r, detail in scored[:limit]:
        numbers = detail.get("numbers", {})
        items = numbers.get("items", [])
        out.append(
            {
                "generation_id": r["id"],
                "prompt_id": r["prompt_id"],
                "technique": r["technique"],
                "model": r["model"],
                "repeat_index": r["repeat_index"],
                "severity": round(severity, 2),
                "contradictions": r.get("contradictions"),
                "unsupported": numbers.get("unsupported"),
                "word_count": r.get("word_count"),
                "word_limit": r.get("word_limit"),
                "examples": [
                    {
                        "raw": i.get("raw"),
                        "classification": i.get("classification"),
                        "evidence": i.get("evidence"),
                    }
                    for i in items
                    if i.get("classification") in ("contradicting", "unsupported")
                ][:8],
                "missing_sections": detail.get("structure", {}).get("missing", []),
            }
        )
    return out


def _load_details(row: dict) -> dict:
    raw = row.get("details_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def build_summary(
    rows: Sequence[dict],
    consistency_rows: Sequence[dict],
    weights: dict[str, float],
) -> dict[str, Any]:
    """The single aggregate payload used by the API, dashboard and PDF."""
    enriched = attach_composites(rows, weights)
    ok_rows = [r for r in enriched if r.get("status") == "ok"]

    cons_map = {(c["prompt_id"], c["model"]): c for c in consistency_rows}
    cons_values = [c.get("tfidf_cosine") for c in consistency_rows]
    rouge_values = [c.get("rouge_l") for c in consistency_rows]

    judged = [r for r in ok_rows if r.get("judge_overall") is not None]

    return {
        "totals": {
            "generations": len(enriched),
            "ok": len(ok_rows),
            "failed": len(enriched) - len(ok_rows),
            "judged": len(judged),
            "self_graded": sum(1 for r in judged if r.get("self_graded")),
            "models": sorted({r["model"] for r in enriched}),
            "techniques": sorted({r["technique"] for r in enriched}),
            "prompts": sorted({r["prompt_id"] for r in enriched}),
        },
        "overall": _group_stats(ok_rows, METRIC_KEYS),
        "operational": _operational(ok_rows),
        "models": by_model(ok_rows),
        "techniques": by_technique(ok_rows),
        "queries": by_query(ok_rows),
        "heatmap": heatmap(ok_rows),
        "cells": [
            {
                **cell,
                "consistency": cons_map.get((cell["prompt_id"], cell["model"]), {}),
            }
            for cell in prompt_cells(ok_rows)
        ],
        "consistency": {
            "tfidf_cosine": summarize(cons_values).to_dict(),
            "rouge_l": summarize(rouge_values).to_dict(),
            "by_model": _consistency_by_model(consistency_rows),
        },
        "failures": failure_examples(ok_rows),
        "weights": weights,
        "metric_labels": METRIC_LABELS,
    }


def _consistency_by_model(consistency_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in consistency_rows:
        grouped[c["model"]].append(c)
    return [
        {
            "model": model,
            "tfidf_cosine": summarize([c.get("tfidf_cosine") for c in group]).to_dict(),
            "rouge_l": summarize([c.get("rouge_l") for c in group]).to_dict(),
        }
        for model, group in sorted(grouped.items())
    ]
