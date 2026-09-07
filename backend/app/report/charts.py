"""Matplotlib chart generation for the PDF report.

Charts are rendered to PNG bytes and embedded, so the report is a single
self-contained file.
"""

from __future__ import annotations

import io
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2f6f9f", "#7aa5c4", "#c1554e", "#d99a3f", "#5b8c5a", "#8a6fa8"]
GRID_KW = {"color": "#dddddd", "linewidth": 0.6}


def _finish(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _short(name: str, width: int = 18) -> str:
    return name if len(name) <= width else name[: width - 1] + "\u2026"


def leaderboard(entries: Sequence[dict], key: str, title: str) -> bytes:
    """Horizontal bars of composite score with 95% CI error bars."""
    labels = [_short(e[key]) for e in entries]
    means = [(e["metrics"]["composite"]["mean"] or 0) for e in entries]
    errs = [(e["metrics"]["composite"]["ci95"] or 0) for e in entries]

    fig, ax = plt.subplots(figsize=(7.2, max(1.8, 0.55 * len(labels) + 1)))
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=errs, color=PALETTE[0], height=0.55,
            error_kw={"ecolor": "#444444", "capsize": 4, "elinewidth": 1})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Composite score (0-1), error bars = 95% CI")
    ax.set_title(title)
    ax.xaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for yi, (m, e) in enumerate(zip(means, errs)):
        ax.text(min(m + (e or 0) + 0.02, 0.97), yi, f"{m:.3f}",
                va="center", fontsize=8.5, color="#333333")
    return _finish(fig)


def metric_grouped_bars(entries: Sequence[dict], key: str, metrics: Sequence[str],
                        labels: dict[str, str], title: str) -> bytes:
    """Grouped bars: one cluster per metric, one bar per model/technique."""
    names = [_short(e[key], 14) for e in entries]
    n = len(entries)
    x = np.arange(len(metrics))
    width = min(0.8 / max(n, 1), 0.22)

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for i, entry in enumerate(entries):
        vals = [(entry["metrics"].get(m, {}).get("mean") or 0) for m in metrics]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
               label=names[i], color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x, [labels.get(m, m) for m in metrics], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0-1)")
    ax.set_title(title)
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncols=min(n, 4), frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _finish(fig)


def heatmap(data: dict[str, Any], title: str) -> bytes:
    """technique x model composite heatmap."""
    grid = np.array(
        [[np.nan if v is None else v for v in row] for row in data["grid"]],
        dtype=float,
    )
    techniques = data["techniques"]
    models = [_short(m, 14) for m in data["models"]]

    fig, ax = plt.subplots(figsize=(1.5 + 1.25 * max(len(models), 1),
                                    1.4 + 0.5 * max(len(techniques), 1)))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(models)), models, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(techniques)), techniques, fontsize=8)
    ax.set_title(title)

    for i in range(len(techniques)):
        for j in range(len(models)):
            value = grid[i][j]
            text = "n/a" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="#111111")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Composite")
    return _finish(fig)


def cost_latency(entries: Sequence[dict]) -> bytes:
    """Cost per run and mean latency side by side."""
    labels = [_short(e["model"], 14) for e in entries]
    costs = [(e["operational"].get("cost_usd") or 0) for e in entries]
    lat = [(e["operational"]["latency_ms"]["mean"] or 0) / 1000 for e in entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.0))
    x = np.arange(len(labels))

    ax1.bar(x, costs, 0.55, color=PALETTE[2])
    ax1.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("USD")
    ax1.set_title("Generation cost")
    ax1.yaxis.grid(True, **GRID_KW)
    ax1.set_axisbelow(True)
    for xi, c in zip(x, costs):
        ax1.text(xi, c, f"${c:.4f}", ha="center", va="bottom", fontsize=7.5)

    ax2.bar(x, lat, 0.55, color=PALETTE[3])
    ax2.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Seconds")
    ax2.set_title("Mean latency per response")
    ax2.yaxis.grid(True, **GRID_KW)
    ax2.set_axisbelow(True)
    for xi, v in zip(x, lat):
        ax2.text(xi, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=7.5)

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    return _finish(fig)


def grounding_breakdown(entries: Sequence[dict], rows: Sequence[dict]) -> bytes:
    """Stacked composition of every number emitted, per model."""
    from collections import defaultdict
    import json as _json

    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"supported": 0, "derived": 0, "unsupported": 0, "contradicting": 0}
    )
    for r in rows:
        if r.get("status") != "ok":
            continue
        try:
            detail = _json.loads(r.get("details_json") or "{}")
        except _json.JSONDecodeError:
            continue
        numbers = detail.get("numbers", {})
        b = buckets[r["model"]]
        b["supported"] += numbers.get("supported", 0)
        b["derived"] += numbers.get("derived", 0)
        b["unsupported"] += numbers.get("unsupported", 0)
        b["contradicting"] += numbers.get("contradicting", 0)

    models = [e["model"] for e in entries if e["model"] in buckets]
    if not models:
        models = list(buckets)
    labels = [_short(m, 14) for m in models]

    kinds = [
        ("supported", "Provided", "#5b8c5a"),
        ("derived", "Derived", "#7aa5c4"),
        ("unsupported", "Unsupported", "#d99a3f"),
        ("contradicting", "Contradicting", "#c1554e"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    for key, label, color in kinds:
        vals = np.array([buckets[m][key] for m in models], dtype=float)
        ax.bar(x, vals, 0.55, bottom=bottom, label=label, color=color)
        bottom += vals

    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Numeric claims")
    ax.set_title("Where every number in the outputs came from")
    ax.legend(fontsize=8, frameon=False, ncols=4)
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _finish(fig)


def consistency_chart(entries: Sequence[dict]) -> bytes:
    """Cross-repeat similarity per model."""
    labels = [_short(e["model"], 14) for e in entries]
    cos = [(e["tfidf_cosine"]["mean"] or 0) for e in entries]
    rouge = [(e["rouge_l"]["mean"] or 0) for e in entries]

    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, cos, 0.4, label="TF-IDF cosine", color=PALETTE[0])
    ax.bar(x + 0.2, rouge, 0.4, label="ROUGE-L", color=PALETTE[4])
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Similarity across repeats")
    ax.set_title("Output stability (higher = more repeatable wording)")
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _finish(fig)


def length_compliance(rows: Sequence[dict], limit_default: int = 800) -> bytes:
    """Word count per model against the prompt's cap."""
    from collections import defaultdict

    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok" and r.get("word_count"):
            grouped[r["model"]].append(float(r["word_count"]))

    models = sorted(grouped)
    if not models:
        fig, ax = plt.subplots(figsize=(7.6, 2.4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.axis("off")
        return _finish(fig)

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.boxplot([grouped[m] for m in models], tick_labels=[_short(m, 14) for m in models],
               patch_artist=True,
               boxprops={"facecolor": PALETTE[1], "edgecolor": "#33556f"},
               medianprops={"color": "#c1554e", "linewidth": 1.6})
    ax.axhline(limit_default, color="#c1554e", linestyle="--", linewidth=1.2,
               label=f"{limit_default}-word limit")
    ax.set_ylabel("Words per response")
    ax.set_title("Response length vs the prompt's word limit")
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _finish(fig)


def judge_radar(entries: Sequence[dict], rows: Sequence[dict], rubric: dict) -> bytes:
    """Per-dimension judge profile for each model."""
    import json as _json
    from collections import defaultdict

    dims = [d["id"] for d in rubric.get("dimensions", [])]
    names = [d["name"] for d in rubric.get("dimensions", [])]
    if not dims:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "no rubric", ha="center", va="center")
        ax.axis("off")
        return _finish(fig)

    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("status") != "ok" or not r.get("scores_json"):
            continue
        try:
            scores = _json.loads(r["scores_json"])
        except _json.JSONDecodeError:
            continue
        for d in dims:
            if d in scores:
                acc[r["model"]][d].append(float(scores[d]))

    models = [e["model"] for e in entries if e["model"] in acc] or list(acc)
    if not models:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "no judge scores", ha="center", va="center")
        ax.axis("off")
        return _finish(fig)

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.0, 5.2), subplot_kw={"polar": True})
    for i, model in enumerate(models):
        vals = [
            (sum(acc[model][d]) / len(acc[model][d])) if acc[model][d] else 0
            for d in dims
        ]
        vals += vals[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=1.6, label=_short(model, 14))
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1], [n.replace(" ", "\n") for n in names], fontsize=7.5)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_title("Judge scores by rubric dimension (1-5)", pad=18)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.28, 1.12))
    return _finish(fig)
