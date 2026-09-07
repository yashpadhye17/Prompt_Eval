"""Configuration loading for the evaluation framework."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "eval_config.yaml"
DEFAULT_RUBRIC_PATH = PROJECT_ROOT / "config" / "rubric.yaml"


def load_api_key() -> str | None:
    """Load GROQ_API_KEY, honouring the repo's existing src/core/.env location."""
    for candidate in (PROJECT_ROOT / "src" / "core" / ".env", PROJECT_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
    load_dotenv(override=False)
    return os.getenv("GROQ_API_KEY")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    family: str = ""
    # Per-model ceiling, for providers that cap output tokens per minute on
    # some models more tightly than others. Falls back to the global setting.
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float | None
    output_per_mtok: float | None

    def cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        """USD cost, or None when pricing is unknown so we never report a wrong number."""
        if self.input_per_mtok is None or self.output_per_mtok is None:
            return None
        p = (prompt_tokens or 0) / 1_000_000 * self.input_per_mtok
        c = (completion_tokens or 0) / 1_000_000 * self.output_per_mtok
        return round(p + c, 8)


@dataclass
class EvalConfig:
    raw: dict[str, Any]
    models: list[ModelSpec]
    pricing: dict[str, Pricing]
    weights: dict[str, float]
    rubric: dict[str, Any]

    # paths
    prompts_root: Path
    runs_root: Path
    db_path: Path

    # generation
    temperature: float
    top_p: float
    max_output_tokens: int
    repeats: int

    # judge
    judge_model: str
    judge_temperature: float
    judge_max_output_tokens: int
    judge_enabled: bool
    judge_repeats_per_cell: int
    judge_compact_rubric: bool
    judge_cache_enabled: bool

    # daily token budget
    budget_enabled: bool
    budget_default_limit: int | None
    budget_limits: dict[str, int]
    budget_reserve_fraction: float

    # runtime
    concurrency: int
    max_retries: int
    retry_initial_backoff_s: float
    retry_max_backoff_s: float
    request_timeout_s: float
    estimate_completion_tokens: int
    model_limits: dict[str, int]

    # scoring / facts
    overflow_zero_multiple: float
    facts_cfg: dict[str, Any] = field(default_factory=dict)
    presets: list[dict[str, Any]] = field(default_factory=list)

    def judge_sample_size(self, repeats: int) -> int:
        """How many repeats of each cell to judge.

        0 or a value at/above `repeats` means judge everything; anything lower
        makes the judge scores a stratified sample of the generations.
        """
        configured = self.judge_repeats_per_cell
        if configured <= 0:
            return repeats
        return min(configured, repeats)

    def pricing_for(self, model: str) -> Pricing:
        return self.pricing.get(model, Pricing(None, None))

    def model_ids(self) -> list[str]:
        return [m.id for m in self.models]

    def label_for(self, model_id: str) -> str:
        for m in self.models:
            if m.id == model_id:
                return m.label
        return model_id

    def max_tokens_for(self, model_id: str, default: int | None = None) -> int:
        """Output ceiling for a model, honouring any hard per-model limit.

        Applies to both generation and judging: the limit is a property of the
        model's quota, not of the role it is playing.
        """
        fallback = default if default is not None else self.max_output_tokens
        if model_id in self.model_limits:
            return min(self.model_limits[model_id], fallback)
        for m in self.models:
            if m.id == model_id and m.max_output_tokens:
                return min(m.max_output_tokens, fallback)
        return fallback


def _normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    clean = {k: float(v) for k, v in (weights or {}).items() if float(v) >= 0}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("scoring.weights must contain at least one positive weight")
    return {k: v / total for k, v in clean.items()}


def load_config(
    config_path: str | Path | None = None,
    rubric_path: str | Path | None = None,
) -> EvalConfig:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    rub_path = Path(rubric_path) if rubric_path else DEFAULT_RUBRIC_PATH

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    with open(rub_path, "r", encoding="utf-8") as fh:
        rubric = yaml.safe_load(fh) or {}

    paths = raw.get("paths", {})
    gen = raw.get("generation", {})
    judge = raw.get("judge", {})
    rt = raw.get("runtime", {})
    scoring = raw.get("scoring", {})
    budget = raw.get("budget", {}) or {}

    models = [
        ModelSpec(
            id=m["id"],
            label=m.get("label", m["id"]),
            family=m.get("family", ""),
            max_output_tokens=m.get("max_output_tokens"),
        )
        for m in raw.get("models", [])
    ]
    if not models:
        raise ValueError("config must define at least one model")

    pricing: dict[str, Pricing] = {}
    for model_id, entry in (raw.get("pricing") or {}).items():
        if not entry:
            pricing[model_id] = Pricing(None, None)
            continue
        pricing[model_id] = Pricing(
            input_per_mtok=entry.get("input_per_mtok"),
            output_per_mtok=entry.get("output_per_mtok"),
        )

    def _p(key: str, default: str) -> Path:
        return (PROJECT_ROOT / paths.get(key, default)).resolve()

    return EvalConfig(
        raw=raw,
        models=models,
        pricing=pricing,
        weights=_normalize_weights(scoring.get("weights", {})),
        rubric=rubric,
        prompts_root=_p("prompts_root", "src/prompts"),
        runs_root=_p("runs_root", "runs"),
        db_path=_p("db_path", "runs/eval.db"),
        temperature=float(gen.get("temperature", 0.9)),
        top_p=float(gen.get("top_p", 1)),
        max_output_tokens=int(gen.get("max_output_tokens", 2048)),
        repeats=int(gen.get("repeats", 3)),
        judge_model=judge.get("model", "openai/gpt-oss-120b"),
        judge_temperature=float(judge.get("temperature", 0)),
        judge_max_output_tokens=int(judge.get("max_output_tokens", 2048)),
        judge_enabled=bool(judge.get("enabled", True)),
        judge_repeats_per_cell=int(judge.get("repeats_per_cell", 0)),
        judge_compact_rubric=bool(judge.get("compact_rubric", False)),
        judge_cache_enabled=bool(judge.get("cache_enabled", True)),
        budget_enabled=bool(budget.get("enabled", True)),
        budget_default_limit=(
            int(budget["daily_token_limit"])
            if budget.get("daily_token_limit")
            else None
        ),
        budget_limits={
            k: int(v) for k, v in (budget.get("per_model") or {}).items() if v
        },
        budget_reserve_fraction=float(budget.get("reserve_fraction", 0.05)),
        concurrency=int(rt.get("concurrency", 4)),
        max_retries=int(rt.get("max_retries", 5)),
        retry_initial_backoff_s=float(rt.get("retry_initial_backoff_s", 2)),
        retry_max_backoff_s=float(rt.get("retry_max_backoff_s", 30)),
        request_timeout_s=float(rt.get("request_timeout_s", 180)),
        estimate_completion_tokens=int(rt.get("estimate_completion_tokens", 2000)),
        model_limits={k: int(v) for k, v in (raw.get("model_limits") or {}).items()},
        overflow_zero_multiple=float(
            (scoring.get("length") or {}).get("overflow_zero_multiple", 3.0)
        ),
        facts_cfg=raw.get("facts", {}) or {},
        presets=list(raw.get("presets") or []),
    )
