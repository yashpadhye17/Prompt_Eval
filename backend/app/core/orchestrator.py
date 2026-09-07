"""Run orchestration: generate, score, judge, persist, and report progress."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..eval import aggregate, consistency, deterministic
from ..eval.judge import Judge
from ..eval.spec import PromptSpec, load_prompt_specs
from ..store.db import Database, utcday, utcnow
from .budget import ROLE_GENERATE, ROLE_JUDGE, BudgetLedger
from .config import EvalConfig
from .groq_client import GroqEvalClient

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-")


class BudgetExceeded(ValueError):
    """The planned matrix does not fit in today's remaining token allowance."""

    def __init__(self, message: str, budget: dict[str, Any]):
        super().__init__(message)
        self.budget = budget


@dataclass
class Task:
    spec: PromptSpec
    model: str
    repeat_index: int
    judge: bool = True


@dataclass
class RunRequest:
    models: list[str]
    prompt_ids: list[str]
    repeats: int
    judge_enabled: bool
    judge_model: str | None = None
    # Repeats per cell to judge; None takes the configured default.
    judge_repeats_per_cell: int | None = None


@dataclass
class ProgressBus:
    """Fan-out of run events to any number of SSE subscribers."""

    _subscribers: dict[str, list[asyncio.Queue]] = field(default_factory=dict)
    _history: dict[str, list[dict]] = field(default_factory=dict)

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, []).append(queue)
        # Replay so a late subscriber still sees how far the run has come.
        for event in self._history.get(run_id, []):
            queue.put_nowait(event)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id, [])
        if queue in subs:
            subs.remove(queue)

    def publish(self, run_id: str, event: dict) -> None:
        event = {**event, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._history.setdefault(run_id, []).append(event)
        # Bound replay memory on long runs.
        if len(self._history[run_id]) > 600:
            self._history[run_id] = self._history[run_id][-600:]
        for queue in list(self._subscribers.get(run_id, [])):
            queue.put_nowait(event)

    def clear(self, run_id: str) -> None:
        self._history.pop(run_id, None)
        self._subscribers.pop(run_id, None)


class Orchestrator:
    def __init__(self, config: EvalConfig, db: Database | None = None):
        self.config = config
        self.db = db or Database(config.db_path)
        self.specs = {s.prompt_id: s for s in load_prompt_specs(config.prompts_root)}
        self.bus = ProgressBus()
        self.ledger = BudgetLedger(
            self.db,
            limits=config.budget_limits,
            default_limit=config.budget_default_limit,
            enabled=config.budget_enabled,
        )
        self.db.prune_token_usage()
        self.db.backfill_token_usage()
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()

    # ---------------- planning / estimation ----------------

    def resolve_prompts(self, prompt_ids: Sequence[str] | None) -> list[PromptSpec]:
        if not prompt_ids:
            return list(self.specs.values())
        missing = [p for p in prompt_ids if p not in self.specs]
        if missing:
            raise ValueError(f"unknown prompt ids: {missing}")
        return [self.specs[p] for p in prompt_ids]

    def resolve_models(self, models: Sequence[str] | None) -> list[str]:
        return list(models) if models else self.config.model_ids()

    def judge_sample_size(self, request: RunRequest) -> int:
        """Repeats per cell that will be judged, after clamping to the matrix."""
        if not request.judge_enabled:
            return 0
        requested = request.judge_repeats_per_cell
        if requested is None:
            return self.config.judge_sample_size(request.repeats)
        if requested <= 0:
            return request.repeats
        return min(requested, request.repeats)

    def plan_tasks(self, request: RunRequest) -> list[Task]:
        """Expand the matrix, marking which generations the judge will score.

        Judging is sampled by taking the first `k` repeats of every cell. The
        repeats are independent draws from the same distribution, so a fixed
        prefix is as unbiased as a random pick and keeps runs reproducible.
        Sampling on repeats rather than on cells is what preserves full
        coverage of the model x technique grid the heatmap draws.
        """
        specs = self.resolve_prompts(request.prompt_ids)
        models = self.resolve_models(request.models)
        judged_repeats = self.judge_sample_size(request)

        return [
            Task(
                spec=spec,
                model=model,
                repeat_index=i,
                judge=request.judge_enabled and i < judged_repeats,
            )
            for spec in specs
            for model in models
            for i in range(request.repeats)
        ]

    def _observed_completion_tokens(self, model: str) -> float | None:
        row = self.db.query_one(
            "SELECT AVG(completion_tokens) AS avg FROM generations"
            " WHERE model = ? AND status = 'ok' AND completion_tokens IS NOT NULL",
            (model,),
        )
        return row["avg"] if row and row["avg"] else None

    def estimate(self, request: RunRequest) -> dict[str, Any]:
        """Price the matrix before spending anything.

        Uses observed per-model averages from previous runs where available and
        falls back to the configured nominal otherwise, so the estimate gets
        sharper the more the framework is used.
        """
        specs = self.resolve_prompts(request.prompt_ids)
        models = self.resolve_models(request.models)
        judged_repeats = self.judge_sample_size(request)
        judge_calls = len(specs) * len(models) * judged_repeats

        per_model: list[dict[str, Any]] = []
        total_cost = 0.0
        unpriced: list[str] = []
        # Tokens each model will draw from its own daily allowance, keyed by
        # model because generation and judging can land on the same one.
        projected: dict[str, int] = {}

        for model in models:
            observed = self._observed_completion_tokens(model)
            completion_est = observed or self.config.estimate_completion_tokens
            # ~4 characters per token is a good enough approximation here.
            prompt_tokens = sum(len(s.text) / 4 for s in specs) * request.repeats
            completion_tokens = completion_est * len(specs) * request.repeats

            pricing = self.config.pricing_for(model)
            cost = pricing.cost(int(prompt_tokens), int(completion_tokens))
            if cost is None:
                unpriced.append(model)
            else:
                total_cost += cost

            projected[model] = projected.get(model, 0) + int(
                prompt_tokens + completion_tokens
            )
            per_model.append(
                {
                    "model": model,
                    "generations": len(specs) * request.repeats,
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "completion_estimate_source": "observed" if observed else "config",
                    "cost_usd": cost,
                }
            )

        judge_cost = 0.0
        judge_model = request.judge_model or self.config.judge_model
        if judge_calls:
            judge_pricing = self.config.pricing_for(judge_model)
            observed_judge = self.ledger.observed_call_tokens(judge_model, ROLE_JUDGE)
            if observed_judge:
                # Split the observed total using the shape seen in practice:
                # the rubric and report dominate, the verdict is small.
                judge_prompt_tokens = observed_judge * 0.85 * judge_calls
                judge_completion_tokens = observed_judge * 0.15 * judge_calls
            else:
                judge_prompt_tokens = (
                    sum(len(s.text) / 4 for s in specs) / len(specs) + 1200
                ) * judge_calls
                judge_completion_tokens = 900 * judge_calls
            judge_cost = (
                judge_pricing.cost(
                    int(judge_prompt_tokens), int(judge_completion_tokens)
                )
                or 0.0
            )
            projected[judge_model] = projected.get(judge_model, 0) + int(
                judge_prompt_tokens + judge_completion_tokens
            )

        return {
            "generations": len(specs) * len(models) * request.repeats,
            "judge_calls": judge_calls,
            "judge_repeats_per_cell": judged_repeats,
            "judge_sampled": bool(judge_calls) and judged_repeats < request.repeats,
            "judge_model": judge_model if judge_calls else None,
            "per_model": per_model,
            "generation_cost_usd": round(total_cost, 6),
            "judge_cost_usd": round(judge_cost, 6),
            "total_cost_usd": round(total_cost + judge_cost, 6),
            "unpriced_models": unpriced,
            "budget": self.budget_check(projected),
            "note": (
                "Rough estimate. Token counts are approximated from character "
                "length and typical output size; treat as an order of magnitude."
            ),
        }

    def budget_check(self, projected: dict[str, int]) -> dict[str, Any]:
        """Compare a run's projected token draw against today's allowance.

        Reported per model because the allowance is per model: a matrix can be
        affordable in total and still be impossible because one model, usually
        the judge, is asked for more than it has left.
        """
        entries = []
        blocking = []
        for model, tokens in sorted(projected.items()):
            status = self.ledger.status(model)
            remaining = status.remaining
            reserve = self.config.budget_reserve_fraction
            usable = (
                None
                if remaining is None
                else int(remaining * (1.0 - reserve))
            )
            fits = usable is None or tokens <= usable
            entries.append(
                {
                    **status.to_dict(),
                    "projected": tokens,
                    "usable": usable,
                    "fits": fits,
                    "shortfall": (
                        0 if fits or usable is None else tokens - usable
                    ),
                }
            )
            if not fits:
                blocking.append(model)

        return {
            "enabled": self.ledger.enabled,
            "day": utcday(),
            "models": entries,
            "fits": not blocking,
            "blocking_models": blocking,
        }

    # ---------------- run lifecycle ----------------

    def create_run(self, request: RunRequest) -> str:
        specs = self.resolve_prompts(request.prompt_ids)
        models = self.resolve_models(request.models)
        judged_repeats = self.judge_sample_size(request)
        run_id = f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

        self.db.create_run(
            run_id=run_id,
            models=models,
            prompt_ids=[s.prompt_id for s in specs],
            repeats=request.repeats,
            judge_enabled=request.judge_enabled,
            judge_model=(
                (request.judge_model or self.config.judge_model)
                if request.judge_enabled
                else None
            ),
            config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_output_tokens": self.config.max_output_tokens,
                "judge_model": request.judge_model or self.config.judge_model,
                "judge_temperature": self.config.judge_temperature,
                "judge_repeats_per_cell": judged_repeats,
                "weights": self.config.weights,
                "match_rel_tolerance": self.config.facts_cfg.get("match_rel_tolerance"),
                "contradiction_rel_window": self.config.facts_cfg.get(
                    "contradiction_rel_window"
                ),
            },
            total_tasks=len(specs) * len(models) * request.repeats,
            judge_repeats_per_cell=judged_repeats,
        )
        return run_id

    def launch(self, request: RunRequest, *, enforce_budget: bool = True) -> str:
        if enforce_budget and self.ledger.enabled:
            preview = self.estimate(request)
            budget = preview.get("budget") or {}
            if not budget.get("fits", True):
                blocked = ", ".join(budget.get("blocking_models") or [])
                raise BudgetExceeded(
                    f"this run exceeds today's remaining token budget for {blocked}",
                    budget=budget,
                )
        run_id = self.create_run(request)
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id, request))
        return run_id

    def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if not task or task.done():
            return False
        self._cancelled.add(run_id)
        task.cancel()
        self.db.set_run_status(run_id, "cancelled", finished=True)
        self.bus.publish(run_id, {"type": "run_cancelled", "run_id": run_id})
        return True

    def is_running(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return bool(task and not task.done())

    async def _execute(self, run_id: str, request: RunRequest) -> None:
        specs = self.resolve_prompts(request.prompt_ids)
        models = self.resolve_models(request.models)
        tasks = self.plan_tasks(request)
        judged_repeats = self.judge_sample_size(request)

        self.db.set_run_status(run_id, "running")
        self.bus.publish(
            run_id,
            {
                "type": "run_started",
                "run_id": run_id,
                "total": len(tasks),
                "models": models,
                "prompts": [s.prompt_id for s in specs],
                "repeats": request.repeats,
                "judge_enabled": request.judge_enabled,
                "judge_repeats_per_cell": judged_repeats,
                "judge_calls": sum(1 for t in tasks if t.judge),
            },
        )

        client = GroqEvalClient(self.config, ledger=self.ledger)
        judge = (
            Judge(client, self.config, model=request.judge_model, cache=self.db)
            if request.judge_enabled
            else None
        )
        run_dir = self.config.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        done = 0
        lock = asyncio.Lock()

        async def worker(task: Task) -> None:
            nonlocal done
            result = await self._run_task(run_id, run_dir, task, client, judge)
            async with lock:
                done += 1
                self.db.bump_run_progress(run_id, ok=result["status"] == "ok")
                self.bus.publish(
                    run_id,
                    {
                        "type": "task_done",
                        "run_id": run_id,
                        "completed": done,
                        "total": len(tasks),
                        **result,
                    },
                )

        try:
            # The client's own semaphore bounds real concurrency; gathering all
            # tasks here just queues them.
            await asyncio.gather(*(worker(t) for t in tasks))

            self.bus.publish(run_id, {"type": "phase", "run_id": run_id, "phase": "consistency"})
            self._compute_consistency(run_id)

            self.db.set_run_status(run_id, "completed", finished=True)
            self.bus.publish(
                run_id, {"type": "run_finished", "run_id": run_id, "status": "completed"}
            )
        except asyncio.CancelledError:
            self.db.set_run_status(run_id, "cancelled", finished=True)
            raise
        except Exception as exc:  # noqa: BLE001 - recorded on the run
            self.db.set_run_status(run_id, "failed", error=str(exc), finished=True)
            self.bus.publish(
                run_id,
                {"type": "run_finished", "run_id": run_id, "status": "failed", "error": str(exc)},
            )
        finally:
            await client.close()

    async def _run_task(
        self,
        run_id: str,
        run_dir: Path,
        task: Task,
        client: GroqEvalClient,
        judge: Judge | None,
    ) -> dict[str, Any]:
        spec = task.spec
        completion = await client.complete(
            task.model,
            spec.text,
            max_output_tokens=self.config.max_tokens_for(task.model),
        )

        model_dir = run_dir / slugify(task.model)
        model_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{slugify(spec.prompt_id)}-r{task.repeat_index}"

        output_path = raw_path = reasoning_path = None
        if completion.ok:
            output_path = model_dir / f"{stem}.txt"
            output_path.write_text(completion.content, encoding="utf-8")
            if completion.raw_content != completion.content:
                raw_path = model_dir / f"{stem}.raw.txt"
                raw_path.write_text(completion.raw_content, encoding="utf-8")
            if completion.reasoning:
                reasoning_path = model_dir / f"{stem}.reasoning.txt"
                reasoning_path.write_text(completion.reasoning, encoding="utf-8")

        pricing = self.config.pricing_for(task.model)
        generation_id = self.db.insert_generation(
            {
                "run_id": run_id,
                "prompt_id": spec.prompt_id,
                "query_id": spec.query_id,
                "technique": spec.technique,
                "model": task.model,
                "repeat_index": task.repeat_index,
                "status": "ok" if completion.ok else "error",
                "error": completion.error,
                "output_path": str(output_path) if output_path else None,
                "raw_output_path": str(raw_path) if raw_path else None,
                "reasoning_path": str(reasoning_path) if reasoning_path else None,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.total_tokens,
                "latency_ms": completion.latency_ms,
                "ttft_ms": completion.ttft_ms,
                "throughput_tps": completion.throughput_tps,
                "retries": completion.retries,
                "cost_usd": pricing.cost(
                    completion.prompt_tokens, completion.completion_tokens
                ),
                "word_count": completion.word_count,
                "finish_reason": completion.finish_reason,
                "truncated": int(completion.truncated),
                "reasoning_words": len(completion.reasoning.split()),
                "created_at": utcnow(),
            }
        )

        base = {
            "generation_id": generation_id,
            "prompt_id": spec.prompt_id,
            "technique": spec.technique,
            "model": task.model,
            "repeat_index": task.repeat_index,
        }
        if not completion.ok:
            return {**base, "status": "error", "error": completion.error}

        metrics = deterministic.evaluate(
            completion.content,
            spec,
            reasoning_leak=completion.reasoning_leak,
            truncated=completion.truncated,
            match_tol=float(self.config.facts_cfg.get("match_rel_tolerance", 0.01)),
            contradiction_window=float(
                self.config.facts_cfg.get("contradiction_rel_window", 0.25)
            ),
            overflow_zero_multiple=self.config.overflow_zero_multiple,
        )
        detail = metrics.detail

        self.db.insert_metrics(
            {
                "generation_id": generation_id,
                "run_id": run_id,
                "fact_recall": metrics.get("fact_recall"),
                "facts_total": detail["facts"]["total"],
                "facts_found": len(detail["facts"]["found"]),
                "numeric_grounding": metrics.get("numeric_grounding"),
                "numbers_total": detail["numbers"]["total"],
                "numbers_supported": detail["numbers"]["supported"],
                "numbers_derived": detail["numbers"]["derived"],
                "numbers_unsupported": detail["numbers"]["unsupported"],
                "contradictions": detail["numbers"]["contradicting"],
                "contradiction_free": metrics.get("contradiction_free"),
                "structural_compliance": metrics.get("structural_compliance"),
                "sections_required": detail["structure"]["required"],
                "sections_found": detail["structure"]["found"],
                "section_order_ok": int(bool(detail["structure"]["order_ok"])),
                "length_compliance": metrics.get("length_compliance"),
                "word_limit": spec.word_limit,
                "required_tokens": metrics.get("required_tokens"),
                "format_clean": metrics.get("format_clean"),
                "reasoning_leak": int(completion.reasoning_leak),
                "deterministic_score": deterministic.deterministic_composite(
                    metrics.scores, self.config.weights
                ),
                "details_json": json.dumps(detail),
            }
        )

        judge_overall = None
        if judge is not None and task.judge:
            verdict = await judge.score(spec, completion.content, task.model)
            judge_overall = verdict.overall
            self.db.insert_judge_score(
                {
                    "generation_id": generation_id,
                    "run_id": run_id,
                    "judge_model": judge.model,
                    "self_graded": int(verdict.self_graded),
                    "cached": int(verdict.cached),
                    "overall": verdict.overall,
                    "scores_json": json.dumps(verdict.scores),
                    "justifications_json": json.dumps(verdict.justifications),
                    "status": "ok" if verdict.ok else "error",
                    "error": verdict.error,
                    "prompt_tokens": verdict.prompt_tokens,
                    "completion_tokens": verdict.completion_tokens,
                    "cost_usd": verdict.cost_usd,
                    "created_at": utcnow(),
                }
            )
        elif judge is not None:
            self.db.insert_judge_score(
                {
                    "generation_id": generation_id,
                    "run_id": run_id,
                    "judge_model": judge.model,
                    "self_graded": 0,
                    "cached": 0,
                    "status": "skipped",
                    "error": "judge sampled: this repeat was not selected for grading",
                    "created_at": utcnow(),
                }
            )

        return {
            **base,
            "status": "ok",
            "scores": metrics.scores,
            "judge_overall": judge_overall,
            "word_count": completion.word_count,
            "latency_ms": completion.latency_ms,
        }

    def _compute_consistency(self, run_id: str) -> None:
        rows = self.db.run_rows(run_id)
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            if row["status"] != "ok":
                continue
            grouped.setdefault((row["prompt_id"], row["model"]), []).append(row)

        for (prompt_id, model), group in grouped.items():
            texts = []
            for row in group:
                path = row.get("output_path")
                if path and Path(path).exists():
                    texts.append(Path(path).read_text(encoding="utf-8"))

            result = consistency.evaluate(texts)
            composites = [
                aggregate.composite_score(r, self.config.weights) for r in group
            ]
            self.db.insert_consistency(
                {
                    "run_id": run_id,
                    "prompt_id": prompt_id,
                    "model": model,
                    "n_samples": result.n_samples,
                    "tfidf_cosine": result.tfidf_cosine,
                    "rouge_l": result.rouge_l,
                    "composite_stdev": aggregate.summarize(composites).stdev,
                    "judge_stdev": aggregate.summarize(
                        [r.get("judge_overall") for r in group]
                    ).stdev,
                }
            )

    # ---------------- reads ----------------

    def summary(self, run_id: str) -> dict[str, Any]:
        rows = self.db.run_rows(run_id)
        cons = self.db.consistency_rows(run_id)
        summary = aggregate.build_summary(rows, cons, self.config.weights)
        run = self.db.get_run(run_id) or {}
        summary["run"] = {
            **run,
            "models": json.loads(run.get("models_json") or "[]"),
            "prompt_ids": json.loads(run.get("prompt_ids_json") or "[]"),
            "config": json.loads(run.get("config_json") or "{}"),
        }
        summary["prompt_meta"] = {
            spec.prompt_id: spec.to_dict() for spec in self.specs.values()
        }
        return summary

    def generation_detail(self, generation_id: int) -> dict[str, Any] | None:
        row = self.db.query_one(
            """
            SELECT g.*, m.details_json, m.fact_recall, m.numeric_grounding,
                   m.contradiction_free, m.structural_compliance,
                   m.length_compliance, m.required_tokens, m.format_clean,
                   j.overall AS judge_overall, j.scores_json,
                   j.justifications_json, j.self_graded, j.error AS judge_error
            FROM generations g
            LEFT JOIN metrics m ON m.generation_id = g.id
            LEFT JOIN judge_scores j ON j.generation_id = g.id
            WHERE g.id = ?
            """,
            (generation_id,),
        )
        if not row:
            return None

        spec = self.specs.get(row["prompt_id"])
        text = ""
        if row.get("output_path") and Path(row["output_path"]).exists():
            text = Path(row["output_path"]).read_text(encoding="utf-8")
        reasoning = ""
        if row.get("reasoning_path") and Path(row["reasoning_path"]).exists():
            reasoning = Path(row["reasoning_path"]).read_text(encoding="utf-8")

        return {
            **row,
            "details": json.loads(row.get("details_json") or "{}"),
            "judge_scores": json.loads(row.get("scores_json") or "{}"),
            "judge_justifications": json.loads(row.get("justifications_json") or "{}"),
            "output": text,
            "reasoning": reasoning,
            "prompt_text": spec.text if spec else "",
            "prompt_meta": spec.to_dict() if spec else {},
        }
