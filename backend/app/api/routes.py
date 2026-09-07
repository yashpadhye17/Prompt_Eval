"""HTTP surface for the evaluation framework."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..core.orchestrator import BudgetExceeded, Orchestrator, RunRequest

router = APIRouter()


class RunPayload(BaseModel):
    models: list[str] | None = Field(default=None)
    prompt_ids: list[str] | None = Field(default=None)
    repeats: int | None = Field(default=None, ge=1, le=10)
    judge_enabled: bool | None = None
    judge_model: str | None = None
    judge_repeats_per_cell: int | None = Field(default=None, ge=0, le=10)

    def to_request(self, orch: Orchestrator) -> RunRequest:
        return RunRequest(
            models=orch.resolve_models(self.models),
            prompt_ids=[s.prompt_id for s in orch.resolve_prompts(self.prompt_ids)],
            repeats=self.repeats or orch.config.repeats,
            judge_enabled=(
                orch.config.judge_enabled
                if self.judge_enabled is None
                else self.judge_enabled
            ),
            judge_model=self.judge_model,
            judge_repeats_per_cell=self.judge_repeats_per_cell,
        )


def _orch(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    orch = _orch(request)
    cfg = orch.config
    return {
        "models": [
            {
                "id": m.id,
                "label": m.label,
                "family": m.family,
                "pricing": {
                    "input_per_mtok": cfg.pricing_for(m.id).input_per_mtok,
                    "output_per_mtok": cfg.pricing_for(m.id).output_per_mtok,
                },
            }
            for m in cfg.models
        ],
        "prompts": [
            {
                "prompt_id": s.prompt_id,
                "query_id": s.query_id,
                "technique": s.technique,
                "variant": s.variant,
                "word_limit": s.word_limit,
                "required_sections": len(s.required_sections),
                "required_tokens": s.required_tokens,
            }
            for s in orch.specs.values()
        ],
        "defaults": {
            "repeats": cfg.repeats,
            "judge_enabled": cfg.judge_enabled,
            "judge_model": cfg.judge_model,
            "judge_repeats_per_cell": cfg.judge_repeats_per_cell,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
            "concurrency": cfg.concurrency,
        },
        # Any model can act as judge. Choosing one that is not among the
        # candidates is what keeps self-preference bias out of the scores, and
        # is also the escape hatch when one model's rate-limit budget is spent.
        "judge_options": sorted(
            {m.id for m in cfg.models} | {cfg.judge_model} | set(cfg.pricing)
        ),
        "presets": cfg.presets,
        "budget": orch.ledger.snapshot(
            list(dict.fromkeys(cfg.model_ids() + [cfg.judge_model] + list(cfg.pricing)))
        ),
        "weights": cfg.weights,
        "rubric": cfg.rubric,
    }


@router.post("/runs/estimate")
def estimate(payload: RunPayload, request: Request) -> dict[str, Any]:
    orch = _orch(request)
    try:
        return orch.estimate(payload.to_request(orch))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/runs")
async def create_run(payload: RunPayload, request: Request) -> dict[str, Any]:
    # Must run on the event loop: launching a run creates an asyncio task.
    orch = _orch(request)
    try:
        req = payload.to_request(orch)
        run_id = orch.launch(req)
    except BudgetExceeded as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "budget": exc.budget},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "run_id": run_id,
        "models": req.models,
        "prompt_ids": req.prompt_ids,
        "repeats": req.repeats,
        "judge_enabled": req.judge_enabled,
        "judge_repeats_per_cell": orch.judge_sample_size(req),
        "total_tasks": len(req.models) * len(req.prompt_ids) * req.repeats,
    }


@router.get("/runs")
def list_runs(request: Request) -> list[dict[str, Any]]:
    orch = _orch(request)
    out = []
    for run in orch.db.list_runs():
        out.append(
            {
                **run,
                "models": json.loads(run.get("models_json") or "[]"),
                "prompt_ids": json.loads(run.get("prompt_ids_json") or "[]"),
                "running": orch.is_running(run["id"]),
                "has_report": bool(orch.db.latest_report(run["id"])),
            }
        )
    return out


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    orch = _orch(request)
    if not orch.db.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    summary = orch.summary(run_id)
    summary["running"] = orch.is_running(run_id)
    summary["has_report"] = bool(orch.db.latest_report(run_id))
    return summary


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, request: Request) -> dict[str, Any]:
    orch = _orch(request)
    if not orch.db.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    orch.cancel(run_id)
    orch.db.delete_run(run_id)
    orch.bus.clear(run_id)
    return {"deleted": run_id}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request) -> dict[str, Any]:
    # Cancelling touches the asyncio task, so stay on the loop thread.
    orch = _orch(request)
    if not orch.db.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return {"cancelled": orch.cancel(run_id)}


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request):
    """Server-sent progress events for a run."""
    orch = _orch(request)
    if not orch.db.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")

    queue = orch.bus.subscribe(run_id)

    async def generator():
        try:
            # Tell a reconnecting client the current state immediately.
            run = orch.db.get_run(run_id) or {}
            yield {
                "event": "snapshot",
                "data": json.dumps(
                    {
                        "type": "snapshot",
                        "status": run.get("status"),
                        "completed": run.get("completed_tasks"),
                        "failed": run.get("failed_tasks"),
                        "total": run.get("total_tasks"),
                        "running": orch.is_running(run_id),
                    }
                ),
            }
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue

                yield {"event": event.get("type", "message"), "data": json.dumps(event)}
                if event.get("type") in ("run_finished", "run_cancelled"):
                    break
        finally:
            orch.bus.unsubscribe(run_id, queue)

    return EventSourceResponse(generator())


@router.get("/generations/{generation_id}")
def get_generation(generation_id: int, request: Request) -> dict[str, Any]:
    orch = _orch(request)
    detail = orch.generation_detail(generation_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"unknown generation {generation_id}")
    return detail


@router.post("/runs/{run_id}/report")
def build_report(run_id: str, request: Request) -> dict[str, Any]:
    from ..report.pdf import build_report as build_pdf

    orch = _orch(request)
    if not orch.db.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")

    summary = orch.summary(run_id)
    if not summary["totals"]["ok"]:
        raise HTTPException(status_code=400, detail="run has no successful generations")

    path = build_pdf(summary, orch.config, run_id, rows=orch.db.run_rows(run_id))
    orch.db.record_report(run_id, str(path))
    return {"run_id": run_id, "path": str(path), "filename": path.name}


@router.get("/runs/{run_id}/report")
def download_report(run_id: str, request: Request):
    orch = _orch(request)
    record = orch.db.latest_report(run_id)
    if not record:
        raise HTTPException(
            status_code=404, detail="no report generated for this run yet"
        )
    from pathlib import Path

    path = Path(record["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="report file is missing on disk")
    return FileResponse(path, media_type="application/pdf", filename=path.name)
