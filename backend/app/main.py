"""FastAPI application entry point.

Run with:  .venv/bin/uvicorn app.main:app --reload --port 8000  (from backend/)
or:        .venv/bin/python -m backend.run
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import load_api_key, load_config
from .core.orchestrator import Orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    if not load_api_key():
        # Surfaced at startup rather than failing deep inside the first run.
        print("WARNING: GROQ_API_KEY not found; runs will fail until it is set.")
    app.state.config = config
    app.state.orchestrator = Orchestrator(config)
    try:
        yield
    finally:
        app.state.orchestrator.db.close()


app = FastAPI(
    title="Prompt Evaluation Framework",
    description=(
        "Benchmarks prompting techniques across models with deterministic "
        "grounding metrics, an LLM-as-judge rubric, and PDF reporting."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "api_key_present": bool(load_api_key())}
