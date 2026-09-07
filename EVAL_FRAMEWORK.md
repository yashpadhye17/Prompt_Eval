# Prompt Evaluation Framework (technical notes)

Companion to the [README](README.md), which covers use cases, localhost setup, and
how to adapt the harness. This file is the internals: metrics, layout, API,
rate-limit ledger, and design constraints.

The original scripts (`src/core/groqclient.py`, `src/core/openaiclient.py`) are
untouched and still work. This framework writes to `runs/` and never touches
`output/`.

## Why these metrics

Every prompt in this benchmark instructs the model to "Base analysis ONLY on
provided facts" and supplies a `[KEY FACTS: ...]` block. That constraint is what
makes automatic grading possible: each number in a response can be traced back to
the prompt or flagged as invented. The grading contract is parsed out of each
prompt file itself, so metrics cannot drift out of sync with the prompts.

| Metric | Meaning |
| --- | --- |
| Fact recall | Share of the prompt's KEY FACTS the response actually cites |
| Numeric grounding | Share of all quantities that were provided or derivable |
| Contradiction-free | Penalizes figures that restate a supplied fact incorrectly |
| Structural compliance | Required numbered sections present, in order |
| Length compliance | Adherence to the prompt's word cap |
| Required markers | Presence of demanded literals (`VERIFY_SOURCE`, `INFERRED:`) |
| Format cleanliness | Penalizes leaked `<think>` reasoning, truncation, refusals |
| Judge overall | Rubric-weighted LLM-as-judge score |

Numbers are normalized before comparison, so `52,000MW` matches `52 GW`, ranges
like `$80-130B` match any value inside the interval, and spelled-out quantities
("four days") are recognized. A figure is classed as:

- **provided** — matches a KEY FACT, or appears elsewhere in the prompt (the
  Few-Shot prompts supply comparison examples; citing those is following
  instructions, not hallucinating)
- **derived** — equals a simple arithmetic combination of supplied facts, with the
  formula recorded
- **contradicting** — same dimension as a supplied fact but materially different
- **unsupported** — invented

## Layout

```
backend/app/
  main.py                  FastAPI app
  api/routes.py            REST + SSE endpoints
  core/config.py           config loading, pricing, per-model token limits
  core/budget.py           per-model daily token ledger and remaining-budget checks
  core/groq_client.py      async client: TTFT, usage, <think> stripping, retries
  core/orchestrator.py     run lifecycle, persistence, progress events
  eval/spec.py             parses the grading contract out of each prompt
  eval/facts.py            numeric extraction, normalization, classification
  eval/deterministic.py    the tier-1 metrics
  eval/judge.py            rubric-based LLM-as-judge
  eval/consistency.py      TF-IDF cosine + ROUGE-L across repeats
  eval/aggregate.py        mean/stdev/95% CI, composites, leaderboards
  report/pdf.py charts.py  consolidated PDF
  store/db.py schema.sql   SQLite
  tests/                   fact-engine checks, UI screenshot script
frontend/                  Vite + React + TypeScript + Recharts
config/eval_config.yaml    models, pricing, weights, limits
config/rubric.yaml         anchored 1-5 judge rubric
runs/<run_id>/             raw outputs, reasoning, generated PDF
```

## Running it

Install dependencies (adds FastAPI, uvicorn, tenacity, numpy, scikit-learn,
matplotlib, sse-starlette to the existing set):

```bash
pip install -r requirements.txt
```

`GROQ_API_KEY` is read from `src/core/.env` (the location the original scripts
already use), a root `.env`, or the environment.

Start the API from the `backend/` directory:

```bash
cd backend
uvicorn app.main:app --port 8000
```

Start the dashboard in a second terminal:

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Vite proxies `/api` to port 8000, so the browser stays on one origin and SSE and
the PDF download work without CORS configuration.

### Without the UI

```bash
# price the matrix before spending anything
curl -X POST localhost:8000/api/runs/estimate -H 'content-type: application/json' \
  -d '{"repeats":3}'

# launch, choosing a judge that is not among the candidates
curl -X POST localhost:8000/api/runs -H 'content-type: application/json' \
  -d '{"models":["openai/gpt-oss-20b"],"repeats":3,"judge_model":"qwen/qwen3.8-27b"}'

# follow progress, then build the report
curl -N localhost:8000/api/runs/<run_id>/events
curl -X POST localhost:8000/api/runs/<run_id>/report
curl -o report.pdf localhost:8000/api/runs/<run_id>/report
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/config` | models, prompts, judge options, weights, rubric |
| POST | `/api/runs/estimate` | dry-run token and cost estimate |
| POST | `/api/runs` | launch a run |
| GET | `/api/runs` | list runs |
| GET | `/api/runs/{id}` | full aggregated summary |
| GET | `/api/runs/{id}/events` | SSE progress stream |
| POST | `/api/runs/{id}/cancel` | cancel a running run |
| DELETE | `/api/runs/{id}` | delete a run |
| GET | `/api/generations/{id}` | one response with per-number verdicts |
| POST | `/api/runs/{id}/report` | build the PDF |
| GET | `/api/runs/{id}/report` | download the PDF |

## Groq rate limits (important)

The on-demand tier imposes limits that materially constrain a benchmark of this
size. The framework is now sized around them rather than discovering them
mid-run:

- **Tokens per day, 200,000 per model.** Generation of 30 cells per model fits
  (~80–90k). Judging every cell on one model does not (~400k). The default
  **Free tier** preset therefore judges only the first repeat of each cell
  (~20 calls) with `qwen/qwen3.8-27b`, which is not a candidate, so the two
  budgets stay independent.
- **Output tokens per minute, 1,000 on the qwen models.** That is less than one
  800-word report needs, so those models cannot complete this task as
  *candidates* and stay commented out. They work as a judge: a JSON verdict
  fits in 900 tokens.

A local ledger in `token_usage` tracks today's spend per model. The estimate
endpoint and the launcher compare the planned matrix against what is left
(minus a 5% reserve) and refuse a run that cannot finish. The ledger is
best-effort — it cannot see spend from outside this tool — so the client
still fail-fasts permanent 429s as a backstop.

Judge verdicts are cached by content hash. Re-running unchanged output costs
nothing, because judging is temperature 0.

When rate limits still truncate a run, the PDF's **Coverage** section reports
each model's completed-versus-expected sample count and excludes under-covered
models from the headline ranking.

## Design notes

- **Token ceiling is deliberately generous (8192).** `qwen3.6-27b` spends
  1,800-2,200 words on internal reasoning before answering and returned empty
  output at 2048 and truncated output at 4096. A ceiling that truncates one model
  but not others invalidates the comparison; the prompts' own 800-word limit is
  what constrains length.
- **All candidates share identical generation settings.** The original two config
  files used different temperatures and token limits, which made the models
  non-comparable.
- **`gpt-oss` returns reasoning in a separate field**, billed as output tokens but
  absent from the visible answer, so cost and visible length come from different
  sources. `qwen3.6` instead emits `<think>` inline; that is stripped before
  scoring and recorded as a format violation.
- **Self-grading is flagged, not hidden.** Rows where the judge is also the
  candidate are marked `self_graded` in the UI, the leaderboard and the caveats.
- **Missing metrics redistribute their weight** rather than scoring zero, so a
  failed judge call lowers confidence instead of silently tanking a model.

## Tests

```bash
.venv/bin/python backend/tests/test_facts.py     # numeric grounding engine
.venv/bin/python backend/tests/test_budget.py    # judge sampling, ledger, cache
.venv/bin/python backend/tests/shoot_ui.py       # dashboard screenshots
```
