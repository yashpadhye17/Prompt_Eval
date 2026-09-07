# Prompt Eval

A production-style **prompt evaluation harness**. It runs a matrix of prompts × prompting techniques × models × repeats, scores every response with deterministic fact-grounding metrics plus an LLM-as-judge rubric, treats those repeats statistically, and produces a live dashboard plus a single PDF report.

The demo task is grounded analysis of grid failures (Texas 2021 and the 2003 Northeast blackout). The reusable piece is the grading pattern: if a prompt supplies its own reference facts, you can measure hallucination, contradiction, and instruction-following without hand-written labels.

Technical internals (metrics, endpoints, rate-limit ledger, design notes): **[EVAL_FRAMEWORK.md](EVAL_FRAMEWORK.md)**.

---

## What problem this solves

Prompt and model choices are usually made by reading a few outputs. That does not scale, is not repeatable, and cannot tell you whether a change actually reduced invented numbers or just made the prose longer.

This system answers questions like:

- Does Tree-of-Thought beat Chain-of-Thought on *this* task, or only on vibes?
- Is the larger model’s quality gain worth its cost and latency?
- When we edit a prompt, did we break grounding, structure, or length?
- After a provider deprecates a model, does the replacement still satisfy the contract?

It does that by auto-deriving the grading spec from each prompt file (`KEY FACTS`, required sections, word limit, required markers), then scoring every generation against that contract.

---

## Practical production use cases

These are the places a harness like this earns its keep. The current prompts are a stand-in; the pipeline is the product.

### 1. Grounded generation and RAG (the strongest fit)

Any system that must answer **only from retrieved or supplied facts** — policy chatbots, knowledge-base assistants, analyst report generators, ticket summarizers that quote a case file.

**What you measure:** unsupported-numeric rate (invented figures), contradictions of source facts, required-citation markers.

**How you adapt it:** put the retrieved chunks or source table into the prompt the same way `[KEY FACTS: …]` is used here. `backend/app/eval/facts.py` already normalizes units and ranges (`52,000MW` = `52 GW`, `$80–130B` as an interval). Swap the prompt parser in `eval/spec.py` if your sources are JSON or a database instead of a markdown block.

### 2. Pre-ship gate for prompt changes

Treat a prompt like code. Before merging a rewrite of a system prompt, run the same matrix and fail the change if fact recall, grounding, or structural compliance drop outside a confidence interval.

**Where it lives:** a CI job that calls `POST /api/runs`, waits on the SSE stream, and fails if the composite for the changed prompt is worse than the last successful run (stored in SQLite under `runs/`).

### 3. Model bake-off before a vendor lock-in

Compare two or more models on *your* task with **identical generation settings**. Public leaderboards will not tell you whether `gpt-oss-20b` is good enough for an 800-word grounded report, or whether the 120B variant’s extra cost buys anything on hallucination.

Operational metrics (latency, TTFT, tokens, estimated USD) sit next to quality metrics so the decision is cost-quality, not quality alone.

### 4. Prompting-technique selection

Five techniques ship in this repo (CoT, ToT, Role, Few-Shot, ReAct) for the same two questions. Production teams often have the same choice: extra reasoning tokens vs. a shorter instruction. The heatmap is the evidence for that choice, with n=3 repeats and 95% CIs so overlapping rankings are not oversold.

### 5. Regression when the ground moves

Providers deprecate models (this repo already lost `llama-3.1-8b-instant`), silently update them, and change tokenizer or safety behavior. Re-run the frozen matrix after any such event. Same prompts, same metrics, comparable PDF.

### 6. Audit trail for regulated or high-stakes outputs

Finance, energy, insurance, and internal ops teams that generate numbers-heavy text need to show *why* a model was allowed to ship. Each run stores raw outputs, per-number verdicts (provided / derived / contradicting / unsupported), judge justifications, and a PDF with methodology and caveats (self-grading, sample size, coverage holes).

That is closer to an evaluation record than a chat log.

### 7. Teaching and internal enablement

A concrete example of how production eval is actually done: deterministic checks first, LLM-as-judge as a correlated second opinion, repeats for variance, explicit coverage when a run is incomplete, and refusal to rank models on a biased subset.

---

## Where it is *not* the right tool

- Open-ended creative writing with no reference facts (nothing to ground against).
- Tasks whose correctness is only human preference (use a preference dataset or human raters).
- Multi-turn agents with tools and environment state (you would need trajectory-level scoring on top of this).
- Live production traffic scoring at request volume (this is an offline batch harness, not an online monitor).

---

## Quick start (localhost)

**Prerequisites:** Python 3.11+, Node.js 18+, a [Groq](https://console.groq.com/) API key.

```bash
git clone https://github.com/yashpadhye17/Prompt_Eval.git
cd Prompt_Eval

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

Create `src/core/.env` (never commit this file):

```env
GROQ_API_KEY=your_key_here
```

Two terminals:

```bash
# terminal 1 — API
cd backend
../.venv/bin/uvicorn app.main:app --port 8000
```

```bash
# terminal 2 — dashboard
cd frontend
npm run dev
```

Open **http://localhost:5173**. The UI proxies `/api` to port 8000.

In the launcher, start with the **Smoke** preset (one model, two prompts, one repeat) to confirm the pipeline, then **Free tier** for a full matrix that fits Groq’s on-demand daily token cap. Use **Estimate** before **Start run**. When the run finishes, generate and download the PDF.

### Without the browser

```bash
curl -X POST localhost:8000/api/runs/estimate -H 'content-type: application/json' \
  -d '{"repeats":1,"models":["openai/gpt-oss-20b"],"judge_model":"qwen/qwen3.8-27b"}'

curl -X POST localhost:8000/api/runs -H 'content-type: application/json' \
  -d '{"models":["openai/gpt-oss-20b"],"repeats":1,"judge_model":"qwen/qwen3.8-27b"}'
```

Full endpoint list: [EVAL_FRAMEWORK.md](EVAL_FRAMEWORK.md).

---

## How scoring works (short)

| Tier | What | Why it matters in production |
| --- | --- | --- |
| Deterministic | Fact recall, numeric grounding, contradictions, structure, length, required markers, format leaks | Cheap, reproducible, no extra model spend. Carries most of the composite weight. |
| LLM-as-judge | Anchored 1–5 rubric, temperature 0, JSON verdicts | Covers qualities rules cannot (reasoning depth, tone). Sampled on the free tier so judging does not exhaust the daily budget. Self-graded rows are flagged. |
| Reliability | Mean, stdev, 95% CI, TF-IDF / ROUGE-L across repeats | Stops you ranking two models whose intervals overlap. |
| Operational | Latency, TTFT, tokens, estimated cost, errors / retries | The bake-off is incomplete without this. |

Composite score is a configurable weighted blend in `config/eval_config.yaml`. Missing metrics (for example an unjudged row) **redistribute remaining weights** instead of scoring zero.

On Groq’s free tier the judge is sampled (first repeat of each cell by default) and a local token ledger refuses a run that cannot finish. Details: [EVAL_FRAMEWORK.md](EVAL_FRAMEWORK.md).

---

## Repository layout

```
backend/app/          FastAPI, orchestrator, metrics, PDF
frontend/             Vite + React dashboard (live progress, leaderboard, drill-down)
config/               eval_config.yaml, rubric.yaml, legacy model configs
src/prompts/          The 10 graded prompts (2 queries × 5 techniques)
src/core/             Original one-shot clients (still work; write to output/)
runs/                 Per-run artifacts (gitignored): outputs, SQLite, PDFs
```

The original scripts `src/core/groqclient.py` and `src/core/openaiclient.py` are unchanged and still dump per-prompt PDFs under `output/`. The framework never writes there.

---

## Adapting this to another task

1. Replace files under `src/prompts/` with your prompts. Keep a machine-readable facts block, a section list, and a length cap if you want the current graders to work unchanged.
2. Set candidate models, pricing, and `judge.model` in `config/eval_config.yaml`. Prefer a judge that is **not** among the candidates.
3. Edit `config/rubric.yaml` if your quality dimensions differ (for example “citation quality” instead of “causal reasoning”).
4. If facts live in JSON or a retrieval index, change `eval/spec.py` only; `facts.py` and the rest of the pipeline stay.

---

## Tests

```bash
.venv/bin/python backend/tests/test_facts.py    # numeric grounding
.venv/bin/python backend/tests/test_budget.py   # sampling, ledger, cache keys
```

---

## Security

- `GROQ_API_KEY` belongs in `src/core/.env` or the environment. `.env` is gitignored.
- Do not commit `runs/` (raw model outputs and the SQLite DB).
- Cost figures in reports come from the price table in `config/eval_config.yaml`. Verify those rates against current Groq pricing before quoting them.

---

## License

MIT. See [LICENSE](LICENSE).
