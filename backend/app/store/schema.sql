-- Prompt evaluation framework persistence.
-- Raw model outputs live on disk under runs/<run_id>/; this DB holds
-- metadata, metrics and judge scores.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL,          -- pending|running|completed|failed|cancelled
    models_json       TEXT NOT NULL,
    prompt_ids_json   TEXT NOT NULL,
    repeats           INTEGER NOT NULL,
    judge_enabled     INTEGER NOT NULL,
    judge_model       TEXT,
    -- How many of each cell's repeats were judged. Below `repeats` the judge
    -- scores are a stratified sample rather than full coverage.
    judge_repeats_per_cell INTEGER,
    config_json       TEXT NOT NULL,
    total_tasks       INTEGER NOT NULL DEFAULT 0,
    completed_tasks   INTEGER NOT NULL DEFAULT 0,
    failed_tasks      INTEGER NOT NULL DEFAULT 0,
    error             TEXT
);

-- One row per (prompt, model, repeat) generation.
CREATE TABLE IF NOT EXISTS generations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    prompt_id          TEXT NOT NULL,          -- e.g. Q1/prompt 1a
    query_id           TEXT NOT NULL,          -- Q1 | Q2
    technique          TEXT NOT NULL,          -- CoT | ToT | Role | Few-Shot | ReAct
    model              TEXT NOT NULL,
    repeat_index       INTEGER NOT NULL,
    status             TEXT NOT NULL,          -- ok|error
    error              TEXT,
    output_path        TEXT,                   -- file holding cleaned visible output
    raw_output_path    TEXT,                   -- file holding pre-strip content
    reasoning_path     TEXT,                   -- separate reasoning field, if any
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    total_tokens       INTEGER,
    latency_ms         REAL,
    ttft_ms            REAL,
    throughput_tps     REAL,
    retries            INTEGER NOT NULL DEFAULT 0,
    cost_usd           REAL,
    word_count         INTEGER,
    finish_reason      TEXT,
    truncated          INTEGER NOT NULL DEFAULT 0,
    reasoning_words    INTEGER,
    created_at         TEXT NOT NULL,
    UNIQUE (run_id, prompt_id, model, repeat_index)
);

CREATE INDEX IF NOT EXISTS idx_generations_run ON generations(run_id);

-- Deterministic metric results, one row per generation.
CREATE TABLE IF NOT EXISTS metrics (
    generation_id           INTEGER PRIMARY KEY REFERENCES generations(id) ON DELETE CASCADE,
    run_id                  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    fact_recall             REAL,
    facts_total             INTEGER,
    facts_found             INTEGER,
    numeric_grounding       REAL,
    numbers_total           INTEGER,
    numbers_supported       INTEGER,
    numbers_derived         INTEGER,
    numbers_unsupported     INTEGER,
    contradictions          INTEGER,
    contradiction_free      REAL,
    structural_compliance   REAL,
    sections_required       INTEGER,
    sections_found          INTEGER,
    section_order_ok        INTEGER,
    length_compliance       REAL,
    word_limit              INTEGER,
    required_tokens         REAL,
    format_clean            REAL,
    reasoning_leak          INTEGER,
    deterministic_score     REAL,
    details_json            TEXT
);

-- LLM-as-judge results, one row per generation.
CREATE TABLE IF NOT EXISTS judge_scores (
    generation_id     INTEGER PRIMARY KEY REFERENCES generations(id) ON DELETE CASCADE,
    run_id            TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    judge_model       TEXT NOT NULL,
    self_graded       INTEGER NOT NULL DEFAULT 0,
    cached            INTEGER NOT NULL DEFAULT 0,
    overall           REAL,
    scores_json       TEXT,
    justifications_json TEXT,
    status            TEXT NOT NULL,          -- ok|error|skipped
    error             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL,
    created_at        TEXT NOT NULL
);

-- Cross-repeat consistency, one row per (prompt, model) cell.
CREATE TABLE IF NOT EXISTS consistency (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    prompt_id        TEXT NOT NULL,
    model            TEXT NOT NULL,
    n_samples        INTEGER NOT NULL,
    tfidf_cosine     REAL,
    rouge_l          REAL,
    composite_stdev  REAL,
    judge_stdev      REAL,
    UNIQUE (run_id, prompt_id, model)
);

CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Per-model, per-day token spend. Providers meter a tokens-per-day allowance
-- separately for each model, so the key is (model, day). Deliberately not
-- scoped to a run: the allowance is shared across every run on that day.
-- This is a local best-effort mirror of the provider's counter -- it cannot
-- see spend from outside this tool -- so it informs planning while real 429
-- handling stays in place as the backstop.
CREATE TABLE IF NOT EXISTS token_usage (
    model             TEXT NOT NULL,
    day               TEXT NOT NULL,          -- UTC date, YYYY-MM-DD
    role              TEXT NOT NULL,          -- generate|judge
    calls             INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (model, day, role)
);

-- Judge verdicts keyed by content, so re-running a run over unchanged output
-- costs nothing. Judging is deterministic by design (temperature 0), which is
-- what makes the cache sound rather than a shortcut.
CREATE TABLE IF NOT EXISTS judge_cache (
    cache_key           TEXT PRIMARY KEY,
    judge_model         TEXT NOT NULL,
    prompt_id           TEXT NOT NULL,
    overall             REAL,
    scores_json         TEXT,
    justifications_json TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    created_at          TEXT NOT NULL
);
