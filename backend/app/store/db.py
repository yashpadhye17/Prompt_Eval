"""SQLite persistence for evaluation runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Columns added after the first release. CREATE TABLE IF NOT EXISTS will not
# add them to a database that already exists, so they are applied explicitly.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("runs", "judge_repeats_per_cell", "INTEGER"),
    ("judge_scores", "cached", "INTEGER NOT NULL DEFAULT 0"),
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utcday() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Database:
    """Thin SQLite wrapper.

    Writes are serialized through a lock because the orchestrator fans out
    across asyncio tasks that all persist into the same file.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            for table, column, decl in _MIGRATIONS:
                existing = {
                    r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")
                }
                if column not in existing:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
                    )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------- generic helpers ----------------

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------------- runs ----------------

    def create_run(
        self,
        run_id: str,
        models: list[str],
        prompt_ids: list[str],
        repeats: int,
        judge_enabled: bool,
        judge_model: str | None,
        config: dict[str, Any],
        total_tasks: int,
        judge_repeats_per_cell: int | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO runs (id, created_at, status, models_json, prompt_ids_json,
                              repeats, judge_enabled, judge_model,
                              judge_repeats_per_cell, config_json, total_tasks)
            VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                utcnow(),
                json.dumps(models),
                json.dumps(prompt_ids),
                repeats,
                int(judge_enabled),
                judge_model,
                judge_repeats_per_cell,
                json.dumps(config),
                total_tasks,
            ),
        )

    def set_run_status(
        self, run_id: str, status: str, error: str | None = None, finished: bool = False
    ) -> None:
        self.execute(
            "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, error, utcnow() if finished else None, run_id),
        )

    def bump_run_progress(self, run_id: str, ok: bool) -> None:
        """Count a finished task. completed_tasks tracks all attempts that are done."""
        if ok:
            self.execute(
                "UPDATE runs SET completed_tasks = completed_tasks + 1 WHERE id = ?",
                (run_id,),
            )
        else:
            self.execute(
                "UPDATE runs SET completed_tasks = completed_tasks + 1,"
                " failed_tasks = failed_tasks + 1 WHERE id = ?",
                (run_id,),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def list_runs(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM runs ORDER BY created_at DESC")

    def delete_run(self, run_id: str) -> None:
        self.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    # ---------------- generations ----------------

    def insert_generation(self, row: dict[str, Any]) -> int:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = (
            f"INSERT INTO generations ({', '.join(cols)}) VALUES ({placeholders})"
            " ON CONFLICT(run_id, prompt_id, model, repeat_index) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("run_id",))
        )
        cur = self.execute(sql, [row[c] for c in cols])
        if cur.lastrowid:
            return int(cur.lastrowid)
        found = self.query_one(
            "SELECT id FROM generations WHERE run_id=? AND prompt_id=? AND model=? AND repeat_index=?",
            (row["run_id"], row["prompt_id"], row["model"], row["repeat_index"]),
        )
        return int(found["id"]) if found else -1

    def insert_metrics(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = (
            f"INSERT INTO metrics ({', '.join(cols)}) VALUES ({placeholders})"
            " ON CONFLICT(generation_id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in cols if c != "generation_id")
        )
        self.execute(sql, [row[c] for c in cols])

    def insert_judge_score(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = (
            f"INSERT INTO judge_scores ({', '.join(cols)}) VALUES ({placeholders})"
            " ON CONFLICT(generation_id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in cols if c != "generation_id")
        )
        self.execute(sql, [row[c] for c in cols])

    def insert_consistency(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = (
            f"INSERT INTO consistency ({', '.join(cols)}) VALUES ({placeholders})"
            " ON CONFLICT(run_id, prompt_id, model) DO UPDATE SET "
            + ", ".join(
                f"{c}=excluded.{c}" for c in cols if c not in ("run_id", "prompt_id", "model")
            )
        )
        self.execute(sql, [row[c] for c in cols])

    def record_report(self, run_id: str, path: str) -> None:
        self.execute(
            "INSERT INTO reports (run_id, path, created_at) VALUES (?, ?, ?)",
            (run_id, path, utcnow()),
        )

    def latest_report(self, run_id: str) -> dict[str, Any] | None:
        return self.query_one(
            "SELECT * FROM reports WHERE run_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (run_id,),
        )

    # ---------------- joined reads ----------------

    def run_rows(self, run_id: str) -> list[dict[str, Any]]:
        """Every generation in a run joined with its metrics and judge score."""
        return self.query(
            """
            SELECT g.*,
                   m.fact_recall, m.facts_total, m.facts_found,
                   m.numeric_grounding, m.numbers_total, m.numbers_supported,
                   m.numbers_derived, m.numbers_unsupported,
                   m.contradictions, m.contradiction_free,
                   m.structural_compliance, m.sections_required, m.sections_found,
                   m.section_order_ok, m.length_compliance, m.word_limit,
                   m.required_tokens, m.format_clean, m.reasoning_leak,
                   m.deterministic_score, m.details_json,
                   j.overall AS judge_overall, j.scores_json, j.justifications_json,
                   j.self_graded, j.status AS judge_status, j.error AS judge_error,
                   j.cost_usd AS judge_cost_usd, j.cached AS judge_cached
            FROM generations g
            LEFT JOIN metrics m ON m.generation_id = g.id
            LEFT JOIN judge_scores j ON j.generation_id = g.id
            WHERE g.run_id = ?
            ORDER BY g.prompt_id, g.model, g.repeat_index
            """,
            (run_id,),
        )

    def consistency_rows(self, run_id: str) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM consistency WHERE run_id = ?", (run_id,))

    # ---------------- token ledger ----------------

    def record_token_usage(
        self,
        model: str,
        role: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        day: str | None = None,
    ) -> None:
        """Add one call's spend to the model's allowance for the day.

        Only successful calls are recorded: a request the provider rejects with
        a 429 never reaches the model and so does not draw down the allowance.
        """
        self.execute(
            """
            INSERT INTO token_usage (model, day, role, calls, prompt_tokens,
                                     completion_tokens, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(model, day, role) DO UPDATE SET
                calls = calls + 1,
                prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                completion_tokens = completion_tokens + excluded.completion_tokens,
                updated_at = excluded.updated_at
            """,
            (
                model,
                day or utcday(),
                role,
                int(prompt_tokens or 0),
                int(completion_tokens or 0),
                utcnow(),
            ),
        )

    def usage_for_day(self, day: str | None = None) -> dict[str, dict[str, Any]]:
        """Tokens spent per model today, broken down by role."""
        rows = self.query(
            "SELECT model, role, calls, prompt_tokens, completion_tokens"
            " FROM token_usage WHERE day = ?",
            (day or utcday(),),
        )
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = out.setdefault(
                row["model"], {"tokens": 0, "calls": 0, "by_role": {}}
            )
            spent = int(row["prompt_tokens"]) + int(row["completion_tokens"])
            entry["tokens"] += spent
            entry["calls"] += int(row["calls"])
            entry["by_role"][row["role"]] = {
                "tokens": spent,
                "calls": int(row["calls"]),
            }
        return out

    def prune_token_usage(self, keep_days: int = 14) -> None:
        self.execute(
            "DELETE FROM token_usage WHERE day < date('now', ?)",
            (f"-{int(keep_days)} days",),
        )

    def backfill_token_usage(self, day: str | None = None) -> int:
        """Seed today's ledger from generations and judge scores already stored.

        The ledger table is new; without this, a restart would report a full
        allowance even after the day's quota is gone. Only runs when the day
        has no ledger rows, so live recording is never double-counted.
        """
        day = day or utcday()
        if self.query_one("SELECT 1 AS ok FROM token_usage WHERE day = ? LIMIT 1", (day,)):
            return 0

        gen_rows = self.query(
            "SELECT model, COUNT(*) AS calls,"
            " COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,"
            " COALESCE(SUM(completion_tokens), 0) AS completion_tokens"
            " FROM generations"
            " WHERE status = 'ok' AND created_at LIKE ?"
            " AND (prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL)"
            " GROUP BY model",
            (f"{day}%",),
        )
        judge_rows = self.query(
            "SELECT judge_model AS model, COUNT(*) AS calls,"
            " COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,"
            " COALESCE(SUM(completion_tokens), 0) AS completion_tokens"
            " FROM judge_scores"
            " WHERE status = 'ok' AND created_at LIKE ?"
            " AND COALESCE(cached, 0) = 0"
            " AND (prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL)"
            " GROUP BY judge_model",
            (f"{day}%",),
        )

        inserted = 0
        for role, rows in (("generate", gen_rows), ("judge", judge_rows)):
            for row in rows:
                if not (row["prompt_tokens"] or row["completion_tokens"]):
                    continue
                self.execute(
                    """
                    INSERT INTO token_usage (model, day, role, calls, prompt_tokens,
                                             completion_tokens, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["model"],
                        day,
                        role,
                        int(row["calls"]),
                        int(row["prompt_tokens"]),
                        int(row["completion_tokens"]),
                        utcnow(),
                    ),
                )
                inserted += 1
        return inserted

    # ---------------- judge cache ----------------

    def judge_cache_get(self, cache_key: str) -> dict[str, Any] | None:
        return self.query_one(
            "SELECT * FROM judge_cache WHERE cache_key = ?", (cache_key,)
        )

    def judge_cache_put(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        self.execute(
            f"INSERT OR REPLACE INTO judge_cache ({', '.join(cols)})"
            f" VALUES ({placeholders})",
            [row[c] for c in cols],
        )
