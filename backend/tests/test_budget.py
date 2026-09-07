"""Checks for judge sampling, the daily token ledger, and verdict cache keys."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_config
from app.core.orchestrator import BudgetExceeded, Orchestrator, RunRequest
from app.eval.judge import Judge
from app.eval.spec import load_prompt_specs
from app.store.db import Database


def _orch(db_path: Path) -> Orchestrator:
    return Orchestrator(load_config(), db=Database(db_path))


def test_judge_samples_first_repeats(tmp: Path) -> None:
    orch = _orch(tmp / "a.db")
    specs = list(orch.specs.values())[:2]
    req = RunRequest(
        models=["openai/gpt-oss-20b"],
        prompt_ids=[s.prompt_id for s in specs],
        repeats=3,
        judge_enabled=True,
        judge_model="qwen/qwen3.8-27b",
        judge_repeats_per_cell=1,
    )
    tasks = orch.plan_tasks(req)
    assert len(tasks) == 6
    judged = [t for t in tasks if t.judge]
    skipped = [t for t in tasks if not t.judge]
    assert len(judged) == 2
    assert {t.repeat_index for t in judged} == {0}
    assert {t.repeat_index for t in skipped} == {1, 2}

    est = orch.estimate(req)
    assert est["judge_calls"] == 2
    assert est["judge_sampled"] is True
    assert est["judge_repeats_per_cell"] == 1


def test_zero_repeats_per_cell_means_all(tmp: Path) -> None:
    orch = _orch(tmp / "b.db")
    req = RunRequest(
        models=["openai/gpt-oss-20b"],
        prompt_ids=[next(iter(orch.specs))],
        repeats=3,
        judge_enabled=True,
        judge_repeats_per_cell=0,
    )
    assert orch.judge_sample_size(req) == 3
    assert all(t.judge for t in orch.plan_tasks(req))


def test_budget_blocks_overspend(tmp: Path) -> None:
    orch = _orch(tmp / "c.db")
    model = "openai/gpt-oss-20b"
    orch.ledger.record(model, "generate", 100_000, 100_000)
    status = orch.ledger.status(model)
    assert status.used == 200_000
    assert status.remaining == 0

    check = orch.budget_check({model: 1_000})
    assert check["fits"] is False
    assert model in check["blocking_models"]


def test_launch_refuses_when_budget_exhausted(tmp: Path) -> None:
    orch = _orch(tmp / "d.db")
    orch.ledger.record("openai/gpt-oss-20b", "generate", 190_000, 10_000)
    req = RunRequest(
        models=["openai/gpt-oss-20b"],
        prompt_ids=[next(iter(orch.specs))],
        repeats=1,
        judge_enabled=False,
    )
    try:
        orch.launch(req)
    except BudgetExceeded as exc:
        assert "openai/gpt-oss-20b" in exc.budget["blocking_models"]
    else:
        raise AssertionError("expected BudgetExceeded")


def test_cache_key_changes_with_rubric_shape() -> None:
    cfg = load_config()
    spec = load_prompt_specs(cfg.prompts_root)[0]
    judge = Judge.__new__(Judge)
    judge.model = "qwen/qwen3.8-27b"
    judge.config = cfg
    judge.compact = True
    judge._rubric_fingerprint = "abc"
    compact = Judge.cache_key(judge, spec, "hello")
    judge.compact = False
    full = Judge.cache_key(judge, spec, "hello")
    assert compact != full
    judge.compact = True
    judge.model = "other"
    assert Judge.cache_key(judge, spec, "hello") != compact


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        test_judge_samples_first_repeats(tmp)
        print("  ok   first repeat of each cell is judged")
        test_zero_repeats_per_cell_means_all(tmp)
        print("  ok   repeats_per_cell=0 grades every repeat")
        test_budget_blocks_overspend(tmp)
        print("  ok   exhausted allowance fails the preflight")
        test_launch_refuses_when_budget_exhausted(tmp)
        print("  ok   launch refuses a doomed matrix")
        test_cache_key_changes_with_rubric_shape()
        print("  ok   cache key includes rubric shape")
    print("\nall budget checks passed")


if __name__ == "__main__":
    main()
