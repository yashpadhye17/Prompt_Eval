"""Per-model daily token budgeting.

Groq's on-demand tier meters a tokens-per-day allowance separately for each
model. That allowance, not price, is the binding constraint on a benchmark of
this size, and it is spent by judging as much as by generation: a judge call
carries the rubric plus the whole report being graded, so it costs more than
the generation it is grading.

Without accounting, a run discovers the ceiling by slamming into it partway
through, leaving a half-judged matrix whose surviving rows are a biased subset.
This module keeps a local mirror of the counter so a run can be planned against
the remaining allowance and stopped cleanly when it runs out.

The mirror is best-effort by construction. It cannot observe spend from outside
this tool, and the provider's window is rolling rather than a calendar day, so
the real 429 handling in the client remains the backstop. Its job is to make
the common case predictable, not to be authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..store.db import Database, utcday

# Roles a call can play, kept distinct in the ledger because the interesting
# question when a budget runs dry is which of the two consumed it.
ROLE_GENERATE = "generate"
ROLE_JUDGE = "judge"


@dataclass
class ModelBudget:
    model: str
    limit: int | None          # None means untracked / unlimited
    used: int
    calls: int
    by_role: dict[str, Any]

    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "calls": self.calls,
            "by_role": self.by_role,
            "fraction_used": (
                None if not self.limit else round(self.used / self.limit, 4)
            ),
        }


class BudgetLedger:
    """Tracks token spend per model against a daily cap."""

    def __init__(
        self,
        db: Database,
        limits: dict[str, int] | None = None,
        default_limit: int | None = None,
        enabled: bool = True,
    ):
        self.db = db
        self.limits = dict(limits or {})
        self.default_limit = default_limit
        self.enabled = enabled

    def limit_for(self, model: str) -> int | None:
        if not self.enabled:
            return None
        if model in self.limits:
            value = self.limits[model]
            # An explicit null/0 in config opts a model out of tracking.
            return int(value) if value else None
        return self.default_limit

    def status(self, model: str, day: str | None = None) -> ModelBudget:
        usage = self.db.usage_for_day(day).get(model, {})
        return ModelBudget(
            model=model,
            limit=self.limit_for(model),
            used=int(usage.get("tokens", 0)),
            calls=int(usage.get("calls", 0)),
            by_role=usage.get("by_role", {}),
        )

    def snapshot(self, models: list[str], day: str | None = None) -> dict[str, Any]:
        usage = self.db.usage_for_day(day)
        # Include any model that spent tokens today even if it is no longer a
        # configured candidate, since it still drew down its own allowance.
        names = list(dict.fromkeys(list(models) + sorted(usage)))
        return {
            "day": day or utcday(),
            "enabled": self.enabled,
            "default_limit": self.default_limit,
            "models": [self.status(m, day).to_dict() for m in names],
        }

    def record(
        self,
        model: str,
        role: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        if not self.enabled:
            return
        if not (prompt_tokens or completion_tokens):
            return
        self.db.record_token_usage(model, role, prompt_tokens, completion_tokens)

    def can_afford(self, model: str, estimated_tokens: int) -> bool:
        """Whether a call of roughly this size still fits in today's allowance."""
        remaining = self.status(model).remaining
        return remaining is None or remaining >= max(0, estimated_tokens)

    def observed_call_tokens(self, model: str, role: str) -> float | None:
        """Mean total tokens for one call, from this model's own history."""
        if role == ROLE_JUDGE:
            row = self.db.query_one(
                "SELECT AVG(prompt_tokens + completion_tokens) AS avg"
                " FROM judge_scores WHERE judge_model = ? AND status = 'ok'"
                " AND prompt_tokens IS NOT NULL",
                (model,),
            )
        else:
            row = self.db.query_one(
                "SELECT AVG(prompt_tokens + completion_tokens) AS avg"
                " FROM generations WHERE model = ? AND status = 'ok'"
                " AND prompt_tokens IS NOT NULL",
                (model,),
            )
        return row["avg"] if row and row["avg"] else None
