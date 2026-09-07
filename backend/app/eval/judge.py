"""LLM-as-judge scoring against an anchored rubric.

Design choices that matter for trustworthiness:

  * temperature 0, so repeat grading of identical text is stable
  * reference-guided: the judge sees the KEY FACTS and the required section
    list, so it grades against the same contract the deterministic metrics use
  * every score must carry a justification, which makes the grade auditable
    instead of an opaque number
  * when the judge model is also the model under test the row is flagged
    ``self_graded``; self-preference bias is well documented and those rows are
    called out in the UI and report rather than silently averaged in
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.budget import ROLE_JUDGE
from ..core.config import EvalConfig
from ..core.groq_client import Completion, GroqEvalClient
from .spec import PromptSpec

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a meticulous evaluator of analytical reports. You grade strictly "
    "against the rubric and the supplied reference facts. You never reward "
    "fluency, length or confident tone on their own. Respond with JSON only."
)


@dataclass
class JudgeResult:
    ok: bool
    overall: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    justifications: dict[str, str] = field(default_factory=dict)
    self_graded: bool = False
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    cached: bool = False
    raw: str = ""


def _rubric_block(rubric: dict[str, Any], compact: bool = False) -> str:
    """Render the rubric for the prompt.

    Compact mode keeps only the endpoint anchors. A judge call carries the
    rubric on every request, so the full anchor ladder is paid for once per
    graded response; the endpoints are what actually calibrate the scale, and
    dropping the middle rungs cuts roughly a third of the prompt with no
    measurable change in the scores.
    """
    lines = []
    lo = rubric.get("scale", {}).get("min", 1)
    hi = rubric.get("scale", {}).get("max", 5)
    for dim in rubric.get("dimensions", []):
        lines.append(f"- {dim['id']} ({dim['name']}), weight {dim.get('weight', 0)}")
        if not compact:
            lines.append(f"    {dim.get('description', '').strip()}")
        anchors = dim.get("anchors", {}) or {}
        levels = sorted(anchors)
        if compact and len(levels) > 2:
            levels = [levels[0], levels[-1]]
        for level in levels:
            lines.append(f"    {level} = {anchors[level]}")
    return f"Score each dimension as an integer from {lo} to {hi}.\n" + "\n".join(lines)


def build_judge_prompt(
    spec: PromptSpec,
    response: str,
    rubric: dict[str, Any],
    compact: bool = False,
) -> str:
    dims = [d["id"] for d in rubric.get("dimensions", [])]
    sections = "\n".join(
        f"  {s.number}. {s.title}" for s in spec.required_sections
    ) or "  (none specified)"
    steps = "\n".join(f"  - {s}" for s in spec.analytical_steps) or "  (none specified)"

    quote = (
        "<max 20 words, quoting the response>"
        if compact
        else "<one sentence quoting the response>"
    )
    schema = ",\n".join(
        f'    "{d}": {{"score": <int>, "justification": "{quote}"}}' for d in dims
    )

    return f"""You are grading one analytical report.

## Reference facts the report was required to rely on exclusively
{spec.key_facts_raw or "(none supplied)"}

## Required report structure
{sections}

## Analytical steps the prompt demanded
{steps}

## Length constraint
Maximum {spec.word_limit or "unspecified"} words. The report is {len(response.split())} words.

## Rubric
{_rubric_block(rubric, compact)}

## Grading instructions
- Judge only against the reference facts above. Any quantitative claim not
  present in, or arithmetically derivable from, those facts is unsupported,
  however plausible it sounds.
- Do not reward length. Exceeding the word limit is a defect, not thoroughness.
- Confident assertion of an unsupported figure must lower both
  factual_accuracy and uncertainty_calibration.
- Quote a short fragment of the report in each justification.

## Report under evaluation
<<<REPORT_START
{response}
REPORT_END

Respond with JSON only, in exactly this shape:
{{
{schema}
}}"""


def _coerce_scores(
    payload: dict[str, Any], rubric: dict[str, Any]
) -> tuple[dict[str, float], dict[str, str]]:
    lo = float(rubric.get("scale", {}).get("min", 1))
    hi = float(rubric.get("scale", {}).get("max", 5))
    scores: dict[str, float] = {}
    justifications: dict[str, str] = {}

    for dim in rubric.get("dimensions", []):
        key = dim["id"]
        entry = payload.get(key)
        if entry is None:
            continue
        if isinstance(entry, dict):
            value = entry.get("score", entry.get("value"))
            note = entry.get("justification", entry.get("reason", ""))
        else:
            value, note = entry, ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        scores[key] = min(max(numeric, lo), hi)
        justifications[key] = str(note)[:600]

    return scores, justifications


def weighted_overall(scores: dict[str, float], rubric: dict[str, Any]) -> float | None:
    """Rubric-weighted mean, rescaled to 0-1."""
    if not scores:
        return None
    lo = float(rubric.get("scale", {}).get("min", 1))
    hi = float(rubric.get("scale", {}).get("max", 5))

    total_w = 0.0
    acc = 0.0
    for dim in rubric.get("dimensions", []):
        key = dim["id"]
        if key not in scores:
            continue
        w = float(dim.get("weight", 0)) or 0.0
        acc += scores[key] * w
        total_w += w

    if total_w <= 0:
        return None
    mean = acc / total_w
    return round((mean - lo) / (hi - lo), 4)


def parse_judge_response(text: str, rubric: dict[str, Any]) -> tuple[dict, dict, str | None]:
    """Extract the JSON verdict, tolerating code fences and stray prose."""
    if not text or not text.strip():
        return {}, {}, "judge returned empty output"

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        block = _JSON_BLOCK.search(candidate)
        if block:
            candidate = block.group(0)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {}, {}, f"unparseable judge JSON: {exc}"

    if not isinstance(payload, dict):
        return {}, {}, "judge JSON was not an object"

    scores, justifications = _coerce_scores(payload, rubric)
    if not scores:
        return {}, {}, "judge JSON contained no recognized dimensions"
    return scores, justifications, None


class Judge:
    def __init__(
        self,
        client: GroqEvalClient,
        config: EvalConfig,
        model: str | None = None,
        cache: Any | None = None,
    ):
        self.client = client
        self.config = config
        self.rubric = config.rubric
        # Overridable per run: the strongest judge is not always the one with
        # remaining rate-limit budget, and picking a judge distinct from the
        # candidates is what avoids self-preference bias in the first place.
        self.model = model or config.judge_model
        self.compact = config.judge_compact_rubric
        # A Database, or None to disable reuse of previous verdicts.
        self.cache = cache if config.judge_cache_enabled else None
        self._rubric_fingerprint = hashlib.sha256(
            json.dumps(self.rubric, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]

    def cache_key(self, spec: PromptSpec, response: str) -> str:
        """Identity of a grading task.

        Includes the rubric fingerprint and the compact flag, so editing the
        rubric or the prompt shape invalidates prior verdicts instead of
        silently serving scores produced under different instructions.
        """
        payload = "\u0000".join(
            [
                self.model,
                self._rubric_fingerprint,
                str(self.config.judge_temperature),
                "compact" if self.compact else "full",
                spec.prompt_id,
                response,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _from_cache(self, key: str, self_graded: bool) -> JudgeResult | None:
        if self.cache is None:
            return None
        row = self.cache.judge_cache_get(key)
        if not row:
            return None
        scores = json.loads(row.get("scores_json") or "{}")
        if not scores:
            return None
        return JudgeResult(
            ok=True,
            overall=row.get("overall"),
            scores=scores,
            justifications=json.loads(row.get("justifications_json") or "{}"),
            self_graded=self_graded,
            # Tokens are attributed to the original call, not this reuse, so a
            # cache hit costs nothing here.
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            cached=True,
        )

    def _to_cache(self, key: str, spec: PromptSpec, result: JudgeResult) -> None:
        if self.cache is None or not result.ok:
            return
        from ..store.db import utcnow

        self.cache.judge_cache_put(
            {
                "cache_key": key,
                "judge_model": self.model,
                "prompt_id": spec.prompt_id,
                "overall": result.overall,
                "scores_json": json.dumps(result.scores),
                "justifications_json": json.dumps(result.justifications),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "created_at": utcnow(),
            }
        )

    async def score(
        self, spec: PromptSpec, response: str, candidate_model: str
    ) -> JudgeResult:
        self_graded = candidate_model == self.model

        if not response.strip():
            return JudgeResult(
                ok=False,
                error="candidate response was empty; nothing to grade",
                self_graded=self_graded,
            )

        key = self.cache_key(spec, response)
        hit = self._from_cache(key, self_graded)
        if hit is not None:
            return hit

        prompt = build_judge_prompt(spec, response, self.rubric, self.compact)
        completion: Completion = await self.client.complete(
            self.model,
            prompt,
            temperature=self.config.judge_temperature,
            top_p=1,
            max_output_tokens=self.config.max_tokens_for(
                self.model, self.config.judge_max_output_tokens
            ),
            system=SYSTEM_PROMPT,
            role=ROLE_JUDGE,
        )

        pricing = self.config.pricing_for(self.model)
        cost = pricing.cost(completion.prompt_tokens, completion.completion_tokens)

        if not completion.ok:
            return JudgeResult(
                ok=False,
                error=completion.error,
                self_graded=self_graded,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost_usd=cost,
            )

        scores, justifications, error = parse_judge_response(
            completion.content, self.rubric
        )
        if error:
            return JudgeResult(
                ok=False,
                error=error,
                self_graded=self_graded,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost_usd=cost,
                raw=completion.content[:2000],
            )

        result = JudgeResult(
            ok=True,
            overall=weighted_overall(scores, self.rubric),
            scores=scores,
            justifications=justifications,
            self_graded=self_graded,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            cost_usd=cost,
            raw=completion.content[:2000],
        )
        self._to_cache(key, spec, result)
        return result
