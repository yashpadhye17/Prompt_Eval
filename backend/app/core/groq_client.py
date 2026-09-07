"""Async Groq chat client instrumented for evaluation.

Captures the operational signals a production eval needs:

  * time-to-first-token, total latency, throughput
  * prompt / completion token usage from the API's own accounting
  * reasoning content kept separate from the visible answer

Two model families behave differently and are normalized here:

  * ``openai/gpt-oss-*`` return chain-of-thought in a distinct ``reasoning``
    delta, which never appears in ``content`` but *is* billed as completion
    tokens. Cost and visible length therefore come from different sources.
  * ``qwen/qwen3.6-*`` emit ``<think>...</think>`` inline in ``content``. Left
    in place it would inflate word counts and pollute judge scoring, so it is
    stripped into ``reasoning`` and recorded as a format violation.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from groq import AsyncGroq
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from .budget import ROLE_GENERATE, BudgetLedger
from .config import EvalConfig, load_api_key

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)

# A 429 is usually a transient burst worth retrying, but two kinds are not:
#   * the request's expected output exceeds the per-minute token budget, so it
#     can never succeed at this max_tokens no matter how long we wait
#   * the daily token allowance is spent, which no practical backoff outlasts
# Retrying either one only burns wall-clock time, so fail fast and surface the
# actual remedy (lower max_tokens, or a higher tier) on the row.
_PERMANENT_429 = (
    "request too large",
    "reduce max_tokens",
    "reduce your message size",
    "tokens per day",
    "requests per day",
)


def is_permanent_rate_limit(error: BaseException) -> bool:
    return any(marker in str(error).lower() for marker in _PERMANENT_429)


def split_reasoning(content: str) -> tuple[str, str, bool]:
    """Strip inline ``<think>`` blocks.

    Returns ``(visible, extracted_reasoning, had_leak)``.
    """
    if not content or "<think>" not in content.lower():
        return content, "", False

    captured = [m.group(1) for m in _THINK_BLOCK.finditer(content)]
    visible = _THINK_BLOCK.sub("", content)

    # A truncated response can leave <think> unterminated; drop the remainder.
    unclosed = _UNCLOSED_THINK.search(visible)
    if unclosed:
        captured.append(unclosed.group(1))
        visible = _UNCLOSED_THINK.sub("", visible)

    return visible.strip(), "\n".join(c.strip() for c in captured), True


@dataclass
class Completion:
    """One model response plus its operational telemetry."""

    model: str
    content: str                 # visible answer, reasoning removed
    raw_content: str             # exactly what came back in `content`
    reasoning: str = ""          # separate reasoning field and/or stripped <think>
    reasoning_leak: bool = False # inline <think> was present in content
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    retries: int = 0
    finish_reason: str | None = None
    truncated: bool = False
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def throughput_tps(self) -> float | None:
        if not self.completion_tokens or self.latency_ms <= 0:
            return None
        return round(self.completion_tokens / (self.latency_ms / 1000.0), 2)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class GroqEvalClient:
    """Rate-limit aware async wrapper around Groq chat completions."""

    def __init__(
        self,
        config: EvalConfig,
        api_key: str | None = None,
        ledger: BudgetLedger | None = None,
    ):
        key = api_key or load_api_key()
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Add it to src/core/.env or the environment."
            )
        self.config = config
        self.ledger = ledger
        self._client = AsyncGroq(api_key=key, timeout=config.request_timeout_s)
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def close(self) -> None:
        await self._client.close()

    async def complete(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        system: str | None = None,
        role: str = ROLE_GENERATE,
    ) -> Completion:
        """Stream one completion, retrying transient failures with backoff."""
        budget_error = self._budget_block(model, role)
        if budget_error:
            return Completion(
                model=model, content="", raw_content="", error=budget_error
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": model,
            "messages": messages,
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "top_p": self.config.top_p if top_p is None else top_p,
            "max_completion_tokens": (
                self.config.max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
            "stream": True,
        }

        backoff = self.config.retry_initial_backoff_s
        last_error: str | None = None

        for attempt in range(self.config.max_retries + 1):
            async with self._semaphore:
                try:
                    completion = await self._stream_once(params, retries=attempt)
                    if self.ledger is not None:
                        self.ledger.record(
                            model,
                            role,
                            completion.prompt_tokens,
                            completion.completion_tokens,
                        )
                    return completion
                except RETRYABLE as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if is_permanent_rate_limit(exc):
                        return Completion(
                            model=model,
                            content="",
                            raw_content="",
                            retries=attempt,
                            error=last_error,
                        )
                except APIStatusError as exc:
                    # 5xx is worth retrying; 4xx (bad model, bad request) is not.
                    if exc.status_code < 500:
                        return Completion(
                            model=model,
                            content="",
                            raw_content="",
                            retries=attempt,
                            error=f"{type(exc).__name__} {exc.status_code}: {exc}",
                        )
                    last_error = f"{type(exc).__name__} {exc.status_code}: {exc}"
                except Exception as exc:  # noqa: BLE001 - surfaced on the row
                    return Completion(
                        model=model,
                        content="",
                        raw_content="",
                        retries=attempt,
                        error=f"{type(exc).__name__}: {exc}",
                    )

            if attempt < self.config.max_retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.config.retry_max_backoff_s)

        return Completion(
            model=model,
            content="",
            raw_content="",
            retries=self.config.max_retries,
            error=last_error or "exhausted retries",
        )

    def _budget_block(self, model: str, role: str) -> str | None:
        """Refuse a call the day's remaining allowance cannot cover.

        Checked against this model's own observed average call size rather than
        merely against zero, so the run stops one call early instead of firing
        a request that is certain to be rejected.
        """
        if self.ledger is None:
            return None

        status = self.ledger.status(model)
        remaining = status.remaining
        if remaining is None:
            return None

        expected = self.ledger.observed_call_tokens(model, role)
        needed = int(expected) if expected else 1
        if remaining >= needed:
            return None

        return (
            f"daily token budget exhausted for {model}: {status.used:,} of "
            f"{status.limit:,} used, {remaining:,} left but a {role} call "
            f"needs about {needed:,}. Budget resets on the provider's rolling "
            f"24h window; lower the matrix size or use a different model."
        )

    async def _stream_once(self, params: dict, retries: int) -> Completion:
        started = time.perf_counter()
        ttft: float | None = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = None
        finish_reason: str | None = None

        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            if not chunk.choices:
                # Groq sends a final usage-only chunk with no choices.
                usage = getattr(chunk, "usage", None) or usage
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            piece = getattr(delta, "content", None)
            if piece:
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                content_parts.append(piece)

            # gpt-oss streams chain-of-thought in its own field.
            think = getattr(delta, "reasoning", None)
            if think:
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                reasoning_parts.append(think)

            if choice.finish_reason:
                finish_reason = choice.finish_reason
            chunk_usage = getattr(chunk, "x_groq", None)
            if chunk_usage is not None:
                usage = getattr(chunk_usage, "usage", None) or usage
            usage = getattr(chunk, "usage", None) or usage

        latency_ms = (time.perf_counter() - started) * 1000
        raw_content = "".join(content_parts)
        visible, inline_reasoning, leaked = split_reasoning(raw_content)

        if inline_reasoning:
            reasoning_parts.append(inline_reasoning)

        return Completion(
            model=params["model"],
            content=visible,
            raw_content=raw_content,
            reasoning="".join(reasoning_parts),
            reasoning_leak=leaked,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            latency_ms=round(latency_ms, 2),
            ttft_ms=round(ttft, 2) if ttft is not None else None,
            retries=retries,
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
        )
