"""
mgba_agent/llm/retry.py — Async retry wrapper for blocking OpenAI API calls.

Runs the blocking call in a thread executor so the asyncio event loop stays
responsive. Retries with exponential backoff on API errors (handles model
reloads in LM Studio / Ollama).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import openai


async def with_retry(fn: Callable[[], Any], *, retries: int = 6, base_delay: float = 10.0) -> Any:
    """Call fn() in a thread executor, retrying on OpenAI API errors.

    Runs the blocking OpenAI call in the default thread pool so the asyncio
    event loop stays responsive. On failure waits base_delay * attempt seconds
    (non-blocking) — handles model swaps / reloads in LM Studio / Ollama.
    """
    loop = asyncio.get_event_loop()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await loop.run_in_executor(None, fn)
        except (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.APITimeoutError,
        ) as exc:
            last_exc = exc
            wait = base_delay * attempt
            print(
                f"  [llm] API error (attempt {attempt}/{retries}): {exc}. "
                f"Model may be loading — retrying in {wait:.0f}s…"
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"VLM call failed after {retries} attempts: {last_exc}")
