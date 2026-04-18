"""
pyboy_agent.llm.retry
=====================
Retry wrapper for OpenAI API calls and JSON extraction utilities.

``with_retry()`` has two execution modes:

Windowed mode (pump_fn provided)
    The VLM call runs in a background thread while ``pump_fn()`` (typically
    ``pyboy.tick(1, render=True)``) is called at ~60 fps on the main thread.
    This keeps the SDL2 window responsive during long VLM waits.

Headless mode (pump_fn is None)
    The VLM call runs directly on the calling thread.  No window to pump.

Token refresh (Copilot backend)
    A 401 Unauthorized response triggers ``on_auth_error()`` (which re-reads
    the OpenClaw token file) and then retries the call immediately without
    consuming a retry slot.  Maximum 2 token refreshes per call.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from typing import Any, Callable

import openai

from pyboy_agent.config import PUMP_INTERVAL_SECONDS


def with_retry(
    fn: Callable[[], Any],
    *,
    retries: int = 6,
    base_delay: float = 10.0,
    pump_fn: Callable[[], None] | None = None,
    on_auth_error: Callable[[], None] | None = None,
) -> Any:
    """Call ``fn()`` (a blocking VLM call), retrying on OpenAI API errors.

    Args:
        fn: Zero-argument callable that makes an OpenAI API call and returns.
        retries: Maximum number of retry attempts after the first failure.
        base_delay: Seconds to wait before retry ``n`` is ``base_delay × n``.
        pump_fn: Called repeatedly (~60 fps) while waiting for the VLM in
                 windowed mode so the SDL2 window stays responsive.
        on_auth_error: Called when a 401 Unauthorized response is received;
                       should refresh the auth token.  Retries immediately.

    Returns:
        Whatever ``fn()`` returns.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    last_exc: Exception | None = None
    auth_refreshes = 0

    for attempt in range(1, retries + 1):
        try:
            if pump_fn is not None:
                # Run VLM in background; pump SDL2 events on main thread.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(fn)
                    while not future.done():
                        pump_fn()
                        time.sleep(PUMP_INTERVAL_SECONDS)
                    return future.result()
            return fn()

        except (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.APITimeoutError,
        ) as exc:
            # 401: token expired — refresh and retry immediately (max 2 times).
            if (
                isinstance(exc, openai.APIStatusError)
                and exc.status_code == 401
                and on_auth_error is not None
                and auth_refreshes < 2
            ):
                auth_refreshes += 1
                print(
                    f"  [llm] 401 Unauthorized — refreshing token "
                    f"(refresh {auth_refreshes}/2)…"
                )
                try:
                    on_auth_error()
                except Exception as refresh_exc:
                    print(f"  [llm] Token refresh failed: {refresh_exc}")
                # Retry immediately; don't consume a retry slot.
                continue

            last_exc = exc
            wait = base_delay * attempt
            print(
                f"  [llm] API error (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {wait:.0f}s…"
            )
            # Keep the window alive during the wait period.
            if pump_fn is not None:
                deadline = time.time() + wait
                while time.time() < deadline:
                    pump_fn()
                    time.sleep(PUMP_INTERVAL_SECONDS)
            else:
                time.sleep(wait)

    raise RuntimeError(f"VLM call failed after {retries} attempts: {last_exc}")


def extract_json(text: str) -> str:
    """Extract the last complete ``{...}`` block from model output.

    Handles common model quirks:
    - Markdown fences (```json ... ```)
    - ``<think>...</think>`` blocks before the JSON
    - Preamble prose before the opening brace

    Scans for the *last* closing brace so stray ``{}`` in preamble prose
    don't interfere.

    Args:
        text: Raw model output string.

    Returns:
        The extracted JSON object string, or ``text.strip()`` if no complete
        ``{...}`` block was found.
    """
    if not text:
        return text

    # Strip markdown fences.
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Strip <think>...</think> blocks (some local models emit these).
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Walk backwards from the last '}', find matching '{'.
    last_close = text.rfind("}")
    if last_close == -1:
        return text.strip()

    depth = 0
    start = -1
    for i in range(last_close, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                break

    if start != -1:
        return text[start : last_close + 1].strip()

    return text.strip()
