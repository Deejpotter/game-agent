"""
mgba_agent/llm/decide.py — Reasoning model: scene description → button decision.

The decide() function sends a text-only prompt (no image) to the reasoning
model and parses its JSON response into an action tuple. The scene description
is produced by vision/perceive.py and passed in as a plain string.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI


def decide(
    reasoning_client: OpenAI,
    reasoning_model: str,
    scene_description: str,
    history: list[dict],
    system_prompt: str,
    *,
    current_goal: str = "",
    stuck_hint: str | None = None,
    memory: str = "",
    story_log: list[str] | None = None,
    goal_log: list[dict] | None = None,
    world_map_summary: str | None = None,
) -> tuple[str, int, str, str | None, str | None, dict | None, str]:
    """Ask the reasoning model what button to press next, given a text scene description.

    Returns (button, repeat, reason, event, new_goal, map_update, new_memory).
    """
    # Build the user text: memory → scene → recent events → world map → goal → nav warnings
    user_parts: list[str] = [
        "CURRENT SCREEN (from vision model):\n" + scene_description,
    ]
    if memory:
        user_parts.insert(0, "YOUR GAME DIARY (your own synthesis — trust this):\n" + memory)
    elif story_log:
        # No memory yet — show raw events as bootstrap context
        user_parts.append(
            "Recent events (bootstrap — summarise these into 'memory' this turn):\n"
            + "\n".join(f"  • {e}" for e in story_log[-10:])
        )
    if goal_log:
        user_parts.append(
            "Goal history — how objectives have changed:\n"
            + "\n".join(f"  turn {g['turn']:04d}: {g['goal']}" for g in goal_log[-5:])
        )
    if world_map_summary:
        user_parts.append(
            "World map — locations and NPCs visited so far:\n" + world_map_summary
        )
    if current_goal:
        user_parts.append(f"Current goal: {current_goal}")
    if stuck_hint:
        user_parts.append(f"⚠ NAVIGATION WARNING: {stuck_hint}")
    user_parts.append("What button should I press next?")
    user_text = "\n\n".join(user_parts)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        # Rolling window of last 6 turns as proper user/assistant pairs.
        # Each pair: compact user summary of what was on screen, then the button chosen.
        # This gives the model a valid alternating message structure so it can
        # detect repeated states (stuck-loop) without excessive token cost.
        *history[-6:],
        {"role": "user", "content": user_text},
    ]

    response = reasoning_client.chat.completions.create(
        model=reasoning_model,
        messages=messages,
        max_tokens=4096,
        temperature=0.2,
        timeout=180.0,
        extra_body={"enable_thinking": False},
    )

    raw = (response.choices[0].message.content or "").strip()
    # If content is empty (thinking model), extract JSON with 'button' key from reasoning_content
    if not raw:
        rc = getattr(response.choices[0].message, "reasoning_content", None) or ""
        if rc:
            _matches = list(re.finditer(r'\{[^{}]*"button"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', rc, re.DOTALL))
            raw = _matches[-1].group(0).strip() if _matches else rc.strip()
    # Some models wrap JSON in ```json fences despite the system prompt telling
    # them not to. Strip fences before parsing.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Strip <think>...</think> reasoning blocks before JSON parsing
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    reason: str = ""
    try:
        parsed = json.loads(raw)
        thinking_raw = parsed.get("thinking")
        if thinking_raw:
            print(f"  [think] {str(thinking_raw).strip()}")
        button = str(parsed.get("button", "A")).strip()
        reason = str(parsed.get("reason", "")).strip()
        # repeat: how many steps to take in this direction (directional buttons only).
        # Clamped to 1-3 so the VLM re-evaluates frequently enough to catch
        # doors, NPCs, and item balls before overshooting them.
        _repeat_raw = parsed.get("repeat", 1)
        try:
            repeat = max(1, min(3, int(_repeat_raw)))
        except (TypeError, ValueError):
            repeat = 1
        # Only allow repeat > 1 for directional buttons — not for A/B/Start etc.
        if button not in {"Up", "Down", "Left", "Right"}:
            repeat = 1
        event_raw = parsed.get("event")
        event: str | None = str(event_raw).strip() if event_raw else None
        goal_raw = parsed.get("goal")
        new_goal: str | None = str(goal_raw).strip() if goal_raw else None
        map_update_raw = parsed.get("map_update")
        map_update: dict | None = map_update_raw if isinstance(map_update_raw, dict) else None
        memory_raw = parsed.get("memory")
        new_memory: str = str(memory_raw).strip() if memory_raw else ""
    except json.JSONDecodeError:
        button, repeat, reason, event, new_goal, map_update, new_memory = (
            "B", 1, f"(parse error — defaulted to B) raw={raw[:80]}", None, None, None, ""
        )

    return button, repeat, reason, event, new_goal, map_update, new_memory
