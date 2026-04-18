"""
pyboy_agent.llm.decide
======================
Reasoning model integration — given a scene description and context,
decide the next button press and update agent state.

The ``decide()`` function assembles a rich prompt from multiple context sources
(RAM state, dialogue, memory, goal log, world map, navigation hints) and sends
it to the reasoning model.  The model's JSON response is parsed into a typed
tuple of action + state updates.

Return value
------------
``decide()`` returns an 8-tuple:

    (button, repeat, reason, event, new_goal, map_update, new_memory, thinking)

- ``button``     : one of A/B/Up/Down/Left/Right/Start/Select
- ``repeat``     : 1-3 (only for directional buttons; clamped to 1 for all others)
- ``reason``     : one-sentence explanation from the model
- ``event``      : notable event to append to story_log (or None)
- ``new_goal``   : updated goal string (or None if unchanged)
- ``map_update`` : dict with location/npc update fields (or None)
- ``new_memory`` : updated memory string (or "" if unchanged)
- ``thinking``   : API-level reasoning_content string (or ""), for debug logging
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from openai import OpenAI

from pyboy_agent.llm.retry import extract_json
from pyboy_agent.config import MAX_HISTORY_MESSAGES


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
    dialogue_text: str | None = None,
    extra_body: dict | None = None,
    ram_state_text: str | None = None,
) -> tuple[str, int, str, str | None, str | None, dict | None, str, str]:
    """Ask the reasoning model what button to press next.

    Assembles a structured prompt from all available context, calls the model,
    parses the JSON response, and returns an 8-tuple of action + state updates.

    On JSON parse failure, defaults to button ``"B"`` and returns an empty
    string for new_memory and thinking so the loop can continue safely.
    """
    # ── Build user prompt ─────────────────────────────────────────────────
    user_parts: list[str] = []

    # RAM state goes first — it is authoritative ground truth.
    if ram_state_text:
        user_parts.append(ram_state_text)

    user_parts.append("CURRENT SCREEN (from vision model):\n" + scene_description)

    # Dialogue gets its own prominent block.
    if dialogue_text:
        user_parts.append(
            f'NPC DIALOGUE ON SCREEN RIGHT NOW: "{dialogue_text}"\n'
            "READ this carefully. Your thinking MUST explain what it means for your "
            "story progress and what you should do next. Copy the key sentence to "
            '"event".'
        )

    # Memory (agent's own synthesised diary) takes priority over raw story_log.
    if memory:
        user_parts.append("YOUR GAME DIARY (your own synthesis — trust this):\n" + memory)
    elif story_log:
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

    # ── API call ──────────────────────────────────────────────────────────
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *history[-MAX_HISTORY_MESSAGES:],
        {"role": "user", "content": user_text},
    ]

    response = reasoning_client.chat.completions.create(
        model=reasoning_model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=4096,
        temperature=0.2,
        timeout=120.0,
        **(({"extra_body": extra_body}) if extra_body else {}),
    )

    raw = (response.choices[0].message.content or "").strip()

    # Surface API-level thinking (reasoning_content) when present.
    thinking_str = ""
    reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
    if reasoning_content:
        thinking_str = str(reasoning_content).strip()
        if thinking_str:
            _lines = textwrap.wrap(thinking_str, width=120)
            print(f"  [thinking] ({len(thinking_str)} chars)")
            for line in _lines[:20]:
                print(f"    {line}")
            if len(_lines) > 20:
                print(f"    … ({len(_lines) - 20} more lines)")

    # Fallback: if content is empty, use reasoning_content as the raw answer.
    if not raw and thinking_str:
        raw = thinking_str

    raw = extract_json(raw)

    # ── Parse response ────────────────────────────────────────────────────
    try:
        parsed = json.loads(raw)

        # Print JSON-embedded thinking only when API-level thinking is absent.
        json_thinking = parsed.get("thinking")
        if json_thinking and not reasoning_content:
            t = str(json_thinking).strip()
            lines = textwrap.wrap(t, width=120)
            print(f"  [thinking/json] ({len(t)} chars)")
            for line in lines[:20]:
                print(f"    {line}")

        button = str(parsed.get("button", "A")).strip()
        reason = str(parsed.get("reason", ""))

        # repeat is only meaningful for directional buttons; cap at 3.
        try:
            repeat = max(1, min(3, int(parsed.get("repeat", 1))))
        except (TypeError, ValueError):
            repeat = 1
        if button not in {"Up", "Down", "Left", "Right"}:
            repeat = 1

        event_raw = parsed.get("event")
        event: str | None = str(event_raw).strip() if event_raw else None

        goal_raw = parsed.get("goal")
        new_goal: str | None = str(goal_raw).strip() if goal_raw else None

        map_update_raw = parsed.get("map_update")
        map_update: dict | None = (
            map_update_raw if isinstance(map_update_raw, dict) else None
        )

        memory_raw = parsed.get("memory")
        new_memory: str = str(memory_raw).strip() if memory_raw else ""

    except json.JSONDecodeError:
        button    = "B"
        repeat    = 1
        reason    = f"(parse error — defaulted to B) raw={raw[:80]}"
        event     = None
        new_goal  = None
        map_update= None
        new_memory = ""

    return button, repeat, reason, event, new_goal, map_update, new_memory, thinking_str
