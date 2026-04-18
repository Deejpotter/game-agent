"""
pyboy_agent.vision.perceive
===========================
Two-stage perception: the vision model reads the raw screenshot and outputs
a compact JSON description of the screen.

This keeps raw images out of the reasoning model's context entirely — the
reasoning model only ever sees the structured JSON description that perceive()
returns, not the pixel data.  This is important for cost and for models that
lack vision support.

Prompt design notes
-------------------
- The prompt asks for a fixed schema with no free text fields except
  ``dialogue_text``, ``location_name``, and ``battle_info``.
- ``passable_directions`` is NOT requested here — it was unreliable in outdoor
  areas.  The loop strips it even if a model emits it, and relies on RAM
  position delta for wall detection instead.
- ``is_outdoor`` helps the reasoning model correct stale memory that thinks the
  player is indoors when the camera shows overworld tiles.
"""

from __future__ import annotations

import json

from openai import OpenAI


# ---------------------------------------------------------------------------
# Perception prompt
# ---------------------------------------------------------------------------

_PERCEIVE_PROMPT = """\
Read this Pokemon Silver screenshot and output compact JSON only.
No strategy. No button advice. No markdown.

Return exactly this schema (no additional keys):
{
  "screen_type": "overworld" | "dialogue" | "battle" | "menu" | "cutscene" | "unknown",
  "dialogue_text": "<exact text shown in the dialogue box, or null>",
  "location_name": "<best guess of location — city, route, or building name — or null>",
  "adjacent_npc": true | false,
  "adjacent_npc_id": "<NPC identifier string or null>",
  "yes_no_cursor": "YES" | "NO" | null,
  "menu_options": ["<option1>", "..."] | null,
  "battle_info": "<summary of battle menu or message visible, or null>",
  "battle_moves": ["MOVE1", "MOVE2", "MOVE3", "MOVE4"] | null,
  "player_facing": "Up" | "Down" | "Left" | "Right" | "Unknown",
  "is_outdoor": true | false | null
}

Rules:
- dialogue_text: copy the EXACT text from the box (preserve punctuation).
- location_name: use "Pallet Town" not "a town"; "Route 29" not "a route".
- adjacent_npc: true only if an NPC is directly in front of the player sprite.
- yes_no_cursor: which option the cursor is on in a Yes/No box (or null).
- battle_moves: list of move names shown in FIGHT menu (null if not visible).
- is_outdoor: true for overworld/routes, false for buildings/caves, null if uncertain.
"""


# ---------------------------------------------------------------------------
# perceive()
# ---------------------------------------------------------------------------

def perceive(
    vision_client: OpenAI,
    vision_model: str,
    screenshot_b64: str,
    *,
    extra_body: dict | None = None,
) -> str:
    """Ask the vision model to describe the current screen as structured JSON.

    The returned string is a JSON object (or best-effort partial JSON if the
    model output could not be parsed).  The caller is responsible for parsing
    and handling parse failures.

    Args:
        vision_client: OpenAI-compatible client for the vision model.
        vision_model: Model name string (e.g. ``"gpt-4o"``).
        screenshot_b64: Base64-encoded PNG of the current screen.
        extra_body: Optional extra fields passed through to the API request
                    (e.g. ``{"enable_thinking": True}`` for local backends).

    Returns:
        JSON string describing the screen, or ``"{}"`` on empty response.
    """
    from pyboy_agent.llm.retry import extract_json  # local import avoids circular

    response = vision_client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {"type": "text", "text": _PERCEIVE_PROMPT},
                ],
            }
        ],
        max_tokens=2048,
        temperature=0.1,
        timeout=60.0,
        **(({"extra_body": extra_body}) if extra_body else {}),
    )

    raw = (response.choices[0].message.content or "").strip()

    # Some models put the answer in reasoning_content when thinking is enabled
    if not raw:
        rc = getattr(response.choices[0].message, "reasoning_content", None)
        if rc:
            raw = rc.strip()

    if not raw:
        choice = response.choices[0]
        usage  = getattr(response, "usage", None)
        print(
            f"  [perceive] WARNING: empty response. "
            f"finish_reason={choice.finish_reason!r} usage={usage}"
        )

    raw = extract_json(raw)

    # Validate — log but don't raise so the loop can continue.
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [perceive/parse-fail] raw ({len(raw)} chars): {raw[:300]}")

    return raw or "{}"
