"""
mgba_agent/vision/perceive.py — Vision model: screenshot → structured scene description.

perceive() sends a base64-encoded screenshot to a multimodal LLM and returns a
JSON string describing the current game screen. This output is fed as plain text
to the reasoning model in llm/decide.py — the reasoning model never sees the raw image.

Helper functions for image processing and MCP result extraction are also here.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from typing import Any

from openai import OpenAI
from PIL import Image

from ..config import SCREENSHOT_SCALE

# ---------------------------------------------------------------------------
# Structured prompt sent to the vision model each turn.
# Output is plain text consumed by the reasoning model — no image needed there.
# ---------------------------------------------------------------------------
_PERCEIVE_PROMPT = """\
Look at this Pokemon GBA screenshot. Describe what you see as JSON.

Classification help:
- If you see FIGHT / BAG / POKEMON / RUN → screen_type is "battle"
- If you see a text box with dialogue → screen_type is "dialogue"
- If you see a menu list (not battle) → screen_type is "menu"
- Otherwise → screen_type is "overworld"

Reply with ONLY this JSON (no markdown fences, no extra text):
{
  "screen_type": "overworld" | "dialogue" | "battle" | "menu",
  "dialogue_text": "<exact text in any dialogue/text box, or null>",
  "menu_options": ["<option1>", "<option2>"] or null,
  "battle_info": "<Pokemon names, levels, HP bars if in battle, or null>",
  "player_facing": "up" | "down" | "left" | "right" | "unknown",
  "adjacent_npc": true | false,
  "surroundings": {
    "up": "<what is directly above the player: open grass, trees, wall, NPC, door, path, water, etc.>",
    "down": "<what is directly below the player>",
    "left": "<what is directly to the left>",
    "right": "<what is directly to the right>"
  },
  "location_name": "<your best guess: 'Route 101', 'Oldale Town', 'Littleroot Town - Prof. Birch Lab', etc.>",
  "notes": "<anything else notable: items on ground, doors, trainers, nameplate banners>"
}
"""


def process_screenshot(b64_data: str, scale: int = SCREENSHOT_SCALE) -> str:
    """Decode base64 PNG, optionally scale it up, and return re-encoded base64."""
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_image(result: Any) -> str | None:
    """Pull the first base64 image from an MCP tool result, or None."""
    if result and result.content:
        for item in result.content:
            if hasattr(item, "data") and item.data:
                return item.data
    return None


def extract_text(result: Any) -> str | None:
    """Pull the first text item from an MCP tool result, or None."""
    if result and result.content:
        for item in result.content:
            if hasattr(item, "text") and item.text:
                return item.text
    return None


def screenshot_hash(b64: str) -> str:
    """MD5 of the raw base64 string — fast way to detect identical frames (wall hit)."""
    return hashlib.md5(b64.encode()).hexdigest()


def perceive(
    vision_client: OpenAI,
    vision_model: str,
    screenshot_b64: str,
) -> str:
    """Ask the vision model to describe the current screen as structured text.

    Returns a JSON string (or a plain-text fallback on parse failure) describing
    what's visible. This is fed as context to the reasoning model in decide().
    """
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
        max_tokens=4096,
        temperature=0.1,
        timeout=120.0,
        extra_body={"enable_thinking": False},
    )
    raw = (response.choices[0].message.content or "").strip()
    # Thinking models may put output in reasoning_content with content empty.
    # Extract the JSON block that contains 'screen_type' from the trace.
    if not raw:
        rc = getattr(response.choices[0].message, "reasoning_content", None) or ""
        if rc:
            print(f"  [perceive] content was empty, extracting JSON from reasoning_content ({len(rc)} chars)")
            # Look specifically for a JSON object containing screen_type — the real output
            _m = list(re.finditer(r'\{[^{}]*"screen_type"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', rc, re.DOTALL))
            raw = _m[-1].group(0).strip() if _m else ""
    if not raw:
        # Log the full finish_reason and token counts to help diagnose
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        print(f"  [perceive] WARNING: empty response. finish_reason={choice.finish_reason!r} usage={usage}")
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Strip <think>...</think> blocks that some reasoning models prefix to their output
    if "<think>" in raw:
        # Remove everything between <think> and </think> (the reasoning scratchpad)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Validate — return raw string either way; reasoning model receives it as text
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        pass  # pass the raw string through; reasoning model can still use it
    return raw
