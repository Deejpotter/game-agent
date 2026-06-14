"""
mgba_agent/config.py — Backend configuration and global constants.

All tunable constants live here. Import from this module rather than
hard-coding values elsewhere in the package.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

# All three backends expose an OpenAI-compatible /v1/chat/completions endpoint,
# so we use the same openai client for all of them — only the base_url differs.
# Model names and API keys are read from .env so switching backends requires
# no code changes.
BACKENDS: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        # Vision model — handles image → scene description.
        # Gemma 4 E4B: compact multimodal, fast on 8 GB VRAM.
        "model": os.getenv("LMS_MODEL", "google/gemma-4-e4b"),
        # Reasoning model — text-only, handles strategy + button decision.
        # Override with LMS_REASON_MODEL; falls back to same model if not set.
        "reasoning_model": os.getenv("LMS_REASON_MODEL", ""),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        "reasoning_model": os.getenv("OLLAMA_REASON_MODEL", ""),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "reasoning_model": os.getenv("OPENAI_REASON_MODEL", ""),
    },
}

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

# GBA runs at 59.7 fps. 8 frames (~134 ms) is enough for most in-game animations
# (menu transitions, text scroll) to settle before the next screenshot is taken.
SETTLE_FRAMES = 8

# GBA native resolution is 240×160. Doubling to 480×320 gives the VLM enough
# pixel detail to read small text (HP numbers, menu items) without bloating the
# base64 payload. NEAREST-neighbour is used so pixels stay crisp, not blurred.
SCREENSHOT_SCALE = 2

# Every N turns the agent executes the game-specific save_sequence from the
# game profile. 60 turns ≈ 2-5 minutes of play depending on turn speed.
AUTOSAVE_EVERY_N_TURNS = 60

# How often (in turns) the goal-tracker VLM call re-assesses the situation.
GOAL_UPDATE_EVERY_N_TURNS = 10

# After this many consecutive presses of the same button the agent is
# considered stuck. A warning is injected into the next VLM prompt and
# a goal-tracker update is triggered immediately.
STUCK_BUTTON_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Fallback prompt
# ---------------------------------------------------------------------------

# Fallback system prompt used when no game profile is loaded.
GENERIC_SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous game-playing AI controlling a game running in the
    mGBA emulator. Study each screenshot carefully and decide the single best
    button to press next.

    Available buttons: A, B, Up, Down, Left, Right, Start, L, R
    Do NOT use Select — it does nothing in this game.

    Reply with a JSON object ONLY — no markdown, no explanation:
    {
      "button": "<one button string>",
      "reason": "<one sentence explaining why>"
    }

    General strategy:
    - Dismiss all dialogue and menus by pressing A (or B to cancel/back out).
    - Navigate toward whatever the current objective appears to be.
    - If the situation looks the same as the last 3 turns: press B, then try
      a directional button to get unstuck.
""").strip()
