"""
pyboy_agent.config
==================
All tuneable constants, backend definitions, and button mappings.

Nothing here should import from the rest of the package — it is a pure
data module that every other module can safely import without circular deps.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Backend definitions
# ---------------------------------------------------------------------------
# Each backend exposes an OpenAI-compatible /v1/chat/completions endpoint.
# Model names are read from .env so switching backends needs no code changes.

BACKENDS: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        # Vision model: handles screenshot → scene description (must support images).
        "model": os.getenv("LMS_MODEL", "google/gemma-4-e2b"),
        # Reasoning model: text-only, handles strategy + button decision.
        # Falls back to the same model as vision if not set.
        "reasoning_model": os.getenv("LMS_REASON_MODEL", ""),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        "reasoning_model": os.getenv("OLLAMA_REASON_MODEL", ""),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "reasoning_model": os.getenv("OPENAI_REASON_MODEL", ""),
    },
    "copilot": {
        "base_url": "https://api.githubcopilot.com",
        # Sentinel value replaced at startup by _load_copilot_token().
        "api_key": "_copilot_",
        "model": os.getenv("COPILOT_MODEL", "gpt-4o"),
        "reasoning_model": os.getenv("COPILOT_REASON_MODEL", ""),
    },
}

# ---------------------------------------------------------------------------
# GBC button mapping: VLM-style names → PyBoy API names (lowercase)
# GBC does NOT have L/R buttons — game profiles must not include them.
# ---------------------------------------------------------------------------

BUTTON_MAP: dict[str, str] = {
    "A": "a",
    "B": "b",
    "Start": "start",
    "Select": "select",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
}

# ---------------------------------------------------------------------------
# Emulator timing constants
# ---------------------------------------------------------------------------
# GBC runs at ~60 fps. These frame counts are the minimum settle time after
# each button type to ensure the game has processed the input before the
# next screenshot is taken.

SETTLE_FRAMES_MOVE = 16     # One tile walk animation (~267 ms at 60 fps)
SETTLE_FRAMES_BUTTON = 8    # Menu/dialogue acknowledgement (~133 ms)
SETTLE_FRAMES_CUTSCENE = 30 # Scene transitions, map loads (~500 ms)

# GBC native resolution: 160×144. Upscaling gives the VLM enough pixel detail
# to read HP numbers and small menu text without inflating token count too much.
SCREENSHOT_SCALE = 2

# ---------------------------------------------------------------------------
# Agent behaviour constants
# ---------------------------------------------------------------------------

# Execute the game's save_sequence every N turns.
# Not too frequent (disturbs gameplay) but often enough to survive a crash.
AUTOSAVE_EVERY_N_TURNS = 60

# After this many identical consecutive button presses, inject a stuck warning.
STUCK_BUTTON_THRESHOLD = 5

# After this many turns at the same tile, start injecting B-press recovery.
STUCK_TILE_THRESHOLD = 8

# How many consecutive A presses before warning the model to stop spamming A.
CONSECUTIVE_A_THRESHOLD = 3

# Maximum number of conversation history messages kept in the LLM context.
# Older turns are dropped to avoid ballooning token cost.
MAX_HISTORY_MESSAGES = 10

# Window event pump interval when windowed (SDL2 must receive events ~60×/s).
PUMP_INTERVAL_SECONDS = 0.016
