"""
nova_agent.config
=================
All tuneable constants, backend definitions, and button mappings.

Pure data module — no imports from the rest of the package.

Key architectural differences from pyboy_agent:
- Single LLM (with vision) instead of separate perceive + decide models.
- The "backend" entry has just one model field (it handles both vision and reasoning).
- Tool-calling instead of rigid JSON schema in system prompt.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Backend definitions — each exposes an OpenAI-compatible endpoint.
# ---------------------------------------------------------------------------

BACKENDS: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        # Single model handles both vision and reasoning.
        "model": os.getenv("LMS_MODEL", "google/gemma-4-e4b"),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    },
    "copilot": {
        "base_url": "https://api.githubcopilot.com",
        "api_key": "_copilot_",  # replaced at startup by _load_copilot_token()
        "model": os.getenv("COPILOT_MODEL", "gpt-4o"),
    },
}

# ---------------------------------------------------------------------------
# GBC button mapping  (GBC has no L/R)
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

# Abbreviations accepted in press_buttons sequences.
BUTTON_ALIASES: dict[str, str] = {
    "U": "Up",
    "D": "Down",
    "L": "Left",
    "R": "Right",
    "S": "Start",
    "X": "Select",
}

# ---------------------------------------------------------------------------
# Emulator timing (frames at ~60 fps)
# ---------------------------------------------------------------------------

SETTLE_FRAMES_MOVE = 16       # One tile walk animation
SETTLE_FRAMES_BUTTON = 8      # Menu / dialogue acknowledgement
SETTLE_FRAMES_CUTSCENE = 30   # Scene transitions
SCREENSHOT_SCALE = 2          # GBC native 160×144 → 320×288

# ---------------------------------------------------------------------------
# Agent loop behaviour
# ---------------------------------------------------------------------------

AUTOSAVE_EVERY_N_TURNS = 60   # Execute save_sequence every N turns
MAX_HISTORY_MESSAGES = 20     # Messages kept before summarization triggers
MAX_SEQUENCE_BUTTONS = 30     # Max buttons in a single press_buttons call
MAX_PATH_STEPS = 60           # Max BFS steps for navigate_to

# ---------------------------------------------------------------------------
# Knowledge base section names (order is display order in prompt)
# ---------------------------------------------------------------------------

KB_SECTIONS: list[str] = [
    "current_status",   # Where am I, what am I doing right now
    "game_progress",    # Badges, story milestones reached
    "objectives",       # Short-term goal (1-3 sentences)
    "party_status",     # Team members, levels, HP notes
    "notes",            # Miscellaneous facts worth remembering
]

# ---------------------------------------------------------------------------
# Overlay drawing constants (PIL)
# ---------------------------------------------------------------------------

OVERLAY_HEIGHT = 36   # pixels of bar added below the screenshot
OVERLAY_FONT_SIZE = 12
OVERLAY_BG_COLOR = (0, 0, 0, 200)    # semi-transparent black
OVERLAY_TEXT_COLOR = (255, 255, 255) # white text
OVERLAY_WARN_COLOR = (255, 80, 80)   # red for low HP
HP_LOW_PCT = 25                       # below this → red text
