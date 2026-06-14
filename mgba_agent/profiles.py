"""
mgba_agent/profiles.py — Game profile loader.

Loads games/<name>.json profiles that supply the system prompt, save sequence,
RAM offsets, and initial goal for a specific game. Falls back to generic
defaults when no profile name is given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import GENERIC_SYSTEM_PROMPT


def load_game_profile(name: str | None) -> dict[str, Any]:
    """Load a game profile JSON from games/<name>.json, or return generic defaults."""
    if name is None:
        return {
            "name": "Generic",
            "system_prompt": GENERIC_SYSTEM_PROMPT,
            "save_sequence": None,
        }
    path = Path(__file__).parent.parent / "games" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No game profile found at {path}. "
            f"Create games/{name}.json or omit --game for generic mode."
        )
    profile = json.loads(path.read_text(encoding="utf-8"))
    if "system_prompt" not in profile:
        profile["system_prompt"] = GENERIC_SYSTEM_PROMPT
    return profile
