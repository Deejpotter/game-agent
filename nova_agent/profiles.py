"""
nova_agent.profiles
===================
Load and validate game profiles from ``games/*.json``.

Profiles are shared with pyboy_agent and mgba_agent — the same JSON files
work for all three agents.  nova_agent requires only a subset of fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GAMES_DIR = Path(__file__).parent.parent / "games"

_REQUIRED_FIELDS = ("name", "system_prompt")


def load_game_profile(name: str) -> dict[str, Any]:
    """Load a game profile by slug or file path.

    Args:
        name: Either a profile slug (e.g. ``"pokemon-silver"``) that matches
              a file in ``games/<slug>.json``, or an absolute/relative path to
              a JSON file.

    Returns:
        Parsed profile dict.

    Raises:
        FileNotFoundError: Profile file not found.
        ValueError: Required field missing from profile.
    """
    path = Path(name)
    if not path.exists():
        path = _GAMES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Game profile not found: {name!r}")

    with open(path, encoding="utf-8") as f:
        profile: dict[str, Any] = json.load(f)

    for field in _REQUIRED_FIELDS:
        if field not in profile:
            raise ValueError(f"Profile {path} is missing required field: {field!r}")

    # Ensure ram_offsets exists (may be empty for games without RAM support).
    profile.setdefault("ram_offsets", {})
    profile.setdefault("initial_goal", "")
    profile.setdefault("save_sequence", [])
    profile.setdefault("console", "gbc")

    return profile


def list_profiles() -> list[str]:
    """Return all available profile slugs."""
    return [p.stem for p in sorted(_GAMES_DIR.glob("*.json"))]
