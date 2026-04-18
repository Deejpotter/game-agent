"""
pyboy_agent.profiles
====================
Game profile loader.

A game profile is a JSON file in the ``games/`` directory that configures:
- The system prompt given to the reasoning model each turn
- The in-game save sequence (button list)
- RAM offset addresses for the specific ROM

See ``.github/instructions/game-profiles.instructions.md`` for the full
authoring guide.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

# Minimal fallback system prompt for games that have no dedicated profile.
# Game-specific profiles in games/<name>.json should always override this.
GENERIC_SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous game-playing AI controlling a Game Boy Color game
    running in the PyBoy emulator. Study each screenshot carefully and decide
    the single best button to press next.

    Available buttons: A, B, Up, Down, Left, Right, Start, Select

    Reply with a JSON object ONLY — no markdown, no explanation:
    {
      "button": "<one button string>",
      "repeat": 1,
      "reason": "<one sentence explaining why>",
      "event": null,
      "goal": null,
      "memory": null,
      "map_update": null
    }

    General strategy:
    - Dismiss all dialogue and menus by pressing A (or B to cancel/back out).
    - Navigate toward whatever the current objective appears to be.
    - If stuck (same screen for 3+ turns): press B then try a different direction.
""").strip()


def load_game_profile(name: str | None) -> dict[str, Any]:
    """Load ``games/<name>.json`` or return a generic profile if name is None.

    Args:
        name: Profile name without extension, e.g. ``"pokemon-silver"``.
              Pass None for a generic, no-profile run.

    Returns:
        Dict containing at minimum ``system_prompt``, ``save_sequence``,
        ``name``, and ``ram_offsets``.

    Raises:
        FileNotFoundError: if the named profile does not exist.
    """
    if name is None:
        return {
            "name": "Generic GBC",
            "console": "gbc",
            "system_prompt": GENERIC_SYSTEM_PROMPT,
            "save_sequence": None,
            "ram_offsets": {},
            "initial_goal": "",
        }

    # Profiles live next to the games/ directory relative to this package's parent.
    games_dir = Path(__file__).parent.parent / "games"
    path = games_dir / f"{name}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No game profile found at {path}.\n"
            f"Create games/{name}.json or omit --game for generic mode."
        )

    # Open with UTF-8 explicitly — some profiles contain the ¥ symbol (money).
    profile: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    # Ensure required keys exist with safe defaults.
    profile.setdefault("system_prompt", GENERIC_SYSTEM_PROMPT)
    profile.setdefault("save_sequence", None)
    profile.setdefault("ram_offsets", {})
    profile.setdefault("initial_goal", "")

    return profile
