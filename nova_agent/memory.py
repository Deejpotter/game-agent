"""
nova_agent.memory
=================
Structured knowledge base — the agent's persistent long-term memory.

Design rationale
----------------
pyboy_agent stores memory as a single free-form string that the model
overwrites each turn.  This leads to information loss when the model decides
to write a new value and forgets to include important old facts.

nova_agent uses a sectioned knowledge base (inspired by Anthropic's Claude
Plays Pokemon implementation).  Each section is independently editable so
the model can update "party_status" without touching "game_progress".

Sections (KB_SECTIONS in config.py):
  current_status  — Where the player is right now, immediate context
  game_progress   — Badges earned, key story milestones
  objectives      — Current short-term goal (1-3 sentences)
  party_status    — Team names, levels, HP notes, moves worth knowing
  notes           — Anything else: blocked paths, NPC info, puzzle hints

The KB is persisted to a JSON file next to the ROM after each update.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nova_agent.config import KB_SECTIONS


class KnowledgeBase:
    """Sectioned knowledge base with file-backed persistence."""

    def __init__(self, path: str | Path, initial_goal: str = "") -> None:
        self.path = Path(path)
        self._data: dict[str, str] = {s: "" for s in KB_SECTIONS}
        self._events: list[str] = []   # rolling story log (most recent last)

        self._load()

        # Seed objectives if this is a fresh file and we have an initial goal.
        if initial_goal and not self._data.get("objectives"):
            self._data["objectives"] = initial_goal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_section(self, section: str, content: str) -> str:
        """Overwrite a KB section.  Returns an error string if section unknown."""
        if section not in KB_SECTIONS:
            return (
                f"Unknown section {section!r}. "
                f"Valid sections: {', '.join(KB_SECTIONS)}"
            )
        self._data[section] = content.strip()
        self._save()
        return f"Section '{section}' updated."

    def append_event(self, text: str, *, max_events: int = 40) -> None:
        """Append a story event to the rolling log."""
        self._events.append(text.strip())
        if len(self._events) > max_events:
            self._events = self._events[-max_events:]
        self._save()

    def recent_events(self, n: int = 15) -> list[str]:
        """Return the last *n* story events."""
        return self._events[-n:]

    def get_section(self, section: str) -> str:
        return self._data.get(section, "")

    def to_prompt_block(self) -> str:
        """Format the entire KB as a text block for the LLM prompt."""
        lines = ["=== KNOWLEDGE BASE ==="]
        for section in KB_SECTIONS:
            content = self._data.get(section, "").strip()
            label = section.replace("_", " ").upper()
            lines.append(f"\n[{label}]\n{content if content else '(empty)'}")
        events = self.recent_events()
        if events:
            lines.append("\n[RECENT EVENTS]")
            for e in events:
                lines.append(f"  • {e}")
        lines.append("=== END KB ===")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"sections": dict(self._data), "events": list(self._events)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[memory] Save failed: {exc}")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            for section in KB_SECTIONS:
                self._data[section] = data.get("sections", {}).get(section, "")
            self._events = data.get("events", [])
        except Exception as exc:
            print(f"[memory] Load failed ({exc}) — starting fresh KB")
