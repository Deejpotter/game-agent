"""
pyboy_agent.goals.tracker
==========================
``NotesTracker`` — persistent agent memory across sessions.

The tracker owns:
  - ``story_log``   : list of notable events (appended each turn)
  - ``goal_log``    : list of ``{turn, goal}`` dicts (appended when goal changes)
  - ``current_goal``: current short-term objective string
  - ``memory``      : agent's own synthesised game diary (updated by the model)

All state is persisted to a JSON file next to the ROM after every update so
crashes don't lose progress.  The file is loaded on startup to resume a
previous session.

File location
-------------
    ``<rom_path>.pyboy_agent_notes.json``

For example:
    ``H:/Games/GBC/Pokemon Silver.gbc.pyboy_agent_notes.json``
"""

from __future__ import annotations

import json
from pathlib import Path


class NotesTracker:
    """Persists and manages the agent's story log, goals, and memory."""

    def __init__(self, notes_path: Path, initial_goal: str = "") -> None:
        """Load existing notes or create an empty tracker.

        Args:
            notes_path: Path to the notes JSON file.
            initial_goal: Goal string from the game profile (used when no
                          saved goal exists).
        """
        self.notes_path = notes_path
        self.story_log: list[str] = []
        self.goal_log: list[dict] = []
        self.current_goal: str = initial_goal
        self.memory: str = ""

        if notes_path.exists():
            try:
                saved = json.loads(notes_path.read_text(encoding="utf-8"))
                self.story_log   = saved.get("story_log", [])
                self.goal_log    = saved.get("goal_log", [])
                self.current_goal = saved.get("current_goal") or initial_goal
                self.memory      = saved.get("memory", "")
                print(
                    f"[notes] Restored {len(self.story_log)} story events, "
                    f"{len(self.goal_log)} goal changes, "
                    f"memory={'yes' if self.memory else 'none'}."
                )
            except Exception as exc:
                print(f"[notes] WARNING: could not load {notes_path}: {exc}")

    # ── Updates ───────────────────────────────────────────────────────────

    def append_event(self, event: str) -> None:
        """Append a notable event to the story log and persist."""
        if event:
            self.story_log.append(event)
            print(f"  [story] {event}")
            self.flush()

    def update_goal(self, new_goal: str, turn: int) -> None:
        """Update the current goal if it changed and persist.

        Args:
            new_goal: New goal string from the reasoning model.
            turn: Current turn number (for the goal log timestamp).
        """
        if new_goal and new_goal != self.current_goal:
            self.goal_log.append({"turn": turn, "goal": new_goal})
            self.current_goal = new_goal
            print(f"  [goal]  {self.current_goal}")
            self.flush()

    def update_memory(self, new_memory: str) -> None:
        """Replace the agent memory string and persist."""
        if new_memory and new_memory != self.memory:
            self.memory = new_memory
            print(f"  [memory] {self.memory[:200]}")
            self.flush()

    # ── Persistence ───────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write current state to the notes file (silently ignores I/O errors)."""
        try:
            self.notes_path.write_text(
                json.dumps(
                    {
                        "story_log": self.story_log,
                        "current_goal": self.current_goal,
                        "goal_log": self.goal_log,
                        "memory": self.memory,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def recent_story(self) -> list[str]:
        """Return the last 15 story_log entries."""
        return self.story_log[-15:]

    @property
    def recent_goals(self) -> list[dict]:
        """Return the last 5 goal_log entries."""
        return self.goal_log[-5:]
