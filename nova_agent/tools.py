"""
nova_agent.tools
================
OpenAI function-calling tool definitions and dispatcher.

Tool design philosophy
----------------------
pyboy_agent encodes all of the model's decision in a rigid JSON schema
(button, repeat, reason, event, new_goal, …) and parses the raw text.
If the model drifts from the schema, the turn fails.

nova_agent uses the OpenAI ``tools`` API instead.  The model calls
functions explicitly; we execute them and return structured results.
This is more robust because:
  - The API enforces argument types
  - The model can call multiple tools per turn (e.g. press_buttons THEN update_knowledge)
  - Each tool returns clear feedback the model can read in the next message
  - Unknown fields never cause parse failures

Available tools
---------------
  press_buttons    — Execute a sequence of GBC button presses
  navigate_to      — Pathfind and walk to a tile coordinate (x, y)
  update_knowledge — Overwrite a knowledge-base section
  add_event        — Append a story event to the rolling log
  set_goal         — Update the current objectives KB section
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nova_agent.config import BUTTON_MAP, BUTTON_ALIASES, MAX_SEQUENCE_BUTTONS
from nova_agent.pathfinder import TileGraph, TileKey

if TYPE_CHECKING:
    from pyboy import PyBoy
    from nova_agent.memory import KnowledgeBase


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "press_buttons",
            "description": (
                "Execute a sequence of Game Boy button presses in order. "
                "Use this for navigation, menu selections, and battle moves. "
                "You can chain multiple buttons: e.g. 'Right Right Down A'. "
                f"Maximum {MAX_SEQUENCE_BUTTONS} buttons per call. "
                "Directional buttons: Up, Down, Left, Right. "
                "Action buttons: A, B, Start, Select."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": (
                            "Space-separated list of button names to press in order. "
                            "Example: 'Right Right Down A'. "
                            "Also accepts abbreviations: U D L R A B S X. "
                            "If you have nothing to press, call add_event or update_knowledge instead — "
                            "do NOT pass 'None', 'none', or an empty string as the sequence."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "One-sentence explanation of why you are pressing these buttons.",
                    },
                },
                "required": ["sequence", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": (
                "Walk to the given tile coordinates on the current map. "
                "The pathfinder will compute the route using learned walkability data "
                "and execute the button presses automatically. "
                "Use this instead of manually specifying direction buttons when you "
                "know where you want to go."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Target X tile coordinate (from RAM overlay on screenshot).",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Target Y tile coordinate (from RAM overlay on screenshot).",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you want to go to this location.",
                    },
                },
                "required": ["x", "y", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_knowledge",
            "description": (
                "Update a section of your persistent knowledge base. "
                "Use this to store important facts you want to remember across turns. "
                "Sections: current_status, game_progress, objectives, party_status, notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["current_status", "game_progress", "objectives", "party_status", "notes"],
                        "description": "Which section of the knowledge base to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for this section (replaces the current value).",
                    },
                },
                "required": ["section", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": (
                "Append a notable story event to the rolling event log. "
                "Use this for milestones: winning a battle, getting a badge, "
                "receiving an item, learning important NPC dialogue, etc. "
                "Keep the event text short (one sentence)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event": {
                        "type": "string",
                        "description": "Short description of the notable event.",
                    },
                },
                "required": ["event"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_goal",
            "description": (
                "Update your current short-term goal. "
                "Use this when your immediate objective changes "
                "(e.g. after getting a badge, after entering a new area)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "New short-term goal (1-3 sentences).",
                    },
                },
                "required": ["goal"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Executes tool calls returned by the LLM and returns result strings."""

    def __init__(
        self,
        pyboy: "PyBoy",
        kb: "KnowledgeBase",
        tile_graph: TileGraph,
        ram_state: dict[str, Any],
        pump_fn=None,
    ) -> None:
        self._pyboy = pyboy
        self._kb = kb
        self._graph = tile_graph
        self._ram = ram_state
        self._pump = pump_fn

        # Buttons actually pressed this turn (for wall detection).
        self.pressed_this_turn: list[str] = []
        # Pre-move RAM state for the first directional action (for wall detection).
        self.pre_move_ram: dict[str, Any] | None = None
        # True if we used navigate_to (pathfinder handled wall recording).
        self.used_pathfinder: bool = False

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool call and return the result string."""
        if tool_name == "press_buttons":
            return self._press_buttons(args)
        if tool_name == "navigate_to":
            return self._navigate_to(args)
        if tool_name == "update_knowledge":
            return self._update_knowledge(args)
        if tool_name == "add_event":
            return self._add_event(args)
        if tool_name == "set_goal":
            return self._set_goal(args)
        return f"Unknown tool: {tool_name!r}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _press_buttons(self, args: dict[str, Any]) -> str:
        from nova_agent.emulator import press_sequence

        raw_seq = str(args.get("sequence", ""))
        buttons = _parse_sequence(raw_seq)
        if not buttons:
            return "No valid buttons in sequence."
        if len(buttons) > MAX_SEQUENCE_BUTTONS:
            buttons = buttons[:MAX_SEQUENCE_BUTTONS]

        pressed = press_sequence(self._pyboy, buttons, pump_fn=self._pump)
        self.pressed_this_turn.extend(pressed)
        return f"Pressed: {' '.join(pressed)}"

    def _navigate_to(self, args: dict[str, Any]) -> str:
        from nova_agent.emulator import press_sequence
        from nova_agent.state import read_ram

        tx = int(args.get("x", 0))
        ty = int(args.get("y", 0))

        # Current tile from RAM.
        mb  = self._ram.get("map_bank")
        mn  = self._ram.get("map_number")
        cx  = self._ram.get("x_pos")
        cy  = self._ram.get("y_pos")

        if None in (mb, mn, cx, cy):
            return "Cannot navigate: player position not available from RAM."

        from_tile: TileKey = (mb, mn, cx, cy)
        to_tile:   TileKey = (mb, mn, tx, ty)

        path = self._graph.find_path(from_tile, to_tile)
        if path is None:
            # Fall back to compass movement.
            path = self._graph.compass_path(from_tile, to_tile)
            method = "compass"
        else:
            method = "BFS"

        if not path:
            return f"Already at ({tx},{ty}) or no path found."

        # Execute path step-by-step, recording wall detection for each move.
        # We read RAM after each directional press to detect walls on the fly
        # and update the tile graph.
        current_tile = from_tile
        executed: list[str] = []

        # We need ram_offsets to re-read RAM mid-path; stored in self._ram.
        ram_offsets = self._ram.get("_offsets", {})

        for direction in path:
            from nova_agent.emulator import press_sequence as ps
            from nova_agent.state import read_ram as rr

            pre_x = current_tile[2]
            pre_y = current_tile[3]

            ps(self._pyboy, [direction], pump_fn=self._pump)
            executed.append(direction)
            self.pressed_this_turn.append(direction)

            if ram_offsets:
                post_ram = rr(self._pyboy, ram_offsets)
                post_x = post_ram.get("x_pos")
                post_y = post_ram.get("y_pos")
                post_mb = post_ram.get("map_bank")
                post_mn = post_ram.get("map_number")
            else:
                post_x = post_y = post_mb = post_mn = None

            # Map warp → definitely moved.
            warped = (post_mb != current_tile[0] or post_mn != current_tile[1])
            moved = warped or (post_x != pre_x or post_y != pre_y)

            self._graph.record_move(current_tile, direction, success=moved)

            if not moved:
                # Wall hit — stop navigation.
                return (
                    f"Navigated {len(executed)-1} step(s) via {method}, "
                    f"then hit a wall going {direction}. "
                    f"Stopped at tile ({current_tile[2]},{current_tile[3]})."
                )

            if not warped and post_x is not None:
                current_tile = (post_mb or current_tile[0], post_mn or current_tile[1], post_x, post_y)

        self.used_pathfinder = True
        return (
            f"Navigated to ({current_tile[2]},{current_tile[3]}) via {method} "
            f"({len(executed)} step(s))."
        )

    def _update_knowledge(self, args: dict[str, Any]) -> str:
        section = str(args.get("section", ""))
        content = str(args.get("content", ""))
        return self._kb.update_section(section, content)

    def _add_event(self, args: dict[str, Any]) -> str:
        event = str(args.get("event", "")).strip()
        if not event:
            return "Event text is empty."
        self._kb.append_event(event)
        return f"Event recorded: {event}"

    def _set_goal(self, args: dict[str, Any]) -> str:
        goal = str(args.get("goal", "")).strip()
        if not goal:
            return "Goal text is empty."
        result = self._kb.update_section("objectives", goal)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sequence(raw: str) -> list[str]:
    """Parse a space-separated button sequence string into a list of button names."""
    tokens = raw.strip().split()
    buttons: list[str] = []
    valid_names = set(BUTTON_MAP.keys())

    for token in tokens:
        title = token.title()
        if title in valid_names:
            buttons.append(title)
            continue
        upper = token.upper()
        if upper in BUTTON_ALIASES:
            buttons.append(BUTTON_ALIASES[upper])
            continue
        # Tolerate lowercase exact matches.
        for name, pyboy_key in BUTTON_MAP.items():
            if token.lower() == pyboy_key:
                buttons.append(name)
                break
        else:
            print(f"[tools] Unknown button token: {token!r} — skipped")

    return buttons
