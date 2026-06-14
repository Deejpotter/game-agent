"""
nova_agent.pathfinder
=====================
Tile-based navigation using a learned walkability graph.

Design rationale
----------------
pyboy_agent tries to navigate by injecting 12 nav hints and hoping the
LLM avoids walls.  This doesn't work well because:
  - The LLM often ignores hints
  - Hints accumulate from multiple conflicting sources
  - The model still has to figure out which direction to go

nova_agent instead gives the model a ``navigate_to(x, y)`` tool.  The
Python code here computes the actual path over a graph of known tiles
and translates it into a button sequence.  The model just has to name a
destination.

Tile graph
----------
``TileGraph`` records what we have learned about each tile from past turns:
  - passable: directions confirmed passable (player actually moved)
  - blocked: directions confirmed impassable (wall detected)
  - unknown: directions not yet tried

Pathfinding algorithm: BFS over confirmed-passable edges.
If the target tile has never been visited, we fall back to a compass
movement toward the target with obstacle avoidance heuristic.

Map scope
---------
Keys include map_bank + map_number so walls in one room don't bleed
into another.  Format: ``(bank, map_num, x, y)``.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from nova_agent.config import MAX_PATH_STEPS

# Direction → (dx, dy) delta for GBC coordinate system (y increases downward).
_DIR_DELTA: dict[str, tuple[int, int]] = {
    "Up":    (0, -1),
    "Down":  (0,  1),
    "Left":  (-1, 0),
    "Right": (1,  0),
}
_OPPOSITE: dict[str, str] = {"Up": "Down", "Down": "Up", "Left": "Right", "Right": "Left"}

TileKey = tuple[int, int, int, int]  # (bank, map, x, y)


class TileGraph:
    """Persistent graph of tile walkability learned from live gameplay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # tile_key → {direction: "pass" | "block" | "?"}
        self._graph: dict[str, dict[str, str]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def record_move(
        self,
        from_tile: TileKey,
        direction: str,
        *,
        success: bool,
    ) -> None:
        """Record whether moving ``direction`` from ``from_tile`` succeeded.

        Args:
            from_tile: (bank, map, x, y) before the move.
            direction: "Up" | "Down" | "Left" | "Right".
            success: True if the player actually moved (position changed or
                     map warped); False if blocked by a wall.
        """
        key = _tk(from_tile)
        self._graph.setdefault(key, {})
        self._graph[key][direction] = "pass" if success else "block"

        if success:
            # The tile we arrived at is reachable — record the reverse edge as passable.
            dx, dy = _DIR_DELTA[direction]
            to_tile: TileKey = (from_tile[0], from_tile[1], from_tile[2] + dx, from_tile[3] + dy)
            rev_key = _tk(to_tile)
            self._graph.setdefault(rev_key, {})
            # Only record reverse as passable if not already blocked from that side.
            if self._graph[rev_key].get(_OPPOSITE[direction]) != "block":
                self._graph[rev_key][_OPPOSITE[direction]] = "pass"

        self._save()

    def get_status(self, tile: TileKey, direction: str) -> str:
        """Return "pass", "block", or "?" for the given tile + direction."""
        return self._graph.get(_tk(tile), {}).get(direction, "?")

    def passable_directions(self, tile: TileKey) -> list[str]:
        """Directions confirmed passable from this tile."""
        entry = self._graph.get(_tk(tile), {})
        return [d for d, s in entry.items() if s == "pass"]

    def blocked_directions(self, tile: TileKey) -> list[str]:
        """Directions confirmed blocked from this tile."""
        entry = self._graph.get(_tk(tile), {})
        return [d for d, s in entry.items() if s == "block"]

    def untried_directions(self, tile: TileKey) -> list[str]:
        """Directions not yet tried from this tile."""
        entry = self._graph.get(_tk(tile), {})
        return [d for d in _DIR_DELTA if d not in entry]

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def find_path(
        self,
        from_tile: TileKey,
        to_tile: TileKey,
    ) -> list[str] | None:
        """BFS over known passable edges.

        Returns:
            List of direction strings (e.g. ["Right", "Right", "Up"]) or
            None if no path found within MAX_PATH_STEPS.
        """
        if from_tile == to_tile:
            return []

        # BFS.
        queue: deque[tuple[TileKey, list[str]]] = deque()
        queue.append((from_tile, []))
        visited: set[TileKey] = {from_tile}

        while queue:
            current, path = queue.popleft()
            if len(path) >= MAX_PATH_STEPS:
                continue

            for direction, (dx, dy) in _DIR_DELTA.items():
                status = self.get_status(current, direction)
                if status == "block":
                    continue  # Known wall — skip.

                next_tile: TileKey = (
                    current[0], current[1],
                    current[2] + dx, current[3] + dy,
                )
                if next_tile in visited:
                    continue

                new_path = path + [direction]
                if next_tile == to_tile:
                    return new_path

                # Only explore tiles we have confirmed passable edges to.
                if status == "pass":
                    visited.add(next_tile)
                    queue.append((next_tile, new_path))

        return None  # No path found in explored graph.

    def compass_path(self, from_tile: TileKey, to_tile: TileKey) -> list[str]:
        """Greedy compass movement when BFS graph has no path.

        Prefers the axis with the greater distance.  Skips directions known
        to be blocked.  Returns at most MAX_PATH_STEPS directions.
        """
        _, _, fx, fy = from_tile
        _, _, tx, ty = to_tile
        steps: list[str] = []
        cx, cy = fx, fy

        for _ in range(MAX_PATH_STEPS):
            if cx == tx and cy == ty:
                break
            dx = tx - cx
            dy = ty - cy

            # Prefer axis with larger delta.
            candidates: list[str] = []
            if abs(dx) >= abs(dy):
                if dx > 0:
                    candidates = ["Right", "Down" if dy > 0 else "Up"]
                else:
                    candidates = ["Left", "Down" if dy > 0 else "Up"]
            else:
                if dy > 0:
                    candidates = ["Down", "Right" if dx > 0 else "Left"]
                else:
                    candidates = ["Up", "Right" if dx > 0 else "Left"]

            current_tile: TileKey = (from_tile[0], from_tile[1], cx, cy)
            moved = False
            for candidate in candidates:
                if self.get_status(current_tile, candidate) == "block":
                    continue
                steps.append(candidate)
                ddx, ddy = _DIR_DELTA[candidate]
                cx += ddx
                cy += ddy
                moved = True
                break

            if not moved:
                break  # All candidates blocked — give up.

        return steps

    # ------------------------------------------------------------------
    # Helpers for the agent loop
    # ------------------------------------------------------------------

    def tile_summary(self, tile: TileKey) -> str:
        """Short human-readable summary of known directions for a tile."""
        p = self.passable_directions(tile)
        b = self.blocked_directions(tile)
        u = self.untried_directions(tile)
        parts = []
        if p:
            parts.append(f"passable: {', '.join(p)}")
        if b:
            parts.append(f"blocked: {', '.join(b)}")
        if u:
            parts.append(f"untried: {', '.join(u)}")
        return "; ".join(parts) if parts else "no data"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._graph, f, separators=(",", ":"))
        except Exception as exc:
            print(f"[pathfinder] Save failed: {exc}")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                self._graph = json.load(f)
        except Exception as exc:
            print(f"[pathfinder] Load failed ({exc}) — starting fresh graph")
            self._graph = {}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _tk(tile: TileKey) -> str:
    """Serialise a tile key to a dict key string."""
    bank, map_num, x, y = tile
    return f"{bank}:{map_num}:{x}:{y}"
