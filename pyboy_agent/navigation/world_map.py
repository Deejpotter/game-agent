"""
pyboy_agent.navigation.world_map
=================================
Persistent cross-session location and NPC tracker.

``WorldMap`` stores everything the agent has learned about the game world:
which locations have been visited, which NPCs have been talked to, and which
directions from each map tile are confirmed walls.

Storage
-------
The world map is stored at:
    ``~/.pyboy-agent/world_maps/<game-slug>.json``

This path survives crashes, restarts, and ROM changes as long as the game
slug (the lowercased game name with spaces replaced by hyphens) stays the same.

Wall tracking
-------------
Walls are keyed by *tile* (``map_{bank}_{number}_x{x}_y{y}``), not by VLM
location name.  This prevents wall data from leaking between similarly-named
locations (e.g. two different "Player's House" tiles on different maps).

When all four directions at a tile are recorded as walls (physically impossible
unless a cutscene is freezing movement), ``clear_walls()`` resets that tile's
wall AND tested data.  The loop then presses B×5 to dismiss any frozen dialogue
before retrying.

Fuzzy location matching
-----------------------
Vision models occasionally add floor suffixes (``" (2F)"``), swap synonyms,
or change capitalisation across turns.  ``best_location_key()`` normalises
incoming names against stored keys so wall data accumulates correctly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class WorldMap:
    """Tracks visited locations, NPCs, and per-tile wall data across sessions."""

    def __init__(self, game_slug: str) -> None:
        """Initialise and load the existing world map for ``game_slug``.

        Args:
            game_slug: Kebab-case game identifier, e.g. ``"pokemon-silver"``.
        """
        maps_dir = Path.home() / ".pyboy-agent" / "world_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        self.path = maps_dir / f"{game_slug}.json"
        self._summary_cache: str | None = None

        if self.path.exists():
            try:
                self.data: dict = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {"locations": {}, "visited_order": []}
        else:
            self.data = {"locations": {}, "visited_order": []}

        self.data.setdefault("visited_order", [])

    # ── Location / NPC updates ────────────────────────────────────────────

    def update(
        self,
        location: str,
        *,
        location_status: str | None = None,
        npc: str | None = None,
        npc_status: str | None = None,
        note: str | None = None,
    ) -> None:
        """Record a location visit and optionally an NPC interaction.

        Args:
            location: Location name or tile key.
            location_status: Override status string (e.g. ``"cleared"``).
            npc: NPC identifier to record (or None).
            npc_status: Override NPC status (e.g. ``"talked"``).
            note: Free text note attached to the location or NPC.
        """
        locs = self.data.setdefault("locations", {})
        is_new = location not in locs
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})

        if is_new:
            order = self.data.setdefault("visited_order", [])
            if location not in order:
                order.append(location)

        if location_status:
            entry["status"] = location_status

        if npc:
            npc_entry = entry.setdefault("npcs", {}).setdefault(npc, {"status": "talked"})
            if npc_status:
                npc_entry["status"] = npc_status
            if note:
                npc_entry["note"] = note
        elif note:
            entry["note"] = note

        self._summary_cache = None
        self.save()

    # ── Wall tracking ─────────────────────────────────────────────────────

    def record_wall(self, location: str, direction: str) -> None:
        """Record that ``direction`` is a wall at the given tile/location.

        Only saves when a new wall is discovered (avoids redundant disk writes).
        """
        entry = self.data.setdefault("locations", {}).setdefault(
            location, {"status": "visited", "npcs": {}}
        )
        walls = entry.setdefault("walls", {})
        if not walls.get(direction):
            walls[direction] = True
            self._summary_cache = None
            self.save()

    def clear_walls(self, location: str) -> None:
        """Remove all wall AND tested data for a tile.

        Called when all 4 directions appear blocked (impossible state). Clearing
        ``tested`` as well is critical — otherwise ``get_untested_directions()``
        returns an empty set and the navigation hint gives the model no guidance.
        """
        entry = self.data.get("locations", {}).get(location)
        if entry:
            changed = bool(entry.get("walls") or entry.get("tested"))
            entry["walls"] = {}
            entry["tested"] = {}
            if changed:
                self._summary_cache = None
                self.save()

    def get_walls(self, location: str) -> set[str]:
        """Return the set of confirmed wall directions at this tile."""
        entry = self.data.get("locations", {}).get(location, {})
        return {d for d, v in entry.get("walls", {}).items() if v}

    def record_tested(self, location: str, direction: str) -> None:
        """Record that ``direction`` was attempted (whether wall or not)."""
        entry = self.data.setdefault("locations", {}).setdefault(
            location, {"status": "visited", "npcs": {}}
        )
        tested = entry.setdefault("tested", {})
        if not tested.get(direction):
            tested[direction] = True
            self._summary_cache = None
            self.save()

    def get_untested_directions(self, location: str) -> set[str]:
        """Return cardinal directions not yet attempted from this tile."""
        entry = self.data.get("locations", {}).get(location, {})
        done = (
            set(entry.get("walls", {}).keys())
            | set(entry.get("tested", {}).keys())
        )
        return {"Up", "Down", "Left", "Right"} - done

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of all visited locations and NPCs.

        Result is cached until the next write.
        """
        if self._summary_cache is not None:
            return self._summary_cache

        locs = self.data.get("locations", {})
        if not locs:
            self._summary_cache = "No locations recorded yet."
            return self._summary_cache

        lines: list[str] = []
        order = self.data.get("visited_order", [])
        if order:
            lines.append("Route taken: " + " → ".join(order))
            lines.append("")

        for loc_name, loc in locs.items():
            status = loc.get("status", "visited")
            note = loc.get("note", "")
            line = f"• {loc_name} [{status}]"
            walls = loc.get("walls", {})
            if walls:
                line += f" | walls: {', '.join(sorted(walls))}"
            if note:
                line += f" — {note}"
            lines.append(line)
            for npc_name, npc in loc.get("npcs", {}).items():
                npc_status = npc.get("status", "talked")
                npc_note = npc.get("note", "")
                npc_line = f"    ↳ NPC: {npc_name} [{npc_status}]"
                if npc_note:
                    npc_line += f" — {npc_note}"
                lines.append(npc_line)

        self._summary_cache = "\n".join(lines)
        return self._summary_cache

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self) -> None:
        """Write current world map data to disk (silently ignores I/O errors)."""
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fuzzy location key matching
# ---------------------------------------------------------------------------

# Words that carry no semantic meaning for location matching.
_STOP_WORDS = {"the", "a", "of", "in", "at", "s", "1f", "2f", "b1f", "town", "city", "route"}


def best_location_key(world_map: WorldMap, location: str) -> str:
    """Find the best matching key in world_map for a possibly-drifted VLM location name.

    Vision models can add floor suffixes, change capitalisation, or use synonyms.
    This function tries several normalisation strategies before falling back to
    the raw input.

    Strategies (in order):
    1. Exact match in stored keys.
    2. Strip floor suffix (``" (1F)"`` etc.) and try again.
    3. Case-insensitive exact match.
    4. Same town prefix, ≥2 meaningful building-name words in common.

    Args:
        world_map: WorldMap instance with stored location keys.
        location: Raw location name from the vision model.

    Returns:
        The best matching stored key, or ``location`` if no match found.
    """
    locs = world_map.data.get("locations", {})
    if not locs or not location:
        return location

    # 1. Exact match.
    if location in locs:
        return location

    # 2. Strip floor suffix.
    base = re.sub(r"\s*\([^)]*\)\s*$", "", location).strip()
    if base and base in locs:
        return base

    # 3. Case-insensitive.
    loc_lower = location.lower()
    for k in locs:
        if k.lower() == loc_lower:
            return k

    # 4. Same town + ≥2 shared building words.
    parts = location.split(" - ", 1)
    if len(parts) == 2:
        town, building = parts[0].lower(), parts[1].lower()
        building_words = set(re.findall(r"\b\w+\b", building)) - _STOP_WORDS
        for k in locs:
            k_parts = k.split(" - ", 1)
            if len(k_parts) == 2 and k_parts[0].lower() == town:
                k_words = set(re.findall(r"\b\w+\b", k_parts[1].lower())) - _STOP_WORDS
                if len(building_words & k_words) >= 2:
                    return k

    return location
