"""
mgba_agent/navigation/world_map.py — Persistent cross-session location and NPC tracker.

WorldMap stores visited buildings, NPCs, confirmed wall directions, and tested
directions for a single game. Data is persisted to
~/.mgba-live-mcp/world_maps/<game-slug>.json so knowledge survives crashes,
restarts, and --session resumes indefinitely.

best_location_key() fuzzy-matches a VLM-supplied location name against the map
to handle model name drift across sessions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class WorldMap:
    """Tracks visited buildings and NPCs across all sessions for a given game.

    Stored at ~/.mgba-live-mcp/world_maps/<game-slug>.json so knowledge
    survives crashes, restarts, and --session resumes indefinitely.

    Location statuses: "visited" (entered but not fully explored),
                       "fully_explored" (all rooms, NPCs, items checked).
    NPC statuses:      "talked", "quest_active", "quest_complete".
    """

    # Regex that matches real Pokemon location names. Prevents VLM hallucinations
    # like "Unknown grassy area" from persisting in the world map across sessions.
    _REAL_LOCATION_RE = re.compile(
        r'^(Route \d|Littleroot|Oldale|Petalburg|Rustboro|Dewford|Slateport|Mauville|'
        r'Verdanturf|Fallarbor|Lavaridge|Fortree|Lilycove|Mossdeep|Sootopolis|'
        r'Pacifidlog|Ever Grande|Pokemon|Pokémon|Poké|Prof\.|map_\d)',
        re.IGNORECASE,
    )

    def __init__(self, game_slug: str) -> None:
        maps_dir = Path.home() / ".mgba-live-mcp" / "world_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        self.path = maps_dir / f"{game_slug}.json"
        self._summary_cache: str | None = None  # invalidated on every update()
        if self.path.exists():
            try:
                raw_data: dict = json.loads(self.path.read_text(encoding="utf-8"))
                # Strip hallucinated location names on load — keeps disk file clean
                clean_locs = {
                    k: v for k, v in raw_data.get("locations", {}).items()
                    if self._REAL_LOCATION_RE.search(k)
                }
                clean_order = [
                    x for x in raw_data.get("visited_order", [])
                    if x in clean_locs
                ]
                removed = len(raw_data.get("locations", {})) - len(clean_locs)
                if removed:
                    print(f"[world_map] Pruned {removed} non-canonical location(s) on load.")
                self.data: dict = {"locations": clean_locs, "visited_order": clean_order}
            except Exception:
                self.data = {"locations": {}, "visited_order": []}
        else:
            self.data = {"locations": {}, "visited_order": []}
        # Ensure visited_order exists in older saved files
        self.data.setdefault("visited_order", [])

    def update(
        self,
        location: str,
        *,
        location_status: str | None = None,
        npc: str | None = None,
        npc_status: str | None = None,
        note: str | None = None,
    ) -> None:
        # Reject hallucinated location names silently — only store real places.
        if not self._REAL_LOCATION_RE.search(location):
            return
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
            npcs = entry.setdefault("npcs", {})
            npc_entry = npcs.setdefault(npc, {"status": "talked"})
            if npc_status:
                npc_entry["status"] = npc_status
            if note:
                npc_entry["note"] = note
        elif note:
            entry["note"] = note
        self._summary_cache = None  # invalidate cache
        self.save()

    def record_wall(self, location: str, direction: str) -> None:
        """Record that walking in `direction` hits a wall at `location`.

        Only writes to disk when a direction is newly discovered, so repeated
        wall hits in the same direction cost nothing after the first.
        """
        locs = self.data.setdefault("locations", {})
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})
        walls = entry.setdefault("walls", {})
        if not walls.get(direction):
            walls[direction] = True
            self._summary_cache = None
            self.save()

    def get_walls(self, location: str) -> set[str]:
        """Return the set of confirmed wall directions for `location`."""
        entry = self.data.get("locations", {}).get(location, {})
        return {d for d, v in entry.get("walls", {}).items() if v}

    def record_tested(self, location: str, direction: str) -> None:
        """Record that `direction` was attempted at `location` (wall or open move).

        Combined with record_wall(), this lets get_untested_directions() return
        only directions the agent has never tried yet — powering the boundary scan.
        Only writes to disk when newly discovered.
        """
        locs = self.data.setdefault("locations", {})
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})
        tested = entry.setdefault("tested", {})
        if not tested.get(direction):
            tested[direction] = True
            self._summary_cache = None
            self.save()

    def get_untested_directions(self, location: str) -> set[str]:
        """Return cardinal directions not yet attempted at `location`.

        A direction is 'done' once it appears in either `walls` or `tested`.
        The boundary scan is complete when this returns an empty set.
        """
        entry = self.data.get("locations", {}).get(location, {})
        done = set(entry.get("walls", {}).keys()) | set(entry.get("tested", {}).keys())
        return {"Up", "Down", "Left", "Right"} - done

    def summary(self) -> str:
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
            line = f"\u2022 {loc_name} [{status}]"
            walls = loc.get("walls", {})
            if walls:
                line += f" | walls: {', '.join(sorted(walls))}"
            if note:
                line += f" \u2014 {note}"
            lines.append(line)
            for npc_name, npc in loc.get("npcs", {}).items():
                npc_status = npc.get("status", "talked")
                npc_note = npc.get("note", "")
                npc_line = f"    \u21b3 NPC: {npc_name} [{npc_status}]"
                if npc_note:
                    npc_line += f" \u2014 {npc_note}"
                lines.append(npc_line)
        self._summary_cache = "\n".join(lines)
        return self._summary_cache

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception:
            pass


def best_location_key(world_map: WorldMap, location: str) -> str:
    """Return the closest matching key already in world_map, handling vision model name drift.

    Handles cases like 'Prof. Birch's House (1F)' vs 'Prof. Birch's Lab' — the vision
    model sometimes hallucinates building names. Priority:
      1. Exact match
      2. Strip floor suffix  ' (1F)', ' (2F)', etc.
      3. Case-insensitive exact
      4. Same town prefix + 2+ shared content words (e.g. both contain 'Birch')
    Falls back to the original string (creates a new world map entry).
    """
    locs = world_map.data.get("locations", {})
    if not locs or not location:
        return location
    if location in locs:
        return location
    # Strip floor suffix
    base = re.sub(r"\s*\([^)]*\)\s*$", "", location).strip()
    if base and base in locs:
        return base
    # Case-insensitive exact
    loc_lower = location.lower()
    for k in locs:
        if k.lower() == loc_lower:
            return k
    # Same town, fuzzy building name (share ≥2 meaningful words)
    _stop = {"the", "a", "of", "in", "at", "s", "1f", "2f", "b1f", "town", "city", "route"}
    parts = location.split(" - ", 1)
    if len(parts) == 2:
        town, building = parts[0].lower(), parts[1].lower()
        building_words = set(re.findall(r"\b\w+\b", building)) - _stop
        for k in locs:
            k_parts = k.split(" - ", 1)
            if len(k_parts) == 2 and k_parts[0].lower() == town:
                k_words = set(re.findall(r"\b\w+\b", k_parts[1].lower())) - _stop
                if len(building_words & k_words) >= 2:
                    return k
    return location
