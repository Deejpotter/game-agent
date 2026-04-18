"""
pyboy_agent.navigation.hints
=============================
Build the navigation hint string that is prepended to the reasoning model's
``stuck_hint`` parameter each turn.

The hint is assembled from multiple sources in priority order:

1. Operator override  (typed by the human operator this turn)
2. Wall detection     (previous button didn't move the character)
3. Consecutive-button warning  (stuck pressing the same button repeatedly)
4. Consecutive-A warning  (mashing A into an NPC or dialogue)
5. RAM tile hint      (confirmed walls/passable/untried from per-tile data)
6. Party-empty critical hint  (party_count=0 — must get starter)
7. Blackout hint      (all Pokémon fainted — go to last Pokémon Center)
8. Badge-phase story hint  (next story objective for current badge count)
9. Battle-mode hint   (type advantage guidance while in a battle)
10. Memory-indoor-but-outdoor correction
11. Battle screen override  (VLM scene_type=="battle")
12. NPC retalk guard  (adjacent NPC already talked to)

Each hint source either sets ``nav_hint`` outright or prepends to it.  The
final string is passed into ``decide()`` as ``stuck_hint``.
"""

from __future__ import annotations

from pyboy_agent.config import STUCK_BUTTON_THRESHOLD, CONSECUTIVE_A_THRESHOLD
from pyboy_agent.navigation.world_map import WorldMap
from pyboy_agent.goals.phase_guide import BADGE_PHASE_MAP


_CARDINAL = {"Up", "Down", "Left", "Right"}


def build_nav_hints(
    *,
    operator_msg: str | None,
    wall_detected: bool,
    wall_button: str | None,
    last_button: str | None,
    consecutive_same: int,
    consecutive_a: int,
    tile_key: str,
    has_pos: bool,
    cx: int | None,
    cy: int | None,
    map_bank: int | None,
    map_number: int | None,
    turns_at_same_tile: int,
    world_map: WorldMap,
    ram_state: dict,
    scene_parsed: dict,
    current_location: str,
    memory: str,
) -> str:
    """Assemble the navigation hint string for the current turn.

    Args:
        operator_msg: Human operator override typed this turn (or None).
        wall_detected: True if the previous button press didn't move the player.
        wall_button: The button that was blocked (or None).
        last_button: The button pressed on the previous turn.
        consecutive_same: How many turns in a row the same button has been pressed.
        consecutive_a: How many consecutive A presses have occurred.
        tile_key: Per-tile wall key (``map_{bank}_{number}_x{x}_y{y}``).
        has_pos: True if all four position bytes were readable from RAM this turn.
        cx, cy: Current X/Y tile coordinates from RAM.
        map_bank, map_number: Current map bank/number from RAM.
        turns_at_same_tile: Turns the player has spent on the same tile.
        world_map: WorldMap instance for wall/tested data.
        ram_state: Full RAM state dict from ``read_ram_state()``.
        scene_parsed: Parsed JSON dict from the vision model (may be ``{}``).
        current_location: VLM-reported location name (may be stale).
        memory: Agent memory string (for indoor/outdoor correction check).

    Returns:
        Assembled hint string (may be empty string if nothing is noteworthy).
    """
    nav_hint: str | None = None

    def _prepend(hint: str) -> None:
        nonlocal nav_hint
        nav_hint = (hint + " | " + nav_hint) if nav_hint else hint

    def _set(hint: str) -> None:
        nonlocal nav_hint
        nav_hint = hint

    # ── 1. Operator override (highest priority) ───────────────────────────
    if operator_msg:
        _set(f"OPERATOR INSTRUCTION (follow this immediately): {operator_msg}")

    # ── 2. Wall detection ─────────────────────────────────────────────────
    if wall_detected and wall_button:
        _prepend(
            f"Pressing {wall_button!r} did NOT move the character — you hit a "
            "wall or obstacle. Choose a DIFFERENT direction immediately."
        )

    # ── 3. Consecutive same-button warning ────────────────────────────────
    if consecutive_same >= STUCK_BUTTON_THRESHOLD:
        _prepend(
            f"You have pressed '{last_button}' {consecutive_same} times in a row. "
            "The character is stuck or looping. Try a completely different "
            "direction, or press B to close any open menus."
        )

    # ── 4. Consecutive A warning ──────────────────────────────────────────
    if consecutive_a >= CONSECUTIVE_A_THRESHOLD:
        _prepend(
            f"You have pressed A {consecutive_a} consecutive times. "
            "STOP pressing A. If stuck on same NPC: navigate away — "
            "press Down or the exit direction."
        )

    # ── 5. RAM tile hint ──────────────────────────────────────────────────
    _wlk = tile_key or current_location
    known_walls = world_map.get_walls(_wlk) if _wlk else set()
    _loc_entry = world_map.data.get("locations", {}).get(_wlk, {})
    _all_tested = {k for k, v in _loc_entry.get("tested", {}).items() if v}
    _passable = _all_tested - known_walls
    _untried = _CARDINAL - known_walls - _all_tested

    if has_pos and tile_key:
        _parts: list[str] = [f"RAM tile ({cx},{cy}) on map {map_bank}/{map_number}."]
        if known_walls:
            _parts.append(f"Blocked from this tile: {', '.join(sorted(known_walls))}.")
        if _passable:
            _parts.append(
                f"Previously passable from this tile: {', '.join(sorted(_passable))}. "
                "Use one of these to escape."
            )
        if _untried:
            _parts.append(
                f"Not yet tried from this tile: {', '.join(sorted(_untried))}. "
                "Try one of these."
            )
        if turns_at_same_tile >= 2:
            _parts.append(
                f"You have NOT moved for {turns_at_same_tile} turns — "
                "character is stuck at this tile."
            )
        _prepend(" ".join(_parts))
    elif known_walls:
        _prepend(
            f"Blocked directions at current position: "
            f"{', '.join(sorted(known_walls))}. Do NOT try these."
        )

    # ── 6. Party-empty critical hint ──────────────────────────────────────
    party_count = ram_state.get("party_count")
    if party_count == 0:
        _prepend(
            "CRITICAL: RAM confirms party_count=0 — you have NO Pokémon. "
            "Your ONLY goal right now is to reach Prof. Elm's Lab in New Bark Town "
            "and press A in front of him to receive your starter Pokémon. "
            "Ignore building exploration until you have a Pokémon."
        )

    # ── 7. Blackout hint ──────────────────────────────────────────────────
    if ram_state.get("all_fainted"):
        _prepend(
            "BLACKOUT: All your Pokémon have fainted. You have been warped to the "
            "last Pokémon Center you visited. Heal at the counter (approach nurse, "
            "press A), then resume traveling toward your goal."
        )

    # ── 8. Badge-phase story hint ─────────────────────────────────────────
    badge_count = ram_state.get("johto_badge_count", 0)
    if isinstance(badge_count, int) and badge_count < 8 and badge_count in BADGE_PHASE_MAP:
        _prepend(BADGE_PHASE_MAP[badge_count])

    # ── 9. Battle hint ────────────────────────────────────────────────────
    if ram_state.get("in_battle") and ram_state.get("enemy_info"):
        einfo = ram_state["enemy_info"]
        lead_moves = ram_state.get("lead_moves", [])
        moves_str = ", ".join(lead_moves) if lead_moves else "unknown"
        _prepend(
            f"BATTLE: Enemy Lv.{einfo['level']} type={einfo['types']} "
            f"HP {einfo['hp_cur']}/{einfo['hp_max']} ({einfo['hp_pct']}%). "
            f"Your lead moves: {moves_str}. "
            "Navigate FIGHT menu with directions (repeat=1), confirm with A. "
            "Pick the move with the best type advantage against the enemy's type."
        )

    # ── 10. Memory indoor/outdoor correction ─────────────────────────────
    if scene_parsed.get("is_outdoor") and memory:
        _indoor_words = [
            "inside", "indoor", "trapped", "lab", "building",
            "house", "center", "mart", "stuck in",
        ]
        if any(w in memory.lower() for w in _indoor_words):
            _prepend(
                "MEMORY CORRECTION: The current scene shows is_outdoor=TRUE — "
                "you are OUTSIDE on the overworld, NOT inside any building. "
                f"Location: {scene_parsed.get('location_name', 'unknown outdoor area')}."
            )

    # ── 11. Battle screen override ────────────────────────────────────────
    if scene_parsed.get("screen_type") == "battle":
        _prepend(
            "YOU ARE IN A BATTLE. Ignore memory about overworld navigation. "
            "Use FIGHT: navigate to it with directions (repeat=1), then press A. "
            "Do NOT press directions with repeat>1 in a battle."
        )

    # ── 12. NPC retalk guard ──────────────────────────────────────────────
    if scene_parsed.get("adjacent_npc") and current_location:
        adj_id = (scene_parsed.get("adjacent_npc_id") or "").strip()
        loc_npcs = (
            world_map.data.get("locations", {})
            .get(current_location, {})
            .get("npcs", {})
        )
        already_talked = {k for k, v in loc_npcs.items() if v.get("status") == "talked"}
        if adj_id and adj_id in already_talked:
            _prepend(
                f"'{adj_id}' is already marked 'talked' in the world map. "
                "Do NOT press A again. Walk to the exit — press Down."
            )
        elif not adj_id and already_talked:
            _prepend(
                f"Adjacent NPC unidentified. All known NPCs in '{current_location}' "
                f"are already 'talked': {', '.join(sorted(already_talked))}. "
                "Walk away, press Down toward the exit mat."
            )

    return nav_hint or ""
