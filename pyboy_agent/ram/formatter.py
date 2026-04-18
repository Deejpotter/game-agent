"""
pyboy_agent.ram.formatter
=========================
Converts a RAM state dict (from ``ram.reader.read_ram_state``) into a compact
human-readable text block that is prepended to the reasoning model's prompt.

The formatter is intentionally terse — every line the model has to read costs
tokens. Only data that meaningfully influences the next action is included.
"""

from __future__ import annotations

from typing import Any


# Full HM list in acquisition order (used to print ✓/✗ for each).
_ALL_HMS = ["Cut", "Fly", "Surf", "Strength", "Flash", "Whirlpool", "Waterfall"]


def format_ram_state(state: dict[str, Any]) -> str:
    """Format a RAM state dict as a concise block for the reasoning model prompt.

    The block begins with the header ``RAM STATE (authoritative — trust over
    vision model):`` which tells the model to prefer RAM values over what it
    perceives visually.

    Args:
        state: Dict returned by ``read_ram_state()``.

    Returns:
        Multi-line string.  Empty if ``state`` is empty.
    """
    if not state:
        return ""

    lines: list[str] = ["RAM STATE (authoritative — trust over vision model):"]

    # ── Position ───────────────────────────────────────────────────────────
    name  = state.get("player_name") or "?"
    bank  = state.get("map_bank")
    map_n = state.get("map_number")
    x     = state.get("x_pos")
    y     = state.get("y_pos")
    lines.append(f"  Player: {name} | Map: bank={bank} map={map_n} | Pos: X={x} Y={y}")

    # ── Badges ─────────────────────────────────────────────────────────────
    johto   = state.get("johto_badges", [])
    kanto   = state.get("kanto_badges", [])
    j_count = state.get("johto_badge_count", 0)
    badges_str = ", ".join(johto) if johto else "none"
    lines.append(f"  Johto badges ({j_count}/8): {badges_str}")
    if kanto:
        lines.append(f"  Kanto badges: {', '.join(kanto)}")

    # ── Battle ─────────────────────────────────────────────────────────────
    if state.get("in_battle"):
        btype_val = state.get("battle_type_val", 0)
        btype_str = "Wild" if btype_val == 1 else ("Trainer" if btype_val == 2 else "Battle")
        enemy = state.get("enemy_info")
        if enemy:
            e_hp_str = f"HP {enemy['hp_cur']}/{enemy['hp_max']} ({enemy['hp_pct']}%)"
            lines.append(
                f"  BATTLE ({btype_str}): Enemy Lv.{enemy['level']} | "
                f"{e_hp_str} | Type: {enemy['types']}"
            )
        else:
            lines.append(f"  BATTLE ({btype_str}): enemy stats unavailable")

    # ── Lead Pokemon ────────────────────────────────────────────────────────
    hp_cur  = state.get("lead_hp_current")
    hp_max  = state.get("lead_hp_max")
    level   = state.get("lead_level")
    pct     = state.get("lead_hp_pct")
    party   = state.get("party_count")
    hp_stable = state.get("hp_stabilised", True)

    if hp_max:
        heal_warn = ""
        if hp_stable and pct is not None and pct < 25:
            heal_warn = " ⚠ LOW HP — HEAL NOW"

        moves_str = ""
        lead_moves = state.get("lead_moves", [])
        if lead_moves:
            moves_str = f" | moves: {', '.join(lead_moves)}"

        lines.append(
            f"  Lead Pokemon: Lv.{level} HP {hp_cur}/{hp_max} ({pct}%)"
            f"{heal_warn}{moves_str}"
        )

    # ── Full party summary ──────────────────────────────────────────────────
    party_slots = state.get("party_slots", [])
    if party_slots and len(party_slots) > 1:
        slot_parts: list[str] = []
        for i, s in enumerate(party_slots):
            if s["fainted"]:
                slot_parts.append(f"Slot{i+1}:Lv{s['level']} FAINTED")
            elif s["hp_pct"] < 25:
                slot_parts.append(
                    f"Slot{i+1}:Lv{s['level']} {s['hp_cur']}/{s['hp_max']}({s['hp_pct']}%)⚠"
                )
            else:
                slot_parts.append(
                    f"Slot{i+1}:Lv{s['level']} {s['hp_cur']}/{s['hp_max']}({s['hp_pct']}%)"
                )
        lines.append(f"  Party ({len(party_slots)}/{party}): {' | '.join(slot_parts)}")

    if state.get("all_fainted"):
        lines.append(
            "  ⚠ BLACKOUT — All Pokemon fainted! You have been returned to the last "
            "Pokemon Center. Navigate back to your goal area."
        )

    if party is not None and not party_slots:
        lines.append(f"  Party size: {party}")

    # ── Money ───────────────────────────────────────────────────────────────
    money = state.get("money")
    if money is not None:
        lines.append(f"  Money: ¥{money:,}")

    # ── HMs ─────────────────────────────────────────────────────────────────
    hms = set(state.get("hms_obtained", []))
    hm_display = " ".join(f"{h}{'✓' if h in hms else '✗'}" for h in _ALL_HMS)
    lines.append(f"  HMs: {hm_display}")

    return "\n".join(lines)
