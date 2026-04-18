"""
pyboy_agent.ram.reader
======================
Reads ground-truth game state from PyBoy emulator WRAM.

``read_ram_state(pyboy, ram_offsets)`` is the single entry point. It returns
a plain dict that ``ram.formatter.format_ram_state()`` turns into a human-
readable string for injection into the reasoning model's prompt.

All reads are wrapped in try/except so a single bad address never crashes the
agent loop — the relevant field is simply omitted or set to None.

Party encoding duality
-----------------------
Some profiles use compact pret/pokegold WRAM layout (``party_struct_base`` +
``party_struct_stride``) while others enumerate per-slot addresses
(``party_slot0_hp_current``, etc.).  Both are supported here.

HP stabilisation
-----------------
After ``pyboy.load_state()`` the HP bytes sometimes read 0/0 for a few frames.
The returned dict includes ``hp_stabilised: bool`` which callers should check
before displaying or acting on HP-based warnings.  The caller tracks the turn
counter for this — see ``loop.py`` for the ``_hp_valid_turns`` logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyboy_agent.ram.gen2_tables import (
    _GEN2_MOVE,
    _GEN2_TYPE,
    _JOHTO_BADGES,
    _KANTO_BADGES,
    decode_gen2_name,
    read_bcd,
)

if TYPE_CHECKING:
    from pyboy import PyBoy


def read_ram_state(pyboy: "PyBoy", ram_offsets: dict[str, str]) -> dict[str, Any]:
    """Read all meaningful game state from WRAM and return it as a plain dict.

    Args:
        pyboy: Running PyBoy emulator instance.
        ram_offsets: Mapping of semantic key → hex address string, as loaded
                     from the game profile JSON (e.g. ``{"x_pos": "0xDA02"}``).

    Returns:
        Dict with decoded values.  Keys that could not be read are None or
        use safe defaults (empty list, False, etc.).
    """
    state: dict[str, Any] = {}

    # ── Player name ───────────────────────────────────────────────────────
    try:
        name_addr = int(ram_offsets.get("player_name_start", "0"), 16)
        name_len = int(ram_offsets.get("player_name_length", 10))
        state["player_name"] = decode_gen2_name(pyboy, name_addr, name_len)
    except Exception:
        state["player_name"] = "?"

    # ── Map coordinates ───────────────────────────────────────────────────
    # These four values form the tile key used by wall detection and the
    # world map.  Reads 0xDA00-0xDA03 for Pokemon Silver (US).
    for key in ("map_bank", "map_number", "x_pos", "y_pos"):
        try:
            addr = int(ram_offsets.get(key, "0"), 16)
            state[key] = pyboy.memory[addr]
        except Exception:
            state[key] = None

    # ── UI / menu flags ───────────────────────────────────────────────────
    try:
        # dialogue_flag / text_flags — defaults to Johto overworld address
        df_addr_str = (
            ram_offsets.get("dialogue_flag")
            or ram_offsets.get("text_flags")
            or "0xC4F2"
        )
        state["dialogue_open"] = bool(pyboy.memory[int(df_addr_str, 16)] & 0x01)
    except Exception:
        state["dialogue_open"] = None

    try:
        state["menu_open"] = bool(
            pyboy.memory[int(ram_offsets.get("menu_open_flag", "0xD72D"), 16)]
        )
    except Exception:
        state["menu_open"] = None

    try:
        state["warp_active"] = bool(
            pyboy.memory[int(ram_offsets.get("warp_active_flag", "0xD2F4"), 16)]
        )
    except Exception:
        state["warp_active"] = None

    # ── Badges ────────────────────────────────────────────────────────────
    try:
        johto_mask = pyboy.memory[int(ram_offsets.get("johto_badges_bitmask", "0xD57C"), 16)]
        state["johto_badges"] = [name for bit, name in _JOHTO_BADGES if johto_mask & bit]
        state["johto_badge_count"] = len(state["johto_badges"])
    except Exception:
        state["johto_badges"] = []
        state["johto_badge_count"] = 0

    try:
        kanto_mask = pyboy.memory[int(ram_offsets.get("kanto_badges_bitmask", "0xD57D"), 16)]
        state["kanto_badges"] = [name for bit, name in _KANTO_BADGES if kanto_mask & bit]
    except Exception:
        state["kanto_badges"] = []

    # ── Money (BCD-encoded, 3 bytes) ──────────────────────────────────────
    try:
        money_addr = int(ram_offsets.get("money", "0xD573"), 16)
        money_len = int(ram_offsets.get("money_length", 3))
        state["money"] = read_bcd(pyboy, money_addr, money_len)
    except Exception:
        state["money"] = None

    # ── Party count + active party slot ──────────────────────────────────
    try:
        state["party_count"] = pyboy.memory[int(ram_offsets.get("party_count", "0xDA22"), 16)]
    except Exception:
        state["party_count"] = None

    try:
        state["current_party_mon"] = pyboy.memory[
            int(ram_offsets.get("current_party_mon", "0xDCCA"), 16)
        ]
    except Exception:
        state["current_party_mon"] = 0

    # ── Party HP / level / moves ─────────────────────────────────────────
    # Supports two WRAM layouts:
    #   A) pret/pokegold compact struct  (party_struct_base + party_struct_stride)
    #   B) per-slot address enumeration  (party_slot0_hp_current, etc.)
    _count = state.get("party_count") or 0

    try:
        if "party_struct_base" in ram_offsets:
            _read_party_compact(pyboy, state, ram_offsets, _count)
        else:
            _read_party_slots(pyboy, state, ram_offsets, _count)
    except Exception:
        state["party_slots"] = []
        state["all_fainted"] = False
        state["any_low_hp"] = False

    # ── HM bag flags ──────────────────────────────────────────────────────
    # Addresses 0xD5B0-0xD5B6 store whether each HM has been obtained.
    # Non-zero = obtained.
    try:
        _hm_keys = [
            "hm01_cut", "hm02_fly", "hm03_surf", "hm04_strength",
            "hm05_flash", "hm06_whirlpool", "hm07_waterfall",
        ]
        _hm_names = ["Cut", "Fly", "Surf", "Strength", "Flash", "Whirlpool", "Waterfall"]
        _default_hm_addrs = [0xD5B0, 0xD5B1, 0xD5B2, 0xD5B3, 0xD5B4, 0xD5B5, 0xD5B6]
        state["hms_obtained"] = [
            _hm_names[i]
            for i, k in enumerate(_hm_keys)
            if pyboy.memory[int(ram_offsets.get(k, hex(_default_hm_addrs[i])), 16)] > 0
        ]
    except Exception:
        state["hms_obtained"] = []

    # ── Battle flags ──────────────────────────────────────────────────────
    try:
        # in_battle_flag and battle_type_flag share the same address in Silver (0xD116).
        _in_b_addr = int(
            ram_offsets.get("in_battle_flag", ram_offsets.get("battle_type_flag", "0xD116")),
            16,
        )
        _in_b_val = pyboy.memory[_in_b_addr]
        state["in_battle"] = _in_b_val != 0
        _bt_addr = int(ram_offsets.get("battle_type_flag", hex(_in_b_addr)), 16)
        state["battle_type_val"] = pyboy.memory[_bt_addr]
    except Exception:
        state["in_battle"] = False
        state["battle_type_val"] = 0

    # ── Enemy stats (only populated during battle) ────────────────────────
    if state.get("in_battle"):
        state["enemy_info"] = _read_enemy(pyboy, ram_offsets)
        state["enemy_species"] = (state["enemy_info"] or {}).get("species")
    else:
        state["enemy_info"] = None
        state["enemy_species"] = None

    # hp_stabilised is set externally by the loop (tracks frame counter after load_state).
    # Default True here so callers that don't set it still get sensible output.
    state.setdefault("hp_stabilised", True)

    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_party_compact(
    pyboy: "PyBoy",
    state: dict[str, Any],
    ram_offsets: dict[str, str],
    count: int,
) -> None:
    """Read party data from the compact pret/pokegold WRAM struct layout."""
    base_addr = int(ram_offsets["party_struct_base"], 16)
    stride = int(ram_offsets.get("party_struct_stride", "0x2C"), 16)

    active_idx = max(0, min(int(state.get("current_party_mon", 0) or 0), count - 1))

    if count > 0:
        lead_base = base_addr + active_idx * stride
        level = pyboy.memory[lead_base + 0x1F]
        hp_cur = (pyboy.memory[lead_base + 0x22] << 8) | pyboy.memory[lead_base + 0x23]
        hp_max = (pyboy.memory[lead_base + 0x24] << 8) | pyboy.memory[lead_base + 0x25]
        state["lead_level"] = level
        state["lead_hp_current"] = hp_cur
        state["lead_hp_max"] = hp_max
        state["lead_hp_pct"] = round(100 * hp_cur / hp_max) if hp_max > 0 else 0
        moves = []
        for j in range(4):
            mv = pyboy.memory[lead_base + 0x02 + j]
            if mv:
                moves.append(_GEN2_MOVE.get(mv, f"Move#{mv}"))
        state["lead_moves"] = moves
    else:
        state["lead_level"] = None
        state["lead_hp_current"] = None
        state["lead_hp_max"] = None
        state["lead_hp_pct"] = None
        state["lead_moves"] = []

    slots = []
    for i in range(min(count, 6)):
        sbase = base_addr + i * stride
        lv = pyboy.memory[sbase + 0x1F]
        hc = (pyboy.memory[sbase + 0x22] << 8) | pyboy.memory[sbase + 0x23]
        hm = (pyboy.memory[sbase + 0x24] << 8) | pyboy.memory[sbase + 0x25]
        st = pyboy.memory[sbase + 0x1E] if (sbase + 0x1E) < 0x10000 else 0
        pct = round(100 * hc / hm) if hm > 0 else 0
        slots.append({
            "level": lv, "hp_cur": hc, "hp_max": hm,
            "hp_pct": pct, "status": st, "fainted": hm > 0 and hc == 0,
        })

    state["party_slots"] = slots
    state["all_fainted"] = len(slots) > 0 and all(s["fainted"] for s in slots)
    state["any_low_hp"] = any((not s["fainted"]) and s["hp_pct"] < 30 for s in slots)


def _read_party_slots(
    pyboy: "PyBoy",
    state: dict[str, Any],
    ram_offsets: dict[str, str],
    count: int,
) -> None:
    """Read party data from per-slot WRAM addresses (pret/pokegold Silver layout)."""
    # Lead (slot 0) — these addresses come from the game profile or use Silver defaults.
    hp_cur_addr = int(ram_offsets.get("party_slot0_hp_current", "0xDA4C"), 16)
    hp_max_addr = int(ram_offsets.get("party_slot0_hp_max", "0xDA4E"), 16)
    level_addr  = int(ram_offsets.get("party_slot0_level",      "0xDA49"), 16)

    # HP is big-endian 16-bit (high byte first).
    hp_cur = (pyboy.memory[hp_cur_addr] << 8) | pyboy.memory[hp_cur_addr + 1]
    hp_max = (pyboy.memory[hp_max_addr] << 8) | pyboy.memory[hp_max_addr + 1]
    level  = pyboy.memory[level_addr]

    state["lead_hp_current"] = hp_cur
    state["lead_hp_max"] = hp_max
    state["lead_level"] = level
    state["lead_hp_pct"] = round(100 * hp_cur / hp_max) if hp_max > 0 else 0

    # Lead move IDs decoded to move names (slots 0-3).
    _m_addrs = [
        int(ram_offsets.get("party_slot0_move1", "0xDA2C"), 16),
        int(ram_offsets.get("party_slot0_move2", "0xDA2D"), 16),
        int(ram_offsets.get("party_slot0_move3", "0xDA2E"), 16),
        int(ram_offsets.get("party_slot0_move4", "0xDA2F"), 16),
    ]
    state["lead_moves"] = [
        _GEN2_MOVE.get(pyboy.memory[a], f"Move#{pyboy.memory[a]}")
        for a in _m_addrs
        if pyboy.memory[a] != 0
    ]

    # All party slots (0-5) — HP, level, status.
    # These are the pret/pokegold Silver (US) WRAM addresses for each slot.
    _slot_hp_cur = [0xDA4C, 0xDA7C, 0xDAAC, 0xDADC, 0xDB0C, 0xDB3C]
    _slot_hp_max = [0xDA4E, 0xDA7E, 0xDAAE, 0xDADE, 0xDB0E, 0xDB3E]
    _slot_level  = [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
    _slot_status = [0xDA4A, 0xDA7A, 0xDAAA, 0xDADA, 0xDB0A, 0xDB3A]

    slots = []
    for i in range(min(count, 6)):
        hc = (pyboy.memory[_slot_hp_cur[i]] << 8) | pyboy.memory[_slot_hp_cur[i] + 1]
        hm = (pyboy.memory[_slot_hp_max[i]] << 8) | pyboy.memory[_slot_hp_max[i] + 1]
        lv = pyboy.memory[_slot_level[i]]
        st = pyboy.memory[_slot_status[i]]
        pct = round(100 * hc / hm) if hm > 0 else 0
        slots.append({
            "level": lv, "hp_cur": hc, "hp_max": hm,
            "hp_pct": pct, "status": st, "fainted": hm > 0 and hc == 0,
        })

    state["party_slots"] = slots
    state["all_fainted"] = count > 0 and all(s["fainted"] for s in slots)
    state["any_low_hp"] = any((not s["fainted"]) and s["hp_pct"] < 30 for s in slots)


def _read_enemy(pyboy: "PyBoy", ram_offsets: dict[str, str]) -> dict[str, Any] | None:
    """Read enemy Pokemon stats from WRAM. Returns None on any read failure."""
    try:
        es_addr   = int(ram_offsets.get("enemy_species",    "0xCFDE"), 16)
        e_hc_addr = int(ram_offsets.get("enemy_hp_current", "0xD0FF"), 16)
        e_hm_addr = int(ram_offsets.get("enemy_hp_max",     "0xD101"), 16)
        e_lv_addr = int(ram_offsets.get("enemy_level",      "0xD0FC"), 16)
        e_t1_addr = int(ram_offsets.get("enemy_type1",      "0xD127"), 16)
        e_t2_addr = int(ram_offsets.get("enemy_type2",      "0xD128"), 16)

        species = pyboy.memory[es_addr]
        # HP is big-endian 16-bit.
        e_hc = (pyboy.memory[e_hc_addr] << 8) | pyboy.memory[e_hc_addr + 1]
        e_hm = (pyboy.memory[e_hm_addr] << 8) | pyboy.memory[e_hm_addr + 1]
        e_lv = pyboy.memory[e_lv_addr]
        e_t1 = _GEN2_TYPE.get(pyboy.memory[e_t1_addr], f"Type{pyboy.memory[e_t1_addr]}")
        e_t2 = _GEN2_TYPE.get(pyboy.memory[e_t2_addr], f"Type{pyboy.memory[e_t2_addr]}")
        e_types = e_t1 if e_t1 == e_t2 else f"{e_t1}/{e_t2}"
        e_pct = round(100 * e_hc / e_hm) if e_hm > 0 else 0

        return {
            "species": species,
            "hp_cur": e_hc, "hp_max": e_hm, "hp_pct": e_pct,
            "level": e_lv, "types": e_types,
        }
    except Exception:
        return None
