"""
nova_agent.state
================
RAM reading and screen-type detection.

Unlike pyboy_agent which has separate reader.py / formatter.py modules,
nova_agent collapses these into one module.  The LLM sees RAM facts via
the visual overlay on the screenshot, not as a separate text block — so
we only need the structured dict here, not a formatted prompt string.

Screen-type detection
---------------------
Instead of 12 ad-hoc nav hints, nova_agent uses a 5-state machine:
  OVERWORLD  — player is walking around the world
  DIALOGUE   — a text box is open (talk to NPC, item found, etc.)
  BATTLE     — in a trainer or wild battle
  MENU       — main menu / PC / bag / etc. (non-dialogue, non-battle)
  CUTSCENE   — warp active or unusual state

The detected state is passed to the tool dispatcher so it can narrow
the available tools and inject a focused hint.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyboy import PyBoy


# ---------------------------------------------------------------------------
# Screen state enum
# ---------------------------------------------------------------------------

class ScreenType(Enum):
    OVERWORLD = auto()
    DIALOGUE  = auto()
    BATTLE    = auto()
    MENU      = auto()
    CUTSCENE  = auto()
    UNKNOWN   = auto()


# ---------------------------------------------------------------------------
# RAM reading
# ---------------------------------------------------------------------------

def read_ram(pyboy: "PyBoy", ram_offsets: dict[str, str]) -> dict[str, Any]:
    """Read all useful game state from WRAM and return a plain dict.

    Reads are wrapped in try/except — a bad address never crashes the loop.
    All addresses come from the game profile ``ram_offsets`` dict.
    """
    state: dict[str, Any] = {}

    # ── Map position ──────────────────────────────────────────────────────
    for key in ("map_bank", "map_number", "x_pos", "y_pos"):
        try:
            addr = int(ram_offsets.get(key, "0"), 16)
            state[key] = pyboy.memory[addr]
        except Exception:
            state[key] = None

    # ── Screen / menu flags ───────────────────────────────────────────────
    try:
        df_addr = int(ram_offsets.get("dialogue_flag", ram_offsets.get("text_flags", "0xC4F2")), 16)
        state["dialogue_open"] = bool(pyboy.memory[df_addr] & 0x01)
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

    # ── Battle flags ──────────────────────────────────────────────────────
    try:
        addr = int(ram_offsets.get("in_battle_flag", ram_offsets.get("battle_type_flag", "0xD116")), 16)
        state["in_battle"] = pyboy.memory[addr] != 0
        state["battle_type_val"] = pyboy.memory[addr]
    except Exception:
        state["in_battle"] = False
        state["battle_type_val"] = 0

    # ── Lead Pokémon HP ───────────────────────────────────────────────────
    try:
        hp_cur_addr = int(ram_offsets.get("party_slot0_hp_current", "0xDA4C"), 16)
        hp_max_addr = int(ram_offsets.get("party_slot0_hp_max",     "0xDA4E"), 16)
        hp_cur = (pyboy.memory[hp_cur_addr] << 8) | pyboy.memory[hp_cur_addr + 1]
        hp_max = (pyboy.memory[hp_max_addr] << 8) | pyboy.memory[hp_max_addr + 1]
        state["lead_hp_current"] = hp_cur
        state["lead_hp_max"] = hp_max
        state["lead_hp_pct"] = int(hp_cur * 100 / hp_max) if hp_max > 0 else 0
    except Exception:
        state["lead_hp_current"] = None
        state["lead_hp_max"] = None
        state["lead_hp_pct"] = None

    # ── Lead Pokémon level & species ─────────────────────────────────────
    try:
        state["lead_level"] = pyboy.memory[int(ram_offsets.get("party_slot0_level", "0xDA49"), 16)]
    except Exception:
        state["lead_level"] = None

    try:
        state["lead_species_id"] = pyboy.memory[int(ram_offsets.get("party_slot0_species", "0xDA2A"), 16)]
    except Exception:
        state["lead_species_id"] = None

    # ── Party count ───────────────────────────────────────────────────────
    try:
        state["party_count"] = pyboy.memory[int(ram_offsets.get("party_count", "0xDA22"), 16)]
    except Exception:
        state["party_count"] = None

    # ── Full party (HP for all slots) ─────────────────────────────────────
    slots_hp: list[dict[str, int]] = []
    _slots = [
        ("party_slot0_hp_current", "party_slot0_hp_max", "party_slot0_level"),
        ("party_slot1_hp_current", "party_slot1_hp_max", "party_slot1_level"),
        ("party_slot2_hp_current", "party_slot2_hp_max", "party_slot2_level"),
        ("party_slot3_hp_current", "party_slot3_hp_max", "party_slot3_level"),
        ("party_slot4_hp_current", "party_slot4_hp_max", "party_slot4_level"),
        ("party_slot5_hp_current", "party_slot5_hp_max", "party_slot5_level"),
    ]
    _defaults = [
        ("0xDA4C", "0xDA4E", "0xDA49"),
        ("0xDA7C", "0xDA7E", "0xDA79"),
        ("0xDAAC", "0xDAAE", "0xDAA9"),
        ("0xDADC", "0xDADE", "0xDAD9"),
        ("0xDB0C", "0xDB0E", "0xDB09"),
        ("0xDB3C", "0xDB3E", "0xDB39"),
    ]
    count = state.get("party_count") or 0
    for i in range(min(count, 6)):
        try:
            cur_key, max_key, lvl_key = _slots[i]
            cur_def, max_def, lvl_def = _defaults[i]
            cur_addr = int(ram_offsets.get(cur_key, cur_def), 16)
            max_addr = int(ram_offsets.get(max_key, max_def), 16)
            lvl_addr = int(ram_offsets.get(lvl_key, lvl_def), 16)
            cur = (pyboy.memory[cur_addr] << 8) | pyboy.memory[cur_addr + 1]
            mx  = (pyboy.memory[max_addr] << 8) | pyboy.memory[max_addr + 1]
            lvl = pyboy.memory[lvl_addr]
            slots_hp.append({"slot": i, "hp_cur": cur, "hp_max": mx, "level": lvl})
        except Exception:
            pass
    state["party_slots"] = slots_hp
    state["all_fainted"] = all(s["hp_cur"] == 0 for s in slots_hp) if slots_hp else False

    # ── Badges ────────────────────────────────────────────────────────────
    try:
        j_mask = pyboy.memory[int(ram_offsets.get("johto_badges_bitmask", "0xD57C"), 16)]
        state["johto_badge_count"] = bin(j_mask).count("1")
        state["johto_badge_mask"] = j_mask
    except Exception:
        state["johto_badge_count"] = 0
        state["johto_badge_mask"] = 0

    try:
        k_mask = pyboy.memory[int(ram_offsets.get("kanto_badges_bitmask", "0xD57D"), 16)]
        state["kanto_badge_count"] = bin(k_mask).count("1")
    except Exception:
        state["kanto_badge_count"] = 0

    # ── Money ─────────────────────────────────────────────────────────────
    try:
        money_addr = int(ram_offsets.get("money", "0xD573"), 16)
        # 3-byte BCD: each nibble is one decimal digit.
        raw = [pyboy.memory[money_addr + i] for i in range(3)]
        money = 0
        for byte in raw:
            money = money * 100 + (byte >> 4) * 10 + (byte & 0x0F)
        state["money"] = money
    except Exception:
        state["money"] = None

    # ── Enemy (battle) ───────────────────────────────────────────────────
    if state.get("in_battle"):
        try:
            e_hp_cur_addr = int(ram_offsets.get("enemy_hp_current", "0xD0FF"), 16)
            e_hp_max_addr = int(ram_offsets.get("enemy_hp_max",     "0xD101"), 16)
            e_lvl_addr    = int(ram_offsets.get("enemy_level",      "0xD0FC"), 16)
            e_sp_addr     = int(ram_offsets.get("enemy_species",    "0xD0ED"), 16)
            e_hp_cur = (pyboy.memory[e_hp_cur_addr] << 8) | pyboy.memory[e_hp_cur_addr + 1]
            e_hp_max = (pyboy.memory[e_hp_max_addr] << 8) | pyboy.memory[e_hp_max_addr + 1]
            state["enemy"] = {
                "hp_cur": e_hp_cur,
                "hp_max": e_hp_max,
                "level": pyboy.memory[e_lvl_addr],
                "species_id": pyboy.memory[e_sp_addr],
            }
        except Exception:
            state["enemy"] = None
    else:
        state["enemy"] = None

    return state


# ---------------------------------------------------------------------------
# Screen type detection
# ---------------------------------------------------------------------------

def detect_screen_type(ram_state: dict[str, Any]) -> ScreenType:
    """Classify the current screen based on RAM flags.

    Priority: BATTLE > DIALOGUE > MENU > CUTSCENE > OVERWORLD.
    """
    if ram_state.get("in_battle"):
        return ScreenType.BATTLE
    if ram_state.get("dialogue_open"):
        return ScreenType.DIALOGUE
    if ram_state.get("menu_open"):
        return ScreenType.MENU
    if ram_state.get("warp_active"):
        return ScreenType.CUTSCENE
    return ScreenType.OVERWORLD


# ---------------------------------------------------------------------------
# State-specific hint strings injected into the LLM system prompt
# ---------------------------------------------------------------------------

SCREEN_HINTS: dict[ScreenType, str] = {
    ScreenType.OVERWORLD: (
        "You are in the OVERWORLD. Use navigate_to(x, y) to move toward a "
        "destination, or press_buttons for fine control. Update your knowledge "
        "base when you discover new locations or facts."
    ),
    ScreenType.DIALOGUE: (
        "A DIALOGUE BOX is open. Read the text carefully. Press A to advance "
        "or B to close/cancel. Copy important information to the knowledge base "
        "using update_knowledge('notes', ...)."
    ),
    ScreenType.BATTLE: (
        "You are in a BATTLE. Use press_buttons to navigate the fight menu. "
        "Prefer super-effective moves. If all Pokémon are low on HP, consider "
        "using items or switching."
    ),
    ScreenType.MENU: (
        "A MENU is open (not battle). Navigate with Up/Down/A/B. "
        "Use B to close menus you don't need."
    ),
    ScreenType.CUTSCENE: (
        "A CUTSCENE or WARP is active. Wait for it to finish by pressing A "
        "or letting it play out."
    ),
    ScreenType.UNKNOWN: (
        "Screen state is unclear from RAM. Look at the screenshot carefully "
        "and press A or B to advance any pending UI."
    ),
}
