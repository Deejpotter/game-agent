"""
mgba_agent/ram/reader.py — GBA RAM state reader for game profiles.

GameState reads structured game data (party HP, badges, money, map ID, battle
flag) from GBA RAM via BridgeClient.read_range(). Addresses are loaded from
the game profile's ram_offsets dict.

Per-Pokemon parsing uses Gen 3 (Ruby/Sapphire/FireRed/LeafGreen) struct layout.
"""

from __future__ import annotations

from typing import Any

from ..bridge.client import BridgeClient


class GameState:
    """Reads game state directly from GBA RAM via the bridge.

    Addresses are loaded from the game profile's ram_offsets. On each call to
    read(), all configured addresses are read in bulk and parsed into a
    structured dict that can be injected into the VLM prompt or used for
    mechanical decision-making (e.g. skipping VLM calls during battles).
    """

    # Per-Pokemon struct offsets (from party base), Gen 3 Ruby/Sapphire
    _PKMN_SIZE = 100  # bytes per Pokemon in party
    _PKMN_OFFSETS = {
        "nickname": (0x08, 10, "str"),   # 10 bytes, Gen 3 encoding
        "level":    (0x54, 1, "u8"),
        "hp":       (0x56, 2, "u16"),
        "max_hp":   (0x58, 2, "u16"),
        "status":   (0x50, 4, "u32"),    # status condition bitfield
        "attack":   (0x5A, 2, "u16"),
        "defense":  (0x5C, 2, "u16"),
        "speed":    (0x5E, 2, "u16"),
        "sp_atk":   (0x60, 2, "u16"),
        "sp_def":   (0x62, 2, "u16"),
    }

    # Gen 3 character encoding table (subset — covers A-Z, a-z, 0-9, common)
    _CHARMAP = {
        0xBB: 'A', 0xBC: 'B', 0xBD: 'C', 0xBE: 'D', 0xBF: 'E',
        0xC0: 'F', 0xC1: 'G', 0xC2: 'H', 0xC3: 'I', 0xC4: 'J',
        0xC5: 'K', 0xC6: 'L', 0xC7: 'M', 0xC8: 'N', 0xC9: 'O',
        0xCA: 'P', 0xCB: 'Q', 0xCC: 'R', 0xCD: 'S', 0xCE: 'T',
        0xCF: 'U', 0xD0: 'V', 0xD1: 'W', 0xD2: 'X', 0xD3: 'Y',
        0xD4: 'Z', 0xD5: 'a', 0xD6: 'b', 0xD7: 'c', 0xD8: 'd',
        0xD9: 'e', 0xDA: 'f', 0xDB: 'g', 0xDC: 'h', 0xDD: 'i',
        0xDE: 'j', 0xDF: 'k', 0xE0: 'l', 0xE1: 'm', 0xE2: 'n',
        0xE3: 'o', 0xE4: 'p', 0xE5: 'q', 0xE6: 'r', 0xE7: 's',
        0xE8: 't', 0xE9: 'u', 0xEA: 'v', 0xEB: 'w', 0xEC: 'x',
        0xED: 'y', 0xEE: 'z', 0xA1: '0', 0xA2: '1', 0xA3: '2',
        0xA4: '3', 0xA5: '4', 0xA6: '5', 0xA7: '6', 0xA8: '7',
        0xA9: '8', 0xAA: '9', 0xAB: '!', 0xAC: '?', 0xAD: '.',
        0xB0: '-', 0x00: ' ', 0xFF: '',  # 0xFF = terminator
    }

    def __init__(self, ram_offsets: dict[str, Any]) -> None:
        self._offsets = ram_offsets
        # Parse hex address strings into ints
        self._addr: dict[str, int] = {}
        for key, val in ram_offsets.items():
            if isinstance(val, str) and val.startswith("0x"):
                self._addr[key] = int(val, 16)

    def _decode_name(self, data: list[int]) -> str:
        """Decode a Gen 3 encoded string from raw bytes."""
        chars = []
        for b in data:
            if b == 0xFF:
                break
            chars.append(self._CHARMAP.get(b, '?'))
        return ''.join(chars)

    def _parse_status(self, status_u32: int) -> str:
        """Convert status condition bitfield to human-readable string."""
        if status_u32 == 0:
            return "healthy"
        parts = []
        slp = status_u32 & 0x07
        if slp:
            parts.append(f"SLP({slp})")
        if status_u32 & 0x08:
            parts.append("PSN")
        if status_u32 & 0x10:
            parts.append("BRN")
        if status_u32 & 0x20:
            parts.append("FRZ")
        if status_u32 & 0x40:
            parts.append("PAR")
        if status_u32 & 0x80:
            parts.append("TOX")  # bad poison
        return "+".join(parts) if parts else "healthy"

    async def read(self, bridge: BridgeClient) -> dict[str, Any]:
        """Read all game state from RAM and return a structured dict."""
        state: dict[str, Any] = {}

        # -- Battle flag --
        # gBattleTypeFlags at 0x020239F8, 2 bytes. Non-zero = in battle.
        battle_flags_addr = self._addr.get("battle_type_flags", 0x020239F8)
        try:
            battle_flags = await bridge.read_u16(battle_flags_addr)
            state["in_battle"] = battle_flags != 0
            state["battle_type_flags"] = battle_flags
        except Exception:
            state["in_battle"] = None

        # -- Player party (batched: 1 IPC call for all 6 slots) --
        party_base = self._addr.get("party_base", 0x03004360)
        try:
            # Read party count first (gPlayerPartyCount)
            party_count_addr = self._addr.get("party_count")
            if party_count_addr:
                party_count = await bridge.read_u8(party_count_addr)
            else:
                party_count = 6

            party_count = min(party_count, 6)
            party = []
            if party_count > 0:
                # Single bulk read for all party members
                bulk = await bridge.read_range(party_base, party_count * self._PKMN_SIZE)
                for i in range(party_count):
                    offset = i * self._PKMN_SIZE
                    raw = bulk[offset:offset + self._PKMN_SIZE]
                    if len(raw) < self._PKMN_SIZE:
                        continue

                    level = raw[0x54]
                    if level == 0 or level > 100:
                        continue  # empty slot

                    hp = raw[0x56] | (raw[0x57] << 8)
                    max_hp = raw[0x58] | (raw[0x59] << 8)
                    status = raw[0x50] | (raw[0x51] << 8) | (raw[0x52] << 16) | (raw[0x53] << 24)

                    pkmn = {
                        "slot": i + 1,
                        "nickname": self._decode_name(raw[0x08:0x12]),
                        "level": level,
                        "hp": hp,
                        "max_hp": max_hp,
                        "hp_pct": round(hp / max_hp * 100) if max_hp > 0 else 0,
                        "status": self._parse_status(status),
                    }
                    party.append(pkmn)

            state["party"] = party
            state["party_count"] = len(party)
        except Exception as exc:
            state["party"] = []
            state["party_error"] = str(exc)
            print(f"  [ram] party read failed ({type(exc).__name__}): {exc}")

        # -- Enemy party (batched: 1 IPC call, only in battle) --
        if state.get("in_battle"):
            enemy_base = self._addr.get("enemy_party_base", 0x030045C0)
            try:
                enemies = []
                bulk = await bridge.read_range(enemy_base, 6 * self._PKMN_SIZE)
                for i in range(6):
                    offset = i * self._PKMN_SIZE
                    raw = bulk[offset:offset + self._PKMN_SIZE]
                    if len(raw) < self._PKMN_SIZE:
                        continue
                    level = raw[0x54]
                    if level == 0 or level > 100:
                        continue
                    hp = raw[0x56] | (raw[0x57] << 8)
                    max_hp = raw[0x58] | (raw[0x59] << 8)
                    enemies.append({
                        "slot": i + 1,
                        "nickname": self._decode_name(raw[0x08:0x12]),
                        "level": level,
                        "hp": hp,
                        "max_hp": max_hp,
                        "hp_pct": round(hp / max_hp * 100) if max_hp > 0 else 0,
                    })
                state["enemies"] = enemies
            except Exception as exc:
                state["enemies"] = []
                print(f"  [ram] enemy read failed ({type(exc).__name__}): {exc}")

        # -- Badges --
        badges_addr = self._addr.get("badges_bitmask")
        if badges_addr:
            try:
                badges_raw = await bridge.read_u16(badges_addr)
                state["badges"] = bin(badges_raw).count("1")
                state["badges_bitmask"] = badges_raw
            except Exception as exc:
                print(f"  [ram] badges read failed ({type(exc).__name__}): {exc}")

        # -- Money --
        money_addr = self._addr.get("money")
        if money_addr:
            try:
                state["money"] = await bridge.read_u32(money_addr)
            except Exception as exc:
                print(f"  [ram] money read failed ({type(exc).__name__}): {exc}")

        # -- Map ID (group + number as u16) --
        map_id_addr = self._addr.get("map_id")
        if map_id_addr:
            try:
                state["map_id"] = await bridge.read_u16(map_id_addr)
            except Exception:
                pass

        return state

    def summary(self, state: dict[str, Any]) -> str:
        """Format a game state dict into a compact text summary for the VLM prompt."""
        lines = []

        # Battle status
        if state.get("in_battle"):
            lines.append("STATUS: IN BATTLE")
        elif state.get("in_battle") is False:
            lines.append("STATUS: Overworld")

        # Party
        party = state.get("party", [])
        if party:
            parts = []
            for p in party:
                status_tag = f" [{p['status']}]" if p["status"] != "healthy" else ""
                parts.append(f"{p['nickname']} Lv{p['level']} {p['hp']}/{p['max_hp']}HP{status_tag}")
            lines.append("PARTY: " + " | ".join(parts))

        # Enemies (battle only)
        enemies = state.get("enemies", [])
        if enemies:
            parts = []
            for e in enemies:
                parts.append(f"{e['nickname']} Lv{e['level']} {e['hp']}/{e['max_hp']}HP")
            lines.append("ENEMY: " + " | ".join(parts))

        # Badges, money, map
        if "badges" in state:
            lines.append(f"BADGES: {state['badges']}/8")
        if "money" in state:
            lines.append(f"MONEY: ¥{state['money']:,}")
        if "map_id" in state:
            lines.append(f"MAP_ID: {state['map_id']}")

        return "\n".join(lines)
