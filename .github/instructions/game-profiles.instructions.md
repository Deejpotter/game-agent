---
description: "Use when creating or editing game profiles in games/*.json. Covers required fields, system prompt format, save sequences, and RAM offsets."
applyTo: "games/**/*.json"
---

# Game Profile Guidelines

Game profiles live at `games/<name>.json` where `<name>` matches the `--game` CLI arg (e.g. `--game pokemon-silver`).

## Required Fields

All fields must be present. Use `"console": "gbc"` for Game Boy Color, `"console": "gba"` for GBA.

```json
{
  "name": "Human-readable game name",
  "console": "gbc",
  "initial_goal": "One sentence — the overarching win condition.",
  "system_prompt": "...",
  "save_sequence": ["Start", "Down", "A"],
  "ram_offsets": {
    "note": "All offsets assume <Game> (<Region>). Read via pyboy.memory[addr]."
  }
}
```

## `system_prompt` Rules

The system prompt is given to the **reasoning model** (`decide()`), not the vision model. It never sees the raw screenshot — only the JSON scene description from `perceive()`.

**Required sections (in order):**
1. Role statement + available buttons
2. Reply format — ALWAYS include all eight fields: `thinking`, `button`, `repeat`, `reason`, `event`, `goal`, `memory`, `map_update`
3. Decision priority ladder (dialogue → menu → battle → NPC → overworld)
4. Navigation rules (wall avoidance, exit mats, re-talk guard)
5. Exploration strategy
6. Game-specific knowledge (story path, gym order, type chart, heal threshold)

**GBC profiles:** Available buttons are `A`, `B`, `Up`, `Down`, `Left`, `Right`, `Start`, `Select` — **no L or R**.

**GBA profiles:** Also include `L` and `R`.

Always open profile files with `encoding="utf-8"` — they may contain the `¥` character (money symbol).

See `games/pokemon-silver.json` for the canonical GBC example. See `games/pokemon-sapphire.json` for the GBA example.

## `save_sequence`

Button list sent every 60 turns (`AUTOSAVE_EVERY_N_TURNS`). Must navigate from the overworld to the in-game save confirmation and back.

- **Pokemon Silver:** `["Start","Down","Down","Down","Down","Down","A"]` (5 Downs to reach SAVE after Pokégear is obtained)
- **Pokemon Sapphire:** `["Start","Down","Down","Down","A","A"]`

## `ram_offsets` — GBC (pyboy_agent.py)

Values are hex address strings. Addresses are read via `pyboy.memory[addr]` (single byte) or multi-byte reads in `read_ram_state()`. Always include a `"note"` key stating the ROM region.

**Standard GBC keys** (from `games/pokemon-silver.json`):

| Key | Type | Notes |
|---|---|---|
| `player_name_start` | hex addr | First byte of player name; `player_name_length` gives count |
| `map_bank` | hex addr | Current map bank (1 byte) |
| `map_number` | hex addr | Current map number (1 byte) |
| `x_pos` | hex addr | Player X tile position |
| `y_pos` | hex addr | Player Y tile position |
| `money` | hex addr | BCD-encoded, `money_length` bytes (default 3) |
| `johto_badges_bitmask` | hex addr | 8-bit bitmask; bit 0 = Falkner … bit 7 = Clair |
| `kanto_badges_bitmask` | hex addr | 8-bit bitmask for Kanto badges |
| `party_count` | hex addr | Number of Pokémon in party |
| `party_slot0_hp_current` | hex addr | Big-endian 16-bit — read 2 bytes |
| `party_slot0_hp_max` | hex addr | Big-endian 16-bit — read 2 bytes |
| `party_slot0_level` | hex addr | 1 byte |

**HP is big-endian 16-bit:** `hp = (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]`

**Guard against `hp_max == 0`** — RAM returns zero until the game fully initialises after a state load. Don't compute percentages or show LOW HP warnings when `hp_max == 0`.

## `ram_offsets` — GBA (agent.py)

Values are hex address strings. Addresses are read via the `mgba_live_read_memory` MCP tool. All offsets assume the US/UE ROM unless noted.
