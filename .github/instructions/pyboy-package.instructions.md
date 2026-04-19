---
description: "Use when working inside pyboy_agent/ package modules. Covers module responsibilities, inter-module rules, public APIs, and extension patterns for ram/, llm/, vision/, navigation/, and goals/ subpackages."
applyTo: "pyboy_agent/**"
---

# `pyboy_agent/` Package — Module Guide

The package is the canonical GBC agent. `pyboy_agent.py` (monolith) is the legacy entry point and must not be broken. Both share `games/*.json` profiles and `*.pyboy_agent.state` snapshots.

## Module Responsibilities (who owns what)

| Module                       | Owns                                         | Does NOT own       |
| ---------------------------- | -------------------------------------------- | ------------------ |
| `config.py`                  | All constants, `BACKENDS` dict, `BUTTON_MAP` | Any I/O or imports |
| `backends.py`                | `make_client()`, Copilot token refresh       | Game logic         |
| `profiles.py`                | Load and validate `games/*.json`             | RAM reads          |
| `emulator.py`                | PyBoy lifecycle, screenshots, button presses | VLM calls          |
| `ram/reader.py`              | All `pyboy.memory[]` reads                   | Formatting or VLM  |
| `ram/formatter.py`           | `format_ram_state()` → prompt string         | RAM reads          |
| `ram/gen2_tables.py`         | Lookup tables only                           | Any live reads     |
| `vision/perceive.py`         | Vision model call → JSON string              | Decision logic     |
| `llm/retry.py`               | `with_retry()`, `extract_json()`             | Domain logic       |
| `llm/decide.py`              | Reasoning model call → 8-tuple               | Emulator/RAM       |
| `navigation/world_map.py`    | `WorldMap` persistence                       | Hint assembly      |
| `navigation/hints.py`        | `build_nav_hints()` → str                    | WorldMap writes    |
| `navigation/wall_tracker.py` | `detect_and_record_wall()`                   | Hint assembly      |
| `goals/phase_guide.py`       | `BADGE_PHASE_MAP` lookup table               | Any I/O            |
| `goals/tracker.py`           | `NotesTracker` — story/goal/memory           | RAM or VLM         |
| `loop.py`                    | `run_agent()` orchestration                  | Domain logic       |
| `main.py`                    | CLI argparse                                 | Loop logic         |

## Public APIs

### `ram/`

```python
from pyboy_agent.ram import read_ram_state, format_ram_state
state = read_ram_state(pyboy, ram_offsets)   # → dict
text  = format_ram_state(state)              # → str prompt block
```

`state` keys: `map_bank`, `map_number`, `x_pos`, `y_pos`, `lead_hp_current`, `lead_hp_max`, `lead_hp_pct`, `johto_badge_count`, `kanto_badge_count`, `money`, `party_count`, `party_slots`, `in_battle`, `hp_stabilised`.

### `navigation/`

```python
from pyboy_agent.navigation import (
    WorldMap, best_location_key,
    detect_and_record_wall, build_nav_hints,
)
world_map = WorldMap("pokemon-silver")          # loads ~/.pyboy-agent/world_maps/...
nav_hint  = build_nav_hints(operator_msg=..., wall_detected=..., ...)
wall_detected, wall_button = detect_and_record_wall(pyboy, button=button, ...)
```

### `goals/`

```python
from pyboy_agent.goals import NotesTracker, BADGE_PHASE_MAP
notes = NotesTracker(notes_path, initial_goal="Beat the Elite Four")
notes.append_event("Defeated Falkner")   # → flush()
notes.update_goal("Beat Bugsy", turn=5)  # → flush()
notes.update_memory("Violet City explored")
notes.recent_story   # last 15 events
```

### `llm/`

```python
from pyboy_agent.llm.retry import with_retry, extract_json
from pyboy_agent.llm.decide import decide

result = with_retry(fn, pump_fn=pump_fn, on_auth_error=callback)
button, repeat, reason, event, new_goal, map_update, new_memory, thinking = decide(...)
```

## Inter-Module Rules

- `loop.py` imports from every subpackage — it is the only module allowed to do so.
- `vision/perceive.py` imports `extract_json` from `llm/retry` via a **local import** inside the function body (avoids circular import at module load time).
- `navigation/hints.py` and `navigation/wall_tracker.py` do NOT import each other.
- `ram/gen2_tables.py` has zero imports — it is a pure data module.
- `config.py` has zero imports — no circular dependency risk.

## Extending Navigation Hints

All 12 hint sources live in `navigation/hints.py::build_nav_hints()`. Add new sources here, in priority order:

```
1. Operator override (highest priority)
2. Wall detected
3. Consecutive-button warning
4. Consecutive-A warning
5. RAM tile hint (walled directions)
6. Party-empty critical
7. Blackout warning
8. Badge-phase story guide
9. Battle hint
10. Memory indoor/outdoor correction
11. Battle screen override
12. NPC retalk guard
```

## Adding a New RAM Key

1. Add the hex address to `games/<name>.json` under `ram_offsets`.
2. Read it in `ram/reader.py::read_ram_state()` — follow existing patterns.
3. Expose it in the returned `state` dict with a clear snake_case key.
4. Optionally surface it in `ram/formatter.py::format_ram_state()` if the model needs it.

## Do Not Touch

- `mgba_live_bridge.lua` — owned by `mgba-live-mcp` package
- `mgba_launcher.lua` — auto-generated at runtime; changes are overwritten on next run
- `pyboy_agent.py` — legacy monolith; do not remove features or break CLI compatibility
