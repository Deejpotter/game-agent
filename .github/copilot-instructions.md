# Game Agent — Copilot Instructions

## Project Overview

There are **two independent agents** in this repo. Both play games 24/7 via a local VLM.

| Agent                  | Emulator                | Transport                 | Console  |
| ---------------------- | ----------------------- | ------------------------- | -------- |
| `pyboy_agent/` package | PyBoy (in-process)      | None — direct Python API  | GBC / GB |
| `pyboy_agent.py`       | PyBoy (in-process)      | None — direct Python API  | GBC / GB |
| `agent.py`             | mGBA (external process) | stdio MCP → Lua IPC files | GBA      |

**`pyboy_agent/` is the canonical package** — use `python -m pyboy_agent` for all new GBC work. `pyboy_agent.py` is the original monolith and still works but is superseded. `agent.py` handles GBA via mGBA; do not break it.

## Architecture

### `pyboy_agent/` package (primary)

```
pyboy_agent/
  config.py          ← BACKENDS dict, BUTTON_MAP, all timing constants
  backends.py        ← make_client() factory (LM Studio / Ollama / OpenAI / Copilot)
  profiles.py        ← load_game_profile() — loads games/<name>.json
  emulator.py        ← create_pyboy(), capture_screenshot(), press_button(), walk_steps()
  loop.py            ← run_agent() — main turn orchestration
  main.py            ← argparse CLI  (--rom, --backend, --headless, --max-turns, --game)
  __main__.py        ← python -m pyboy_agent entry point
  ram/
    gen2_tables.py   ← _GEN2_CHAR / _GEN2_MOVE / _GEN2_TYPE lookup tables, read_bcd()
    reader.py        ← read_ram_state(pyboy, ram_offsets) → dict
    formatter.py     ← format_ram_state(state) → str  (prompt block)
  vision/
    perceive.py      ← perceive() — vision model call → JSON scene string
  llm/
    retry.py         ← with_retry(), extract_json()
    decide.py        ← decide() → 8-tuple (button, repeat, reason, event, new_goal,
                                            map_update, new_memory, thinking)
  navigation/
    world_map.py     ← WorldMap class + best_location_key()
    hints.py         ← build_nav_hints() — 12 prioritised hint sources → str
    wall_tracker.py  ← detect_and_record_wall() — RAM delta primary, hash fallback
  goals/
    phase_guide.py   ← BADGE_PHASE_MAP dict (badge count → next objective text)
    tracker.py       ← NotesTracker — story_log / goal_log / memory, persists to JSON
```

**Two-stage turn loop** (`loop.py`):

1. RAM read → `read_ram_state()` — position, HP, badges, battle flag
2. Build `nav_hint` — `build_nav_hints()` (wall, stuck, operator override, badge phase…)
3. `perceive()` — vision model → JSON scene (skipped by RAM fast-paths when unneeded)
4. `decide()` — reasoning model → 8-tuple action
5. `press_button()` / `walk_steps()` — execute
6. `detect_and_record_wall()` — compare pre/post RAM position
7. `NotesTracker.flush()` + PyBoy state snapshot

**Wall keys:** `map_{bank}_{number}_x{cx}_y{cy}` — per-tile, from RAM. Never use VLM location names as wall keys.

**Windowed mode:** `pump_fn = lambda: pyboy.tick(1, render=True)` is passed to `with_retry()` so the SDL2 window stays responsive during VLM calls. Never call `pyboy.tick()` from a worker thread.

### `agent.py` (mGBA / GBA)

```
agent.py ──[stdio]--> uvx mgba-live-mcp ──[file IPC]--> mgba_live_bridge.lua ──> mGBA
```

mGBA's `--script` CLI flag doesn't work on Windows Qt builds. Instead the agent generates `mgba_launcher.lua` (bakes in session dir + patches `os.getenv`) which the user loads via `Tools → Scripting`. The agent polls for `heartbeat.json` before entering the loop.

## Tech Stack

- **Python 3.11+** · `uv` for packages · `uv pip install -r requirements.txt`
- **PyBoy** — GBC emulator, in-process synchronous API
- **OpenAI SDK** (`openai>=1.30`) — backend-agnostic; works with LM Studio / Ollama / OpenAI / Copilot
- **Pillow** — screenshot RGBA→RGB + 2× nearest-neighbour upscale before VLM
- **python-dotenv** — `.env` for secrets / paths

## Key Workflows

**Run the package (preferred):**

```bash
python -m pyboy_agent --rom "H:/Games/GBC/Pokemon Silver.gbc"
python -m pyboy_agent --headless --max-turns 5       # quick test
python -m pyboy_agent --backend ollama               # switch backend
```

**Run the legacy monolith (still works):**

```bash
python pyboy_agent.py                                # uses ROM_PATH from .env
python pyboy_agent.py --headless --max-turns 5
```

**Crash recovery** — state snapshot auto-loads from `<rom>.pyboy_agent.state` next to the ROM on both entry points.

**Operator override** — type into the terminal while running, or drop text into `./agent_message.txt`. Injected as highest-priority nav hint next turn.

## Environment (`.env`)

```
ROM_PATH=H:/Games/GBC/ROMs/0/Pokemon Silver.gbc
LMS_MODEL=google/gemma-4-e4b
LMS_REASON_MODEL=google/gemma-4-e4b
MGBA_PATH=C:\Program Files\mGBA\mGBA.exe   # only needed for agent.py
```

## Game Profiles (`games/<name>.json`)

- `system_prompt` — given to the **reasoning model** only; never sees raw screenshots
- `save_sequence` — button list sent every `AUTOSAVE_EVERY_N_TURNS` (60) turns
- `ram_offsets` — hex address strings; always include a `"note"` metadata key
- `initial_goal` — one-sentence overarching objective (loaded into `NotesTracker`)

GBC: no `L`/`R`. HP = big-endian 16-bit. Money = 3-byte BCD. Open with `encoding="utf-8"` (contains `¥`).

## Conventions

- **`decide()` returns an 8-tuple:** `(button, repeat, reason, event, new_goal, map_update, new_memory, thinking)` — the 8th element `thinking` is the model's chain-of-thought (logged, not acted on).
- **`perceive()` returns a JSON string.** Parse failures are logged; the loop continues with `{}`.
- **`has_ram`**: `bool({k for k in ram_offsets if k != "note"})` — always gate RAM reads behind this check.
- **`enable_thinking: True`** injected via `extra_body` for any `http://localhost` backend automatically.
- **History** capped at `MAX_HISTORY_MESSAGES` (10). `story_log` injects the last 15 entries per turn.
- **`passable_directions`** from VLM is stripped before `decide()` when RAM position is available — RAM delta is authoritative; VLM values are frequently wrong.
- **HP guard:** `hp_max == 0` means RAM not yet initialised (2-turn warmup after state load). Never compute percentages or show LOW HP when `hp_max == 0`.

## Key Files

| File / Directory                                     | Purpose                                                  |
| ---------------------------------------------------- | -------------------------------------------------------- |
| `pyboy_agent/loop.py`                                | Main turn orchestration (`run_agent()`)                  |
| `pyboy_agent/navigation/hints.py`                    | All 12 nav hint sources — edit here for new hint logic   |
| `pyboy_agent/navigation/wall_tracker.py`             | Wall detection — RAM delta primary, hash fallback        |
| `pyboy_agent/ram/reader.py`                          | All WRAM reads — add new RAM keys here                   |
| `pyboy_agent/goals/tracker.py`                       | `NotesTracker` — story/goal/memory persistence           |
| `pyboy_agent.py`                                     | Legacy monolith — do not break; shares `games/` profiles |
| `agent.py`                                           | mGBA/GBA agent — do not break; shares `games/` profiles  |
| `games/pokemon-silver.json`                          | Reference GBC profile with RAM offsets                   |
| `games/pokemon-sapphire.json`                        | Reference GBA profile                                    |
| `.github/instructions/agent-loop.instructions.md`    | Editing guide for the agent loop and package modules     |
| `.github/instructions/game-profiles.instructions.md` | `games/*.json` authoring guide                           |
| `.github/instructions/pyboy-package.instructions.md` | Package module responsibilities and inter-module rules   |
| `.github/todos.md`                                   | Backlog and completed items                              |
