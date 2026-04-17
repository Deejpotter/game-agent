# Game Agent — Copilot Instructions

## Project Overview

There are **two independent agents** in this repo. Both play games 24/7 via a local VLM. Prefer `pyboy_agent.py` for new GBC work — it is simpler, actively maintained, and has richer observability.

| Agent            | Emulator                | Transport                 | Console  |
| ---------------- | ----------------------- | ------------------------- | -------- |
| `pyboy_agent.py` | PyBoy (in-process)      | None — direct Python API  | GBC / GB |
| `agent.py`       | mGBA (external process) | stdio MCP → Lua IPC files | GBA      |

## Architecture

### `pyboy_agent.py` (primary — use this one)

```
pyboy_agent.py
  ├── PyBoy(rom)              ← in-process GBC emulator, synchronous
  ├── OpenAI client ×2        ← vision model (perceive) + reasoning model (decide)
  ├── WorldMap                ← persistent cross-session location/NPC/wall tracker
  ├── notes.json              ← story_log, goal_log, memory (saved next to ROM)
  └── <rom>.pyboy_agent.state ← binary emulator snapshot for crash recovery
```

**Two-stage turn loop:** `perceive()` (vision-only → JSON scene description) → `decide()` (reasoning → button + memory update). Each stage uses a separate model call so the reasoning model never sees the raw image.

**Wall detection:** Uses RAM position delta (`x_pos`/`y_pos`) as the primary check — screenshot hash comparison is unreliable in small indoor rooms where the camera doesn't scroll. Falls back to hash only when `has_ram` is False.

**Wall keys:** Stored under `map_{bank}_{number}` (from RAM), not VLM location names, so they persist correctly across sessions and don't bleed between similarly-named rooms.

**Windowed mode:** VLM calls run in a `ThreadPoolExecutor` thread while `pump_fn = pyboy.tick(1, render=True)` runs on the main thread at ~60 fps so the SDL2 window stays responsive.

### `agent.py` (mGBA / GBA)

```
agent.py ──[stdio]--> uvx mgba-live-mcp ──[file IPC]--> mgba_live_bridge.lua ──> mGBA
```

mGBA's `--script` CLI flag doesn't work on Windows Qt builds. Instead the agent generates `mgba_launcher.lua` (bakes in session dir + patches `os.getenv`) which the user loads via `Tools → Scripting`. The agent polls for `heartbeat.json` before entering the loop.

## Tech Stack

- **Python 3.11+** · `uv` for packages · `uv pip install -r requirements.txt`
- **PyBoy** — GBC emulator, in-process synchronous API
- **OpenAI SDK** (`openai>=1.30`) — backend-agnostic, works with LM Studio / Ollama / OpenAI
- **Pillow** — screenshot RGBA→RGB + 2× nearest-neighbour upscale before VLM
- **python-dotenv** — `.env` for secrets / paths

## Key Workflows

**Run Pokemon Silver (windowed, LM Studio):**

```bash
python pyboy_agent.py
# ROM_PATH must be set in .env, or pass --rom "path/to/rom.gbc"
```

**Run headless (no window, max speed) for quick testing:**

```bash
python pyboy_agent.py --headless --max-turns 5
```

**Resume after crash** (state snapshot auto-loads from `<rom>.pyboy_agent.state`):

```bash
python pyboy_agent.py  # state auto-resumes if snapshot exists next to ROM
```

**Switch backend:**

```bash
python pyboy_agent.py --backend ollama   # port 11434
python pyboy_agent.py --backend openai   # needs OPENAI_API_KEY in .env
```

## Environment (`.env`)

```
ROM_PATH=H:/Games/GBC/ROMs/0/Pokemon Silver.gbc
LMS_MODEL=google/gemma-4-e4b
LMS_REASON_MODEL=google/gemma-4-e4b
MGBA_PATH=C:\Program Files\mGBA\mGBA.exe   # only needed for agent.py
```

## Game Profiles (`games/<name>.json`)

- `system_prompt` — game-specific VLM instructions; must end with the JSON-only reply format
- `save_sequence` — button list sent every 60 turns for in-game save
- `ram_offsets` — hex address strings keyed by semantic name (see `games/pokemon-silver.json`)

GBC profiles: no `L`/`R` buttons. Values read via `pyboy.memory[addr]`. HP is big-endian 16-bit. Money is 3-byte BCD.

## Conventions

- **perceive()** returns a JSON string. Parse failures are logged with raw output; the loop continues.
- **decide()** returns `(button, repeat, reason, event, new_goal, map_update, new_memory)`. JSON parse errors default to `"A"`.
- **Thinking**: `{"enable_thinking": True}` is passed as `extra_body` to localhost backends automatically.
- **History** is capped at 10 messages. `story_log` injects the last 15 entries per turn.
- **`has_ram`**: `bool({k for k in ram_offsets if k != "note"})` — always check before reading memory.
- Open files under `games/` with `encoding="utf-8"` (contains `¥` character).

## Key Files

| File                                                 | Purpose                                                    |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| `pyboy_agent.py`                                     | Primary agent — entire loop, VLM, WorldMap, RAM reader     |
| `agent.py`                                           | Legacy mGBA agent — do not break; shares `games/` profiles |
| `games/pokemon-silver.json`                          | Reference GBC profile with RAM offsets                     |
| `games/pokemon-sapphire.json`                        | Reference GBA profile                                      |
| `.github/instructions/agent-loop.instructions.md`    | `pyboy_agent.py` editing guide                             |
| `.github/instructions/game-profiles.instructions.md` | `games/*.json` authoring guide                             |
| `.github/todos.md`                                   | Backlog and completed items                                |
