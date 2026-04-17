# Game Agent — Copilot Instructions

## Project Overview

Autonomous AI agent that plays GBA games 24/7 by driving **mGBA** via `mgba-live-mcp` (a stdio MCP server). The loop is: screenshot → VLM decision → button press, running entirely in Python without VS Code.

## Tech Stack

- **Python 3.11+** with `uv` for package management
- **MCP Python SDK** (`mcp>=1.6`) for `mgba-live-mcp` stdio transport
- **OpenAI-compatible API** (`openai>=1.30`) for VLM decisions — works with LM Studio, Ollama, or OpenAI
- **Pillow** for screenshot scaling (GBA native 240×160 → 2× upscale before sending to VLM)
- **mGBA 0.10.x** (Windows Qt build) + `mgba_live_bridge.lua` for file-based IPC
- **Lua bridge IPC**: file-based via `~/.mgba-live-mcp/runtime/<session_id>/` — `command.lua`, `response.json`, `heartbeat.json`

## Architecture & Data Flow

```
agent.py ──[stdio]--> mgba-live-mcp (uvx) ──[file IPC]--> mgba_live_bridge.lua ──[mGBA Scripting API]--> mGBA
```

1. `agent.py` spawns `uvx mgba-live-mcp` as a stdio MCP server
2. On new session, generates `mgba_launcher.lua` (hardcodes session dir, patches `os.getenv`, calls `dofile()` on `mgba_live_bridge.lua`)
3. User loads the ROM in mGBA (`File → Load ROM`), then loads `mgba_launcher.lua` via `Tools → Scripting → File → Load script → Run`
4. Bridge registers a frame callback; writes `heartbeat.json` every 30 frames once the ROM is running
5. Agent detects `heartbeat.json` and enters the game loop
6. Each turn: `mgba_live_export_screenshot` → VLM → `mgba_live_input_tap` (returns inline screenshot, no extra round-trip)

**Why `mgba_launcher.lua` instead of `--script`?** mGBA's `--script` CLI flag does not execute Lua on the Windows Qt build. The launcher is a generated wrapper that bakes in the session directory path and patches `os.getenv` so `mgba_live_bridge.lua` knows where to write IPC files — without needing shell environment variables set in the mGBA process.

**Why poll for `heartbeat.json`?** Frame callbacks only fire when a ROM is actively running. `heartbeat.json` appearing means both conditions are satisfied: the script ran _and_ the ROM is loaded.

## Game Profiles (`games/<name>.json`)

Each profile contains:

- `system_prompt` — full game-specific instructions for the VLM
- `save_sequence` — button list for in-game save (triggered every 60 turns)
- `ram_offsets` — named memory addresses for reading game state

Add new games by creating `games/<name>.json`. See `games/pokemon-sapphire.json` as the reference. Pass `--game <name>` to use it.

## Key Workflows

**Install:**

```bash
uv pip install -r requirements.txt
```

**Run (Pokemon Sapphire example):**

```bash
python agent.py --rom "H:/Games/GBA/ROMs/8/Pokemon Sapphire.gba" --game pokemon-sapphire
```

The agent prints step-by-step mGBA instructions. Follow them: load ROM → open Scripting window → load `mgba_launcher.lua` → click Run. The agent waits for `heartbeat.json` then starts automatically.

**Resume after a crash** (mGBA still running with the same session):

```bash
python agent.py --rom "..." --game pokemon-sapphire --session 20260417-102408
```

The session ID is printed when the agent first starts. `--session` skips the entire manual mGBA setup and re-attaches to the live bridge.

**Backends:** `--backend lmstudio` (default, port 1234) | `--backend ollama` (port 11434) | `--backend openai`

## Environment (`.env`)

```
MGBA_PATH=C:\Program Files\mGBA\mGBA.exe
LMS_MODEL=google/gemma-4-e4b
```

## Conventions

- VLM must return `{"button": "<name>", "reason": "<sentence>"}` — parse errors default to `"A"` with a logged message
- History is capped at last 6 messages to limit token usage
- `mgba_launcher.lua` is auto-generated each run — **do not edit by hand**
- `mgba_live_bridge.lua` is sourced from the `mgba-live-mcp` uv cache — do not modify; if re-copying, use: `uvx --with mgba-live-mcp python -c "import mgba_live_mcp; print(mgba_live_mcp.__file__)"`

## Key Files

| File                          | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| `agent.py`                    | Entire agent: CLI, VLM, MCP wrappers, game loop |
| `games/pokemon-sapphire.json` | Reference game profile                          |
| `mgba_live_bridge.lua`        | Lua IPC bridge (do not modify)                  |
| `mgba_launcher.lua`           | Auto-generated session launcher (do not commit) |
| `.env`                        | Local secrets and paths (not committed)         |
