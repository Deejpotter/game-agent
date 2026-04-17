---
description: "Use when editing agent.py — the main game loop, VLM integration, MCP tool wrappers, session management, or CLI args. Covers IPC flow, Windows-specific gotchas, and extension patterns."
applyTo: "agent.py"
---

# Agent Loop Guidelines

## IPC Flow (must understand before editing session code)

1. `agent.py` spawns `uvx mgba-live-mcp` as a **stdio MCP server**
2. Agent generates `mgba_launcher.lua` with the session dir baked in and `os.getenv` patched
3. User loads the ROM in mGBA (`File → Load ROM`), then loads `mgba_launcher.lua` via `Tools → Scripting → File → Load script → Run`
4. Bridge registers a frame callback; writes `heartbeat.json` every 30 frames once the ROM is running
5. Agent detects `heartbeat.json` and enters the game loop
6. Each turn: `mgba_live_export_screenshot` → VLM → `mgba_live_input_tap` (returns inline screenshot — no separate screenshot call needed)

**Why `mgba_launcher.lua` instead of `--script`?** mGBA's `--script` CLI flag does not execute Lua on the Windows Qt build (confirmed on 0.10.5). The launcher bakes in the session directory path and patches `os.getenv` so `mgba_live_bridge.lua` can find its IPC files without needing shell environment variables set inside the mGBA process.

**Why poll for `heartbeat.json`?** Frame callbacks only fire when a ROM is actively running. The heartbeat file appearing means both conditions are satisfied: the Lua script ran _and_ the ROM is loaded and emulating frames.

## VLM Response Contract

`decide()` always returns `(button: str, reason: str)`. On JSON parse failure it returns `("A", "(parse error — defaulted to A) ...")`. Never propagate exceptions from the VLM — log and continue.

History is capped at 6 messages (`history[-6:]`) before sending to the model. Do not increase this without measuring token cost.

## Adding a New MCP Tool Call

Follow the pattern in `capture_screenshot()` and `press_button()`:

- Accept `mcp_session: ClientSession` and `session_id: str | None`
- Always include `"session": session_id` in args when `session_id` is set
- Use `extract_image()` for base64 image responses, `extract_text()` for text
- Retry transient failures with `asyncio.sleep` between attempts

## Key Constants (top of file)

| Constant                 | Default | Effect                                                    |
| ------------------------ | ------- | --------------------------------------------------------- |
| `SETTLE_FRAMES`          | 8       | Frames mGBA emulates between button press and screenshot  |
| `SCREENSHOT_SCALE`       | 2       | Upscale multiplier for VLM (GBA native 240×160 → 480×320) |
| `AUTOSAVE_EVERY_N_TURNS` | 60      | How often the save_sequence fires                         |

## Adding a New Backend

Add a key to `BACKENDS` dict with `base_url`, `api_key`, and `model`. The OpenAI client is backend-agnostic. Then add the key to `--backend choices`.

## Do Not Touch

- `mgba_live_bridge.lua` — owned by the `mgba-live-mcp` package; re-copy with `uvx --with mgba-live-mcp python -c "import mgba_live_mcp; print(mgba_live_mcp.__file__)"`
- `mgba_launcher.lua` — auto-generated at runtime; changes are overwritten on next run
