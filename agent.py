"""
Autonomous game agent for mGBA-supported consoles.

Drives mGBA via the mgba-live-mcp stdio MCP server using the MCP Python SDK,
keeping the loop entirely in Python so it runs 24/7 without VS Code.

Game-specific knowledge is loaded from games/<name>.json profiles, making
this script reusable across any game mGBA can run.

────────────────────────────────────────────
Architecture overview
────────────────────────────────────────────
agent.py ─[stdio]─► mgba-live-mcp (uvx) ─[file IPC]─► mgba_live_bridge.lua ─► mGBA

  1. This script spawns `uvx mgba-live-mcp` as a child process and speaks to it
     over stdin/stdout using the MCP protocol.
  2. mgba-live-mcp communicates with mGBA by reading/writing files in a session
     directory under ~/.mgba-live-mcp/runtime/<session_id>/.
  3. The Lua bridge (mgba_live_bridge.lua) runs inside mGBA's scripting engine,
     polls that directory every frame, and writes heartbeat.json every 30 frames.

────────────────────────────────────────────
Startup workflow (one-time per mGBA session)
────────────────────────────────────────────
  1. Run agent.py — it generates mgba_launcher.lua and prints step-by-step
     instructions with the exact file path to load.
  2. In mGBA: File → Load ROM (or drag-drop). Wait for the title screen.
  3. In mGBA: Tools → Scripting → File → Load script → select mgba_launcher.lua.
  4. Click ▶ Run in the Scripting toolbar.
     A 1-second freeze is normal — it's `mkdir` creating the session directory.
  5. The bridge registers a frame callback and begins writing heartbeat.json
     every 30 frames. This script detects it and starts the game loop.

To resume after a crash or restart: pass --session <id> to skip steps 2-4
and attach directly to the still-running mGBA instance.

────────────────────────────────────────────
Why mgba_launcher.lua instead of --script?
────────────────────────────────────────────
mGBA's --script CLI flag does not execute Lua on the Windows Qt build.
The launcher is a thin wrapper that:
  • hardcodes the session directory path (so the bridge knows where to write files)
  • patches os.getenv at runtime (so the bridge doesn't need shell env vars)
  • calls dofile() on mgba_live_bridge.lua (the real bridge from mgba-live-mcp)

────────────────────────────────────────────
Requirements (install with: uv pip install -r requirements.txt)
────────────────────────────────────────────
  mcp>=1.6
  openai>=1.30        # OpenAI-compatible client — works with LM Studio, Ollama, OpenAI
  pillow>=10.0        # screenshot resizing before sending to VLM
  python-dotenv>=1.0

────────────────────────────────────────────
Usage
────────────────────────────────────────────
  # Pokemon Sapphire with a local vision model in LM Studio (default):
  python agent.py --rom "H:/Games/GBA/ROMs/Pokemon Sapphire.gba" --game pokemon-sapphire

  # Any GBA game without a profile (generic mode with fallback prompt):
  python agent.py --rom "C:/ROMs/game.gba"

  # Use Ollama instead of LM Studio:
  python agent.py --rom "..." --game pokemon-sapphire --backend ollama

  # Resume an existing session after a crash (skips the manual mGBA setup steps):
  python agent.py --rom "..." --game pokemon-sapphire --session 20260417-102408

Press Ctrl+C to stop the agent gracefully.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from PIL import Image
import io

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All three backends expose an OpenAI-compatible /v1/chat/completions endpoint,
# so we use the same openai client for all of them — only the base_url differs.
# Model names and API keys are read from .env so switching backends requires
# no code changes.
BACKENDS: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        # LM Studio serves whichever model is active in its UI.
        # Gemma 4 E4B: vision + reasoning + tool use, 6 GB, full GPU offload on 8 GB cards.
        "model": os.getenv("LMS_MODEL", "google/gemma-4-e4b"),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    },
}

# GBA runs at 59.7 fps. 8 frames (~134 ms) is enough for most in-game animations
# (menu transitions, text scroll) to settle before the next screenshot is taken.
SETTLE_FRAMES = 8

# GBA native resolution is 240×160. Doubling to 480×320 gives the VLM enough
# pixel detail to read small text (HP numbers, menu items) without bloating the
# base64 payload. NEAREST-neighbour is used so pixels stay crisp, not blurred.
SCREENSHOT_SCALE = 2

# Every N turns the agent executes the game-specific save_sequence from the
# game profile. 60 turns ≈ 2-5 minutes of play depending on turn speed.
AUTOSAVE_EVERY_N_TURNS = 60

# Fallback system prompt used when no game profile is loaded.
GENERIC_SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous game-playing AI controlling a game running in the
    mGBA emulator. Study each screenshot carefully and decide the single best
    button to press next.

    Available buttons: A, B, Up, Down, Left, Right, Start, Select, L, R

    Reply with a JSON object ONLY — no markdown, no explanation:
    {
      "button": "<one button string>",
      "reason": "<one sentence explaining why>"
    }

    General strategy:
    - Dismiss all dialogue and menus by pressing A (or B to cancel/back out).
    - Navigate toward whatever the current objective appears to be.
    - If the situation looks the same as the last 3 turns: press B, then try
      a directional button to get unstuck.
""").strip()


# ---------------------------------------------------------------------------
# Game profile loader
# ---------------------------------------------------------------------------

def load_game_profile(name: str | None) -> dict[str, Any]:
    """Load a game profile JSON from games/<name>.json, or return generic defaults."""
    if name is None:
        return {
            "name": "Generic",
            "system_prompt": GENERIC_SYSTEM_PROMPT,
            "save_sequence": None,
        }
    path = Path(__file__).parent / "games" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No game profile found at {path}. "
            f"Create games/{name}.json or omit --game for generic mode."
        )
    profile = json.loads(path.read_text(encoding="utf-8"))
    if "system_prompt" not in profile:
        profile["system_prompt"] = GENERIC_SYSTEM_PROMPT
    return profile


# ---------------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------------

def process_screenshot(b64_data: str, scale: int = SCREENSHOT_SCALE) -> str:
    """Decode base64 PNG, optionally scale it up, and return re-encoded base64."""
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_image(result: Any) -> str | None:
    """Pull the first base64 image from an MCP tool result, or None."""
    if result and result.content:
        for item in result.content:
            if hasattr(item, "data") and item.data:
                return item.data
    return None


def extract_text(result: Any) -> str | None:
    """Pull the first text item from an MCP tool result, or None."""
    if result and result.content:
        for item in result.content:
            if hasattr(item, "text") and item.text:
                return item.text
    return None


# ---------------------------------------------------------------------------
# VLM decision
# ---------------------------------------------------------------------------

def decide(
    client: OpenAI,
    model: str,
    screenshot_b64: str,
    history: list[dict],
    system_prompt: str,
) -> tuple[str, str]:
    """Ask the VLM what button to press next. Returns (button, reason)."""
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        # Rolling window of the last 6 turns — enough for the model to notice
        # repeated states (stuck-loop detection) without burning excessive tokens.
        *history[-6:],
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                },
                {"type": "text", "text": "What button should I press next?"},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    # Some models wrap JSON in ```json fences despite the system prompt telling
    # them not to. Strip fences before parsing.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
        button = str(parsed.get("button", "A")).strip()
        reason = str(parsed.get("reason", ""))
    except json.JSONDecodeError:
        # Never crash the game loop on a bad VLM response — default to A (safe
        # "confirm" action) and log the raw output for debugging.
        button, reason = "A", f"(parse error — defaulted to A) raw={raw[:80]}"

    return button, reason


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------

async def capture_screenshot(
    mcp_session: ClientSession, session_id: str | None, retries: int = 3
) -> str:
    """Call mgba_live_export_screenshot and return base64 PNG string."""
    args: dict[str, Any] = {}
    if session_id:
        args["session"] = session_id
    for attempt in range(1, retries + 1):
        result = await mcp_session.call_tool("mgba_live_export_screenshot", args)
        data = extract_image(result)
        if data is not None:
            return data
        if attempt < retries:
            print(f"  [screenshot] no image data (attempt {attempt}/{retries}), retrying…")
            await asyncio.sleep(1.0)
    raise RuntimeError(
        "mgba_live_export_screenshot returned no image data after all retries. "
        "Check that mGBA launched and the ROM loaded correctly."
    )


async def press_button(
    mcp_session: ClientSession,
    session_id: str | None,
    button: str,
    wait_frames: int = SETTLE_FRAMES,
) -> str | None:
    """Tap a button and return the inline screenshot captured after wait_frames."""
    # mgba_live_input_tap holds the button for `frames` emulated frames then
    # waits `wait_frames` more frames before taking a screenshot — all in one
    # MCP call. This avoids an extra round-trip for a separate screenshot call.
    args: dict[str, Any] = {"key": button, "frames": 2, "wait_frames": wait_frames}
    if session_id:
        args["session"] = session_id
    result = await mcp_session.call_tool("mgba_live_input_tap", args)
    return extract_image(result)  # None if the tool returned no image


async def save_game(
    mcp_session: ClientSession,
    session_id: str | None,
    save_sequence: list[str],
) -> str | None:
    """Execute the game-specific save button sequence. Returns the last screenshot."""
    print("  [autosave] running save sequence…")
    last_screenshot: str | None = None
    for key in save_sequence:
        last_screenshot = await press_button(mcp_session, session_id, key, wait_frames=4)
        await asyncio.sleep(0.15)
    print("  [autosave] done.")
    return last_screenshot


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

async def run_agent(
    rom: str,
    session_id: str | None,
    backend_cfg: dict,
    game_profile: dict,
    *,
    max_turns: int = 0,
    mgba_path: str | None = None,
) -> None:
    vlm = OpenAI(base_url=backend_cfg["base_url"], api_key=backend_cfg["api_key"])
    model = backend_cfg["model"]
    system_prompt: str = game_profile["system_prompt"]
    save_sequence: list[str] | None = game_profile.get("save_sequence")
    game_name: str = game_profile.get("name", "game")

    server_params = StdioServerParameters(
        command="uvx",
        args=["mgba-live-mcp"],
        env=None,
    )

    print(f"[agent] Game={game_name} | model={model} | rom={rom}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()

            # ── Session management ──────────────────────────────────────────
            if session_id:
                # Resume path: mGBA is still running with the bridge active.
                # mgba_live_attach tells the MCP server to reconnect to an
                # existing session directory — no Lua reload required.
                print(f"[agent] Attaching to session {session_id}…")
                result = await mcp_session.call_tool(
                    "mgba_live_attach", {"session": session_id}
                )
                print(f"[agent] Attached: {str(extract_text(result) or '(ok)')[:120]}")
            else:
                # New session path: generate a launcher Lua script and wait for
                # the user to load it in mGBA's Scripting window.
                #
                # Why generate a wrapper instead of loading the bridge directly?
                # mgba_live_bridge.lua reads its session directory from env vars
                # (MGBA_LIVE_SESSION_DIR etc.), but those vars aren't set in the
                # mGBA process. The launcher hardcodes the paths and patches
                # os.getenv() at runtime so the bridge picks them up when
                # dofile() loads it.
                import datetime
                session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                runtime_dir = Path.home() / ".mgba-live-mcp" / "runtime"
                session_dir = runtime_dir / session_id
                session_dir.mkdir(parents=True, exist_ok=True)

                bridge_path = Path(__file__).parent / "mgba_live_bridge.lua"
                launcher_path = Path(__file__).parent / "mgba_launcher.lua"

                # Write forward-slash paths — Lua on Windows is fine with them
                sdir_lua = session_dir.as_posix()
                bridge_lua = bridge_path.as_posix()

                launcher_src = f"""\
-- Auto-generated by agent.py — do not edit.
-- Load this file in mGBA: Tools > Scripting > File > Load script > Run

local session_dir = "{sdir_lua}"

-- Create the session directory if needed (safety net)
os.execute('mkdir "' .. session_dir:gsub("/", "\\\\") .. '"')

-- Patch os.getenv so the bridge picks up our hardcoded paths
local _orig_getenv = os.getenv
os.getenv = function(k)
  if k == "MGBA_LIVE_SESSION_DIR"        then return session_dir end
  if k == "MGBA_LIVE_COMMAND"            then return session_dir .. "/command.lua" end
  if k == "MGBA_LIVE_RESPONSE"           then return session_dir .. "/response.json" end
  if k == "MGBA_LIVE_HEARTBEAT"          then return session_dir .. "/heartbeat.json" end
  if k == "MGBA_LIVE_HEARTBEAT_INTERVAL" then return "30" end
  return _orig_getenv(k)
end

-- Load the real bridge script
dofile("{bridge_lua}")
"""
                launcher_path.write_text(launcher_src, encoding="utf-8")

                print(f"[agent] Session ID : {session_id}")
                print(f"[agent] Session dir: {session_dir}")
                print()
                print("[agent] ══ Steps to start ══════════════════════════════════════════")
                print(f"[agent]   STEP 1 — Load the ROM  ← do this FIRST, before scripting!")
                print(f"[agent]            mGBA: File → Load ROM (or drag-drop the file)")
                print(f"[agent]            {rom}")
                print(f"[agent]            Wait until the game title screen is visible.")
                print()
                print(f"[agent]   STEP 2 — Open the Scripting window")
                print(f"[agent]            mGBA: Tools → Scripting")
                print()
                print(f"[agent]   STEP 3 — Load the launcher script")
                print(f"[agent]            In the Scripting window: File → Load script")
                print(f"[agent]            {launcher_path}")
                print()
                print("[agent]   STEP 4 — Run it")
                print("[agent]            Click the  ▶ Run  button in the Scripting toolbar.")
                print("[agent]            A 1-second freeze is normal (directory is created).")
                print("[agent]            After that, the bridge will start on the next frame.")
                print("[agent] ═══════════════════════════════════════════════════════════════")
                print()

                # Wait for the bridge to confirm it's alive before starting.
                # heartbeat.json is written by the bridge's frame callback every
                # 30 frames (~0.5 s). It won't appear until both conditions are
                # true: (1) the Lua script has been loaded and Run clicked, AND
                # (2) a ROM is running so frame callbacks fire.
                heartbeat = session_dir / "heartbeat.json"
                poll = 0
                while not heartbeat.exists():
                    poll += 1
                    if poll % 5 == 0:
                        print(f"[agent] Waiting for mGBA bridge (poll {poll})…")
                    await asyncio.sleep(2.0)
                print(f"[agent] Bridge ready! Starting game loop…")

            # ── Initial screenshot ──────────────────────────────────────────
            await asyncio.sleep(1.0)
            current_b64 = await capture_screenshot(mcp_session, session_id, retries=6)

            history: list[dict] = []
            turn = 0
            start_time = time.time()

            while True:
                turn += 1
                elapsed = int(time.time() - start_time)
                print(f"\n[turn {turn:04d} | {elapsed//60:02d}:{elapsed%60:02d}]", end=" ")

                # ── Autosave ────────────────────────────────────────────────
                if save_sequence and turn > 1 and turn % AUTOSAVE_EVERY_N_TURNS == 0:
                    saved_screenshot = await save_game(mcp_session, session_id, save_sequence)
                    if saved_screenshot:
                        current_b64 = saved_screenshot
                    else:
                        current_b64 = await capture_screenshot(mcp_session, session_id)

                # ── VLM decision ────────────────────────────────────────────
                processed = process_screenshot(current_b64)
                button, reason = decide(vlm, model, processed, history, system_prompt)
                print(f"→ {button:6s} | {reason}")

                history.append({
                    "role": "assistant",
                    "content": json.dumps({"button": button, "reason": reason}),
                })

                # Press the button AND capture the result frame in one MCP call.
                # mgba_live_input_tap emulates N frames then returns a screenshot,
                # so we don't need a separate screenshot round-trip each turn.
                next_b64 = await press_button(mcp_session, session_id, button)
                if next_b64 is not None:
                    current_b64 = next_b64
                else:
                    # Fallback explicit screenshot — the tap tool occasionally
                    # returns no image if mGBA is mid-transition or paused.
                    current_b64 = await capture_screenshot(mcp_session, session_id)

                if max_turns and turn >= max_turns:
                    print(f"[agent] Reached max_turns={max_turns}, stopping.")
                    break

                await asyncio.sleep(0.05)  # brief yield; real pacing set by wait_frames


def _handle_sigint(signum: int, frame: Any) -> None:
    print("\n[agent] Ctrl+C received — shutting down gracefully…")
    sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(
        description="Autonomous game agent — drives mGBA via a local vision model"
    )
    parser.add_argument(
        "--rom",
        required=True,
        help="Absolute path to the ROM file (.gba / .gb / .gbc)",
    )
    parser.add_argument(
        "--game",
        default=None,
        metavar="NAME",
        help=(
            "Game profile name (loads games/<NAME>.json). "
            "Omit for generic mode. Example: pokemon-sapphire"
        ),
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Existing mgba-live-mcp session ID to attach to (skips start)",
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="lmstudio",
        help="LLM backend (default: lmstudio)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name for the chosen backend",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Stop after this many turns (0 = run forever)",
    )
    parser.add_argument(
        "--mgba-path",
        default=os.getenv("MGBA_PATH"),
        metavar="PATH",
        help=(
            "Absolute path to the mGBA executable. "
            "Defaults to MGBA_PATH env var. "
            "Required on Windows if mGBA is not on PATH. "
            "Example: \"C:/Program Files/mGBA/mGBA.exe\""
        ),
    )
    args = parser.parse_args()

    game_profile = load_game_profile(args.game)

    cfg = dict(BACKENDS[args.backend])
    if args.model:
        cfg["model"] = args.model

    if args.backend == "openai" and not cfg["api_key"]:
        print("ERROR: Set OPENAI_API_KEY in your environment or .env file.")
        sys.exit(1)

    asyncio.run(run_agent(
        rom=args.rom,
        session_id=args.session,
        backend_cfg=cfg,
        game_profile=game_profile,
        max_turns=args.max_turns,
        mgba_path=args.mgba_path,
    ))


if __name__ == "__main__":
    main()
