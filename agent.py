"""
Pokemon Sapphire autonomous agent.

Drives mGBA via the mgba-live-mcp REST/stdio bridge using its Python HTTP
companion.  Because mgba-live-mcp exposes a *stdio MCP server* (not HTTP),
this script drives it directly via subprocess + the MCP Python SDK, keeping
the loop entirely in Python so it runs 24/7 without VS Code.

Requirements (install with: uv pip install -r requirements.txt):
  mcp>=1.6
  openai>=1.30        # used for both LM Studio and OpenAI-compatible APIs
  pillow>=10.0        # screenshot resizing before sending to VLM
  python-dotenv>=1.0

Usage:
  # LM Studio (local model, no API key needed):
  python agent.py --rom "C:/path/to/Pokemon Sapphire.gba" --backend lmstudio

  # OpenAI-compatible (Ollama on port 11434):
  python agent.py --rom "C:/path/to/Pokemon Sapphire.gba" --backend ollama

  # GitHub Copilot / OpenAI (cloud, needs OPENAI_API_KEY in .env):
  python agent.py --rom "C:/path/to/Pokemon Sapphire.gba" --backend openai

  # Resume an existing mGBA session:
  python agent.py --rom "C:/path/to/Pokemon Sapphire.gba" --session 20260417-120000

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

BACKENDS: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        # LM Studio loads whichever model is active in its UI.
        # Gemma 4 E4B is the default: vision + reasoning, 6 GB, full GPU offload on 8GB cards.
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

# How many frames to wait after a button press before taking the next screenshot.
SETTLE_FRAMES = 8

# Resize screenshots sent to the VLM to reduce token usage (GBA is 240x160).
# 2x gives a crisp 480x320 image; still very readable.
SCREENSHOT_SCALE = 2

# Maximum turns before the agent saves the game. Safety net.
AUTOSAVE_EVERY_N_TURNS = 60

# System prompt injected into every VLM call.
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Pokemon player controlling Pokemon Sapphire running in
    the mGBA emulator. Study the screenshot carefully and decide the single
    best button to press next.

    Available buttons: A, B, Up, Down, Left, Right, Start, Select, L, R

    Reply with a JSON object ONLY — no markdown, no explanation — in this exact shape:
    {
      "button": "<one button string>",
      "reason": "<one sentence explaining why>"
    }

    Game knowledge:
    - Gym order: Roxanne > Brawly > Wattson > Flannery > Norman > Winona > Tate&Liza > Juan
    - Story path: Littleroot → Petalburg → Rustboro → Dewford → Mauville → Lavaridge → Fortree → Lilycove → Mossdeep → Sootopolis → Elite Four
    - Heal when HP < 25%. Save every 15 minutes. Always dismiss dialogue with A.
    - In battles: use super-effective moves, switch if badly disadvantaged.
    - If stuck on same screen: press B then A.
""").strip()


# ---------------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------------

def encode_screenshot(raw_png: bytes, scale: int = SCREENSHOT_SCALE) -> str:
    """Return a base64-encoded PNG string, optionally scaled up (nearest neighbour)."""
    img = Image.open(io.BytesIO(raw_png))
    if scale != 1:
        new_size = (img.width * scale, img.height * scale)
        img = img.resize(new_size, Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# VLM decision
# ---------------------------------------------------------------------------

def decide(client: OpenAI, model: str, screenshot_b64: str, history: list[dict]) -> tuple[str, str]:
    """Ask the VLM what button to press next. Returns (button, reason)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-6:],  # short rolling context window
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
    # Strip markdown code fences if the model adds them despite instructions.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
        button = parsed.get("button", "A").strip()
        reason = parsed.get("reason", "")
    except json.JSONDecodeError:
        # Fallback: just press A if JSON is malformed.
        button, reason = "A", f"(parse error, defaulted) raw={raw[:80]}"

    return button, reason


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------

async def mcp_call(session: ClientSession, tool: str, args: dict) -> Any:
    result = await session.call_tool(tool, args)
    # MCP returns a list of content items; extract the first text or image.
    if result.content:
        item = result.content[0]
        if hasattr(item, "text"):
            return item.text
        if hasattr(item, "data"):
            return item.data  # base64 image data for image content items
    return None


async def take_screenshot(session: ClientSession, tmp_path: Path) -> bytes:
    """Export screenshot to a temp file and return raw bytes."""
    png_path = str(tmp_path / "frame.png")
    await mcp_call(session, "mgba_live_export_screenshot", {
        "out": png_path,
    })
    return Path(png_path).read_bytes()


async def press_button(session: ClientSession, button: str) -> None:
    await mcp_call(session, "mgba_live_input_tap", {
        "key": button,
        "frames": 2,
        "wait_frames": SETTLE_FRAMES,
    })


async def save_game(session: ClientSession) -> None:
    """Open menu and navigate to Save."""
    print("  [autosave] opening menu…")
    for key in ["Start", "Down", "Down", "Down", "A"]:  # Start → Save menu item
        await mcp_call(session, "mgba_live_input_tap", {"key": key, "frames": 2, "wait_frames": 4})
        await asyncio.sleep(0.2)
    # Confirm overwrite (Yes)
    await mcp_call(session, "mgba_live_input_tap", {"key": "A", "frames": 2, "wait_frames": 8})
    print("  [autosave] saved.")


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

async def run_agent(
    rom: str,
    session_id: str | None,
    backend_cfg: dict,
    *,
    max_turns: int = 0,
) -> None:
    tmp_dir = Path(os.getenv("TEMP", "/tmp")) / "pokemon_agent"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    vlm = OpenAI(base_url=backend_cfg["base_url"], api_key=backend_cfg["api_key"])
    model = backend_cfg["model"]

    server_params = StdioServerParameters(
        command="uvx",
        args=["mgba-live-mcp"],
        env=None,
    )

    print(f"[agent] Starting — model={model}, rom={rom}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()

            # Start or attach to mGBA session
            if session_id:
                print(f"[agent] Attaching to session {session_id}…")
                result = await mcp_call(mcp_session, "mgba_live_attach", {"session": session_id})
                print(f"[agent] Attached: {str(result)[:120]}")
            else:
                print("[agent] Starting new mGBA session…")
                result = await mcp_call(mcp_session, "mgba_live_start", {
                    "rom": rom,
                    "fast": False,
                })
                if result:
                    try:
                        info = json.loads(result) if isinstance(result, str) else result
                        session_id = info.get("session") or info.get("session_id")
                        print(f"[agent] Session started: {session_id}")
                    except (json.JSONDecodeError, AttributeError):
                        print(f"[agent] Session info: {str(result)[:120]}")

            history: list[dict] = []
            turn = 0
            start_time = time.time()

            while True:
                turn += 1
                elapsed = int(time.time() - start_time)
                print(f"\n[turn {turn:04d} | {elapsed//60:02d}:{elapsed%60:02d}] Taking screenshot…")

                # Autosave periodically
                if turn > 1 and turn % AUTOSAVE_EVERY_N_TURNS == 0:
                    await save_game(mcp_session)

                # Capture & encode screenshot
                raw_png = await take_screenshot(mcp_session, tmp_dir)
                b64 = encode_screenshot(raw_png)

                # Ask VLM
                button, reason = decide(vlm, model, b64, history)
                print(f"[turn {turn:04d}] → {button:6s} | {reason}")

                # Add assistant decision to rolling history (text only, no images)
                history.append({
                    "role": "assistant",
                    "content": json.dumps({"button": button, "reason": reason}),
                })

                # Execute button press
                await press_button(mcp_session, button)

                if max_turns and turn >= max_turns:
                    print(f"[agent] Reached max_turns={max_turns}, stopping.")
                    break

                await asyncio.sleep(0.05)  # brief yield; real pacing from wait_frames


def _handle_sigint(signum, frame):
    print("\n[agent] Ctrl+C received — shutting down gracefully…")
    sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Pokemon Sapphire autonomous agent")
    parser.add_argument(
        "--rom",
        required=True,
        help="Absolute path to Pokemon Sapphire .gba ROM",
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
        help="LLM backend to use (default: lmstudio)",
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
    args = parser.parse_args()

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
        max_turns=args.max_turns,
    ))


if __name__ == "__main__":
    main()
