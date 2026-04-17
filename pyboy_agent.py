"""
Autonomous game agent for Game Boy Color games using PyBoy.

Drives PyBoy directly in-process — no MCP, no Lua bridge, no IPC files.
The emulator runs synchronously alongside the VLM decision loop.

Game-specific knowledge is loaded from games/<name>.json profiles.
Defaults to Pokemon Silver if --game is not specified.

────────────────────────────────────────────
Architecture overview
────────────────────────────────────────────
pyboy_agent.py
  └── PyBoy(rom)          ← in-process GBC emulator
  └── OpenAI client(s)    ← VLM (perceive) + reasoning (decide)
  └── WorldMap            ← persistent cross-session location/NPC tracker
  └── notes.json          ← story log, goal, memory (saved next to ROM)
  └── .pyboy_agent.state  ← binary emulator snapshot for crash recovery

────────────────────────────────────────────
Key differences from agent.py (mGBA)
────────────────────────────────────────────
  • No asyncio — everything runs synchronously
  • No MCP server, no Lua bridge, no IPC polling
  • PyBoy loads the ROM directly; .sav file auto-loads on startup
  • Buttons are lowercase for PyBoy API ('a', 'b', 'start', 'select', etc.)
  • GBC native resolution: 160×144 (upscaled 2× to 320×288 for VLM)
  • No L/R buttons on GBC — game profiles must not include them
  • Autosave writes both the in-game save and a PyBoy state snapshot

────────────────────────────────────────────
Usage
────────────────────────────────────────────
  # Pokemon Silver (default game profile):
  python pyboy_agent.py --rom "H:/Games/GBC/Pokemon Silver.gbc"

  # Explicit game profile:
  python pyboy_agent.py --rom "..." --game pokemon-silver

  # Headless (no window, max speed):
  python pyboy_agent.py --rom "..." --headless

  # Resume from a saved state file:
  python pyboy_agent.py --rom "..." --state "Pokemon Silver.gbc.pyboy_agent.state"

  # Use Ollama instead of LM Studio:
  python pyboy_agent.py --rom "..." --backend ollama

Press Ctrl+C to stop gracefully (saves state and notes before exit).
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import re
import concurrent.futures
import signal
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import openai
from openai import OpenAI
from PIL import Image
from pyboy import PyBoy  # type: ignore[import-untyped]

load_dotenv()

# ---------------------------------------------------------------------------
# GitHub Copilot backend helpers
# ---------------------------------------------------------------------------

_COPILOT_TOKEN_PATH = Path.home() / ".openclaw" / "credentials" / "github-copilot.token.json"

# Headers required by the GitHub Copilot API to identify the client.
_COPILOT_HEADERS: dict[str, str] = {
    "Editor-Version": "vscode/1.95.0",
    "Editor-Plugin-Version": "copilot-chat/0.22.4",
    "Copilot-Integration-Id": "vscode-chat",
    "Openai-Intent": "conversation-panel",
    "User-Agent": "GitHubCopilotGame/1.0",
}


def _load_copilot_token() -> str:
    """Read the current session token from the OpenClaw credentials file.

    OpenClaw refreshes this file automatically, so re-reading on 401 gives a
    fresh token without any user interaction.
    """
    if not _COPILOT_TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"GitHub Copilot token not found at {_COPILOT_TOKEN_PATH}. "
            "Ensure OpenClaw is running and you are signed in to GitHub Copilot."
        )
    data = json.loads(_COPILOT_TOKEN_PATH.read_text(encoding="utf-8"))
    return data["token"]


def _make_copilot_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at the GitHub Copilot API."""
    return OpenAI(
        base_url="https://api.githubcopilot.com",
        api_key=_load_copilot_token(),
        default_headers=_COPILOT_HEADERS,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKENDS: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": os.getenv("LMS_MODEL", "google/gemma-4-e2b"),
        "reasoning_model": os.getenv("LMS_REASON_MODEL", ""),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        "reasoning_model": os.getenv("OLLAMA_REASON_MODEL", ""),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "reasoning_model": os.getenv("OPENAI_REASON_MODEL", ""),
    },
    "copilot": {
        "base_url": "https://api.githubcopilot.com",
        "api_key": "_copilot_",  # sentinel; replaced at startup by _load_copilot_token()
        "model": os.getenv("COPILOT_MODEL", "gpt-4o"),
        "reasoning_model": os.getenv("COPILOT_REASON_MODEL", ""),
    },
}

# GBC runs at ~60 fps. 16 frames (~267 ms) covers one tile walk animation.
# A/B/Start presses only need ~8 frames for menus to respond.
SETTLE_FRAMES_MOVE = 16   # directional button settle
SETTLE_FRAMES_BUTTON = 8  # A/B/Start/Select settle
SETTLE_FRAMES_CUTSCENE = 30  # longer settle for transitions/cutscenes

# GBC native resolution 160×144. 2× upscale → 320×288 for VLM legibility.
SCREENSHOT_SCALE = 2

# Every N turns the agent executes the game-specific save_sequence.
AUTOSAVE_EVERY_N_TURNS = 60

# After this many consecutive identical button presses, inject a stuck warning.
STUCK_BUTTON_THRESHOLD = 5

# GBC button names (VLM style) → PyBoy button names (lowercase).
# Note: GBC has no L/R buttons — those should not appear in game profiles.
BUTTON_MAP: dict[str, str] = {
    "A": "a",
    "B": "b",
    "Start": "start",
    "Select": "select",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
}

GENERIC_SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous game-playing AI controlling a Game Boy Color game
    running in the PyBoy emulator. Study each screenshot carefully and decide
    the single best button to press next.

    Available buttons: A, B, Up, Down, Left, Right, Start, Select

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
    """Load games/<name>.json or return generic defaults."""
    if name is None:
        return {
            "name": "Generic GBC",
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

def capture_screenshot(
    pyboy: PyBoy,
    scale: int = SCREENSHOT_SCALE,
    *,
    shots_dir: Path | None = None,
) -> str:
    """Return the current screen as a base64-encoded PNG string.

    Converts RGBA→RGB, optionally upscales, optionally saves to disk.
    """
    raw_img = pyboy.screen.image
    if raw_img is None:
        pyboy.tick(1, render=True)
        raw_img = pyboy.screen.image
    img: Image.Image = raw_img.convert("RGB")  # type: ignore[union-attr]
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    if shots_dir:
        shots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        img.save(shots_dir / f"shot-{ts}.png")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def screenshot_hash(b64: str) -> str:
    """MD5 of the base64 string — fast way to detect identical frames (wall hit)."""
    return hashlib.md5(b64.encode()).hexdigest()


# ---------------------------------------------------------------------------
# RAM state reader — ground-truth data from emulator memory
# ---------------------------------------------------------------------------

# Gen 2 character set (subset sufficient for player names and locations)
_GEN2_CHAR: dict[int, str] = {
    0x80: "A", 0x81: "B", 0x82: "C", 0x83: "D", 0x84: "E", 0x85: "F", 0x86: "G",
    0x87: "H", 0x88: "I", 0x89: "J", 0x8A: "K", 0x8B: "L", 0x8C: "M", 0x8D: "N",
    0x8E: "O", 0x8F: "P", 0x90: "Q", 0x91: "R", 0x92: "S", 0x93: "T", 0x94: "U",
    0x95: "V", 0x96: "W", 0x97: "X", 0x98: "Y", 0x99: "Z",
    0xA0: "a", 0xA1: "b", 0xA2: "c", 0xA3: "d", 0xA4: "e", 0xA5: "f", 0xA6: "g",
    0xA7: "h", 0xA8: "i", 0xA9: "j", 0xAA: "k", 0xAB: "l", 0xAC: "m", 0xAD: "n",
    0xAE: "o", 0xAF: "p", 0xB0: "q", 0xB1: "r", 0xB2: "s", 0xB3: "t", 0xB4: "u",
    0xB5: "v", 0xB6: "w", 0xB7: "x", 0xB8: "y", 0xB9: "z",
    0xF6: "0", 0xF7: "1", 0xF8: "2", 0xF9: "3", 0xFA: "4",
    0xFB: "5", 0xFC: "6", 0xFD: "7", 0xFE: "8", 0xFF: "9",
    0x50: "",   # string terminator
    0x7F: " ",  # space
}

_JOHTO_BADGES = [
    (0x01, "Zephyr (Falkner)"),
    (0x02, "Hive (Bugsy)"),
    (0x04, "Plain (Whitney)"),
    (0x08, "Fog (Morty)"),
    (0x10, "Mineral (Jasmine)"),
    (0x20, "Storm (Chuck)"),
    (0x40, "Glacier (Pryce)"),
    (0x80, "Rising (Clair)"),
]

_KANTO_BADGES = [
    (0x01, "Boulder (Brock)"),
    (0x02, "Cascade (Misty)"),
    (0x04, "Thunder (Lt. Surge)"),
    (0x08, "Rainbow (Erika)"),
    (0x10, "Soul (Janine)"),
    (0x20, "Marsh (Sabrina)"),
    (0x40, "Volcano (Blaine)"),
    (0x80, "Earth (Blue)"),
]


def _decode_gen2_name(pyboy: PyBoy, start_addr: int, length: int) -> str:
    chars = []
    for i in range(length):
        b = pyboy.memory[start_addr + i]
        if b == 0x50:
            break
        chars.append(_GEN2_CHAR.get(b, "?"))
    return "".join(chars)


def _read_bcd(pyboy: PyBoy, start_addr: int, length: int) -> int:
    """Read a BCD-encoded integer (used for money in Gen 2)."""
    val = 0
    for i in range(length):
        b = pyboy.memory[start_addr + i]
        val = val * 100 + (((b >> 4) & 0xF) * 10) + (b & 0xF)
    return val


def read_ram_state(pyboy: PyBoy, ram_offsets: dict) -> dict:
    """Read ground-truth game state from emulator RAM.

    Returns a dict with decoded values; any field that fails reads safely.
    """
    state: dict[str, Any] = {}
    try:
        name_addr = int(ram_offsets.get("player_name_start", "0"), 16)
        name_len = int(ram_offsets.get("player_name_length", 10))
        state["player_name"] = _decode_gen2_name(pyboy, name_addr, name_len)
    except Exception:
        state["player_name"] = "?"

    for key in ("map_bank", "map_number", "x_pos", "y_pos"):
        try:
            addr = int(ram_offsets.get(key, "0"), 16)
            state[key] = pyboy.memory[addr]
        except Exception:
            state[key] = None

    try:
        johto_mask = pyboy.memory[int(ram_offsets.get("johto_badges_bitmask", "0xD57C"), 16)]
        state["johto_badges"] = [name for bit, name in _JOHTO_BADGES if johto_mask & bit]
        state["johto_badge_count"] = len(state["johto_badges"])
    except Exception:
        state["johto_badges"] = []
        state["johto_badge_count"] = 0

    try:
        kanto_mask = pyboy.memory[int(ram_offsets.get("kanto_badges_bitmask", "0xD57D"), 16)]
        state["kanto_badges"] = [name for bit, name in _KANTO_BADGES if kanto_mask & bit]
    except Exception:
        state["kanto_badges"] = []

    try:
        money_addr = int(ram_offsets.get("money", "0xD573"), 16)
        money_len = int(ram_offsets.get("money_length", 3))
        state["money"] = _read_bcd(pyboy, money_addr, money_len)
    except Exception:
        state["money"] = None

    try:
        party_count_addr = int(ram_offsets.get("party_count", "0xDA22"), 16)
        state["party_count"] = pyboy.memory[party_count_addr]
    except Exception:
        state["party_count"] = None

    try:
        hp_cur_addr = int(ram_offsets.get("party_slot0_hp_current", "0xDA4C"), 16)
        hp_max_addr = int(ram_offsets.get("party_slot0_hp_max", "0xDA4E"), 16)
        level_addr = int(ram_offsets.get("party_slot0_level", "0xDA49"), 16)
        # HP is big-endian 16-bit
        hp_cur = (pyboy.memory[hp_cur_addr] << 8) | pyboy.memory[hp_cur_addr + 1]
        hp_max = (pyboy.memory[hp_max_addr] << 8) | pyboy.memory[hp_max_addr + 1]
        level = pyboy.memory[level_addr]
        state["lead_hp_current"] = hp_cur
        state["lead_hp_max"] = hp_max
        state["lead_level"] = level
        state["lead_hp_pct"] = round(100 * hp_cur / hp_max) if hp_max > 0 else 0
    except Exception:
        state["lead_hp_current"] = None
        state["lead_hp_max"] = None
        state["lead_level"] = None
        state["lead_hp_pct"] = None

    return state


def format_ram_state(state: dict) -> str:
    """Format RAM state as a concise block for injection into the decide() prompt."""
    lines = ["RAM STATE (authoritative — trust over vision model):"]
    name = state.get("player_name") or "?"
    bank = state.get("map_bank")
    map_n = state.get("map_number")
    x = state.get("x_pos")
    y = state.get("y_pos")
    lines.append(
        f"  Player: {name} | Map: bank={bank} map={map_n} | Pos: X={x} Y={y}"
    )
    johto = state.get("johto_badges", [])
    kanto = state.get("kanto_badges", [])
    j_count = state.get("johto_badge_count", 0)
    badges_str = ", ".join(johto) if johto else "none"
    lines.append(f"  Johto badges ({j_count}/8): {badges_str}")
    if kanto:
        lines.append(f"  Kanto badges: {', '.join(kanto)}")
    hp_cur = state.get("lead_hp_current")
    hp_max = state.get("lead_hp_max")
    level = state.get("lead_level")
    pct = state.get("lead_hp_pct")
    party = state.get("party_count")
    if hp_max:
        heal_warn = " ⚠ LOW HP — HEAL NOW" if pct is not None and pct < 25 else ""
        lines.append(f"  Lead Pokémon: Lv.{level} HP {hp_cur}/{hp_max} ({pct}%){heal_warn}")
    if party is not None:
        lines.append(f"  Party size: {party}")
    money = state.get("money")
    if money is not None:
        lines.append(f"  Money: ¥{money:,}")
    return "\n".join(lines)




def _tick_button(pyboy: PyBoy, vlm_button: str, settle_frames: int, render: bool = True) -> None:
    """Press one button and advance settle_frames, with optional rendering."""
    gbc_key = BUTTON_MAP.get(vlm_button)
    if gbc_key:
        pyboy.button(gbc_key)
    pyboy.tick(settle_frames, render)


def press_button(
    pyboy: PyBoy,
    button: str,
    settle_frames: int = SETTLE_FRAMES_BUTTON,
    shots_dir: Path | None = None,
) -> str:
    """Press a button, wait settle_frames, and return a screenshot."""
    _tick_button(pyboy, button, settle_frames, render=True)
    return capture_screenshot(pyboy, shots_dir=shots_dir)


def walk_steps(
    pyboy: PyBoy,
    button: str,
    repeat: int,
    settle_frames: int = SETTLE_FRAMES_MOVE,
    shots_dir: Path | None = None,
) -> str:
    """Press a directional button repeat times, only rendering on the final step."""
    for i in range(repeat):
        is_last = (i == repeat - 1)
        _tick_button(pyboy, button, settle_frames, render=is_last)
    return capture_screenshot(pyboy, shots_dir=shots_dir)


def save_game(pyboy: PyBoy, save_sequence: list[str]) -> str:
    """Execute the in-game save sequence and return the final screenshot."""
    print("  [autosave] running save sequence…")
    for button in save_sequence:
        _tick_button(pyboy, button, SETTLE_FRAMES_BUTTON, render=True)
        time.sleep(0.08)  # small real-time gap so the game can process inputs
    print("  [autosave] done.")
    pyboy.tick(30, render=True)  # wait for save animation to finish
    return capture_screenshot(pyboy)


# ---------------------------------------------------------------------------
# VLM retry — runs fn() in a thread, pumps PyBoy on main thread if windowed
# ---------------------------------------------------------------------------

_PUMP_INTERVAL = 0.016  # ~60 fps window event pump


def _with_retry(
    fn: Any,
    *,
    retries: int = 6,
    base_delay: float = 10.0,
    pump_fn: Any = None,
    on_auth_error: Any = None,
) -> Any:
    """Call fn() (a blocking VLM call), retrying on OpenAI API errors.

    If pump_fn is provided, the VLM call runs in a background thread while
    pump_fn() (typically pyboy.tick(1, render=False)) is called ~60×/s on the
    main thread so the SDL2 window stays responsive.  When headless, pump_fn
    is None and fn() runs directly on the main thread.

    On failure waits base_delay × attempt seconds before retrying.

    If on_auth_error is provided, a 401 response calls it (to refresh tokens)
    then retries immediately without consuming a retry slot (max 2 refreshes).
    """
    last_exc: Exception | None = None
    _auth_refreshes = 0
    for attempt in range(1, retries + 1):
        try:
            if pump_fn is not None:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(fn)
                    while not future.done():
                        pump_fn()
                        time.sleep(_PUMP_INTERVAL)
                    return future.result()
            return fn()
        except (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.APITimeoutError,
        ) as exc:
            # 401 Unauthorized: token expired — refresh and retry immediately
            if (
                isinstance(exc, openai.APIStatusError)
                and exc.status_code == 401
                and on_auth_error is not None
                and _auth_refreshes < 2
            ):
                _auth_refreshes += 1
                print(
                    f"  [llm] 401 Unauthorized — refreshing Copilot token "
                    f"(refresh {_auth_refreshes}/2)..."
                )
                try:
                    on_auth_error()
                except Exception as refresh_exc:
                    print(f"  [llm] Token refresh failed: {refresh_exc}")
                continue  # retry immediately, don't count against retries
            last_exc = exc
            wait = base_delay * attempt
            print(
                f"  [llm] API error (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {wait:.0f}s..."
            )
            if pump_fn is not None:
                deadline = time.time() + wait
                while time.time() < deadline:
                    pump_fn()
                    time.sleep(_PUMP_INTERVAL)
            else:
                time.sleep(wait)
    raise RuntimeError(f"VLM call failed after {retries} attempts: {last_exc}")


def _extract_json(text: str) -> str:
    """Extract the last complete {...} block from text, stripping any preamble.

    Scanning for the *last* object avoids picking up stray {} inside
    preamble prose that some models emit before the actual JSON.
    """
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Strip <think>...</think> blocks
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Walk backwards from the last '}', then match its opening '{'
    last_close = text.rfind("}")
    if last_close == -1:
        return text.strip()
    depth = 0
    start = -1
    for i in range(last_close, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start != -1:
        return text[start : last_close + 1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Perceive — vision model screen description
# ---------------------------------------------------------------------------

_PERCEIVE_PROMPT = """\
You are a screen-reader for a Game Boy Color Pokemon Silver game running in an emulator.
Look at this screenshot and describe ONLY what you see — no strategy, no button suggestions.

━━ GBC VISUAL SYMBOL GUIDE ━━
Learn these visual patterns to accurately describe the scene:

INDOOR vs OUTDOOR — check the floor tile:
  • OUTDOOR: repeating light/dark green checkerboard grass tiles. Trees appear as rounded
    green blobs or square dark-green blocks at edges. Route paths are narrower corridors.
  • INDOOR: geometric floor tiles (brick, wood planks, carpet, tile patterns). Solid walls.
    A counter or bookshelves usually visible. No sky.
  If you see a checkerboard green grass floor → you are OUTDOORS.

ITEM BALL (collectible on the ground):
  • A small round white/cream sphere with a red top half, sitting on the ground.
  • About the same size as the player character's head. Often surrounded by a small sparkle.
  • Always report item balls in "visible_items" with direction from the player.

WILD POKEMON on overworld:
  • In tall grass patches, you can encounter wild Pokemon — show as flashing encounter, not a
    sprite on the overworld. Report "tall grass ahead" in notes if tall grass is visible.
  • Sprites visible on the overworld that are non-human animal shapes are WILD POKEMON,
    NOT NPCs.

PASSABLE vs IMPASSABLE tiles:
  • Light/dark green checkerboard grass = passable (player can walk there)
  • Dark tree trunk blocks = impassable
  • Ledges (short brown step tiles) = passable going DOWN only, impassable going UP
  • Water = impassable without Surf
  • Building walls = impassable
  • Open path/corridor = passable in all directions it extends

NPC naming rules:
  An NPC is a HUMAN-SHAPED sprite. Wild Pokemon are NOT NPCs.
  Identify NPCs by role and screen position:
  • "Prof. Elm (behind desk)", "Mom (near front door)", "Boy (near bookcase)",
    "Aid (standing left)", "Rival (blocking path north)"
  Use the position always. False negatives (missing an NPC) are far less harmful
  than false positives (calling a Pokemon an NPC). If unsure, do NOT include it.

LOCATION NAMEPLATE:
  In Gen 2, a location nameplate appears as a dark box in the upper-right area of the screen
  when you enter a new area. It shows the town or route name in white text.
  If visible, set nameplate_text to that exact name.

LOCATION NAMING:
  - Outdoors: "New Bark Town", "Route 29", "Cherrygrove City"
  - Indoors: "New Bark Town - Prof. Elm's Lab", "Cherrygrove City - Pokemon Center"
  If you see checkerboard grass floor → location is outdoors (Route or Town name only).

Reply with a JSON object ONLY — no markdown fences:
{
  "screen_type": "overworld" | "dialogue" | "battle" | "menu" | "cutscene" | "unknown",
  "is_outdoor": true | false,
  "nameplate_text": "<exact text from the location banner if visible right now, otherwise null>",
  "dialogue_text": "<exact text in the dialogue box, or null>",
  "menu_options": ["<option 1>", "<option 2>"] or null,
  "battle_info": "<brief description of battle state including HP bars and moves visible, or null>",
  "player_facing": "Up" | "Down" | "Left" | "Right" | "Unknown",
  "adjacent_npc": true | false,
  "adjacent_npc_id": "<npc_id — REQUIRED when adjacent_npc=true, use 'Unknown NPC (<position>)' if unclear. null only when adjacent_npc=false>",
  "visible_npcs": [
    {"npc_id": "<npc_id>", "position": "<e.g. 2 tiles north>"}
  ],
  "visible_door_or_mat": true | false,
  "door_direction": "<direction + distance, or null>",
  "visible_items": "<item balls on the ground with direction from player, or null>",
  "passable_directions": ["Up", "Down", "Left", "Right" — include only directions the player can actually walk],
  "location_name": "<if is_outdoor=true: just Route/Town name. If is_outdoor=false: Town - Building>",
  "notes": "<anything else important — tall grass patches, suspicious sprites, day/night state>"
}
"""


def perceive(
    vision_client: OpenAI,
    vision_model: str,
    screenshot_b64: str,
    *,
    extra_body: dict | None = None,
) -> str:
    """Ask the vision model to describe the current screen as structured text."""
    response = vision_client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {"type": "text", "text": _PERCEIVE_PROMPT},
                ],
            }
        ],
        max_tokens=2048,
        temperature=0.1,
        timeout=60.0,
        **(({"extra_body": extra_body}) if extra_body else {}),
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        rc = getattr(response.choices[0].message, "reasoning_content", None)
        if rc:
            raw = rc.strip()
    if not raw:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        print(f"  [perceive] WARNING: empty response. finish_reason={choice.finish_reason!r} usage={usage}")
    raw = _extract_json(raw)
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        # Full raw for debugging — the game loop will also log it on parse fail
        print(f"  [perceive/parse-fail] raw ({len(raw)} chars): {raw[:300]}")
    return raw


# ---------------------------------------------------------------------------
# World map — persistent cross-session location and NPC tracker
# ---------------------------------------------------------------------------

class WorldMap:
    """Tracks visited buildings and NPCs across all sessions for a given game.

    Stored at ~/.pyboy-agent/world_maps/<game-slug>.json so knowledge
    survives crashes and restarts.
    """

    def __init__(self, game_slug: str) -> None:
        maps_dir = Path.home() / ".pyboy-agent" / "world_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        self.path = maps_dir / f"{game_slug}.json"
        self._summary_cache: str | None = None
        if self.path.exists():
            try:
                self.data: dict = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {"locations": {}, "visited_order": []}
        else:
            self.data = {"locations": {}, "visited_order": []}
        self.data.setdefault("visited_order", [])

    def update(
        self,
        location: str,
        *,
        location_status: str | None = None,
        npc: str | None = None,
        npc_status: str | None = None,
        note: str | None = None,
    ) -> None:
        locs = self.data.setdefault("locations", {})
        is_new = location not in locs
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})
        if is_new:
            order = self.data.setdefault("visited_order", [])
            if location not in order:
                order.append(location)
        if location_status:
            entry["status"] = location_status
        if npc:
            npcs = entry.setdefault("npcs", {})
            npc_entry = npcs.setdefault(npc, {"status": "talked"})
            if npc_status:
                npc_entry["status"] = npc_status
            if note:
                npc_entry["note"] = note
        elif note:
            entry["note"] = note
        self._summary_cache = None
        self.save()

    def record_wall(self, location: str, direction: str) -> None:
        locs = self.data.setdefault("locations", {})
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})
        walls = entry.setdefault("walls", {})
        if not walls.get(direction):
            walls[direction] = True
            self._summary_cache = None
            self.save()

    def clear_walls(self, location: str) -> None:
        """Remove all recorded walls for a location (used to reset impossible states)."""
        entry = self.data.get("locations", {}).get(location)
        if entry and entry.get("walls"):
            entry["walls"] = {}
            self._summary_cache = None
            self.save()

    def get_walls(self, location: str) -> set[str]:
        entry = self.data.get("locations", {}).get(location, {})
        return {d for d, v in entry.get("walls", {}).items() if v}

    def record_tested(self, location: str, direction: str) -> None:
        locs = self.data.setdefault("locations", {})
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})
        tested = entry.setdefault("tested", {})
        if not tested.get(direction):
            tested[direction] = True
            self._summary_cache = None
            self.save()

    def get_untested_directions(self, location: str) -> set[str]:
        entry = self.data.get("locations", {}).get(location, {})
        done = set(entry.get("walls", {}).keys()) | set(entry.get("tested", {}).keys())
        return {"Up", "Down", "Left", "Right"} - done

    def summary(self) -> str:
        if self._summary_cache is not None:
            return self._summary_cache
        locs = self.data.get("locations", {})
        if not locs:
            self._summary_cache = "No locations recorded yet."
            return self._summary_cache
        lines: list[str] = []
        order = self.data.get("visited_order", [])
        if order:
            lines.append("Route taken: " + " → ".join(order))
            lines.append("")
        for loc_name, loc in locs.items():
            status = loc.get("status", "visited")
            note = loc.get("note", "")
            line = f"\u2022 {loc_name} [{status}]"
            walls = loc.get("walls", {})
            if walls:
                line += f" | walls: {', '.join(sorted(walls))}"
            if note:
                line += f" \u2014 {note}"
            lines.append(line)
            for npc_name, npc in loc.get("npcs", {}).items():
                npc_status = npc.get("status", "talked")
                npc_note = npc.get("note", "")
                npc_line = f"    \u21b3 NPC: {npc_name} [{npc_status}]"
                if npc_note:
                    npc_line += f" \u2014 {npc_note}"
                lines.append(npc_line)
        self._summary_cache = "\n".join(lines)
        return self._summary_cache

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception:
            pass


def _best_location_key(world_map: WorldMap, location: str) -> str:
    """Return the closest matching key in world_map, handling vision model name drift."""
    locs = world_map.data.get("locations", {})
    if not locs or not location:
        return location
    if location in locs:
        return location
    # Strip floor suffix like ' (1F)' or ' (2F)'
    base = re.sub(r"\s*\([^)]*\)\s*$", "", location).strip()
    if base and base in locs:
        return base
    # Case-insensitive exact match
    loc_lower = location.lower()
    for k in locs:
        if k.lower() == loc_lower:
            return k
    # Same town, fuzzy building name (share ≥2 meaningful words)
    _stop = {"the", "a", "of", "in", "at", "s", "1f", "2f", "b1f", "town", "city", "route"}
    parts = location.split(" - ", 1)
    if len(parts) == 2:
        town, building = parts[0].lower(), parts[1].lower()
        building_words = set(re.findall(r"\b\w+\b", building)) - _stop
        for k in locs:
            k_parts = k.split(" - ", 1)
            if len(k_parts) == 2 and k_parts[0].lower() == town:
                k_words = set(re.findall(r"\b\w+\b", k_parts[1].lower())) - _stop
                if len(building_words & k_words) >= 2:
                    return k
    return location


# ---------------------------------------------------------------------------
# VLM decision
# ---------------------------------------------------------------------------

def decide(
    reasoning_client: OpenAI,
    reasoning_model: str,
    scene_description: str,
    history: list[dict],
    system_prompt: str,
    *,
    current_goal: str = "",
    stuck_hint: str | None = None,
    memory: str = "",
    story_log: list[str] | None = None,
    goal_log: list[dict] | None = None,
    world_map_summary: str | None = None,
    dialogue_text: str | None = None,
    extra_body: dict | None = None,
    ram_state_text: str | None = None,
) -> tuple[str, int, str, str | None, str | None, dict | None, str]:
    """Ask the reasoning model what button to press next.

    Returns (button, repeat, reason, event, new_goal, map_update, new_memory).
    """
    user_parts: list[str] = [
        "CURRENT SCREEN (from vision model):\n" + scene_description,
    ]
    # RAM state goes first — it's authoritative ground truth
    if ram_state_text:
        user_parts.insert(0, ram_state_text)
    # Dialogue gets its own prominent block so the model can't dismiss it
    if dialogue_text:
        user_parts.append(
            f'NPC DIALOGUE ON SCREEN RIGHT NOW: "{dialogue_text}"\n'
            "READ this carefully. Your \"thinking\" MUST explain what it means for your story progress "
            "and what you should do next. Copy the key sentence to \"event\"."
        )
    if memory:
        user_parts.insert(0, "YOUR GAME DIARY (your own synthesis — trust this):\n" + memory)
    elif story_log:
        user_parts.append(
            "Recent events (bootstrap — summarise these into 'memory' this turn):\n"
            + "\n".join(f"  • {e}" for e in story_log[-10:])
        )
    if goal_log:
        user_parts.append(
            "Goal history — how objectives have changed:\n"
            + "\n".join(f"  turn {g['turn']:04d}: {g['goal']}" for g in goal_log[-5:])
        )
    if world_map_summary:
        user_parts.append(
            "World map — locations and NPCs visited so far:\n" + world_map_summary
        )
    if current_goal:
        user_parts.append(f"Current goal: {current_goal}")
    if stuck_hint:
        user_parts.append(f"⚠ NAVIGATION WARNING: {stuck_hint}")
    user_parts.append("What button should I press next?")
    user_text = "\n\n".join(user_parts)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *history[-10:],
        {"role": "user", "content": user_text},
    ]

    response = reasoning_client.chat.completions.create(
        model=reasoning_model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=4096,
        temperature=0.2,
        timeout=120.0,
        **(({"extra_body": extra_body}) if extra_body else {}),
    )

    raw = (response.choices[0].message.content or "").strip()
    # Always surface API-level thinking (reasoning_content) when present.
    # With enable_thinking=True the model writes thinking into reasoning_content
    # AND the JSON answer into content — both can be non-empty at the same time.
    _reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
    if _reasoning_content:
        _rc_str = str(_reasoning_content).strip()
        if _rc_str:
            # Print full thinking, 120 chars per line for readability
            _lines = textwrap.wrap(_rc_str, width=120)
            print(f"  [thinking] ({len(_rc_str)} chars)")
            for _l in _lines[:20]:  # cap at 20 lines so terminal doesn't flood
                print(f"    {_l}")
            if len(_lines) > 20:
                print(f"    … ({len(_lines) - 20} more lines)")
    # Fallback: if content is empty, use reasoning_content as the raw response
    if not raw and _reasoning_content:
        raw = str(_reasoning_content).strip()
    raw = _extract_json(raw)

    try:
        parsed = json.loads(raw)
        # Also print embedded "thinking" field if model put it in the JSON
        thinking_raw = parsed.get("thinking")
        if thinking_raw and not _reasoning_content:
            # Only print JSON-embedded thinking when there's no API-level thinking
            # (avoids duplicating the same content)
            _t = str(thinking_raw).strip()
            _lines = textwrap.wrap(_t, width=120)
            print(f"  [thinking/json] ({len(_t)} chars)")
            for _l in _lines[:20]:
                print(f"    {_l}")
        button = str(parsed.get("button", "A")).strip()
        reason = str(parsed.get("reason", ""))
        _repeat_raw = parsed.get("repeat", 1)
        try:
            repeat = max(1, min(3, int(_repeat_raw)))
        except (TypeError, ValueError):
            repeat = 1
        if button not in {"Up", "Down", "Left", "Right"}:
            repeat = 1
        event_raw = parsed.get("event")
        event: str | None = str(event_raw).strip() if event_raw else None
        goal_raw = parsed.get("goal")
        new_goal: str | None = str(goal_raw).strip() if goal_raw else None
        map_update_raw = parsed.get("map_update")
        map_update: dict | None = map_update_raw if isinstance(map_update_raw, dict) else None
        memory_raw = parsed.get("memory")
        new_memory: str = str(memory_raw).strip() if memory_raw else ""
    except json.JSONDecodeError:
        button, repeat, reason, event, new_goal, map_update, new_memory = (
            "B", 1, f"(parse error — defaulted to B) raw={raw[:80]}", None, None, None, ""
        )

    return button, repeat, reason, event, new_goal, map_update, new_memory


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(
    rom: str,
    game_profile: dict,
    backend_cfg: dict,
    *,
    headless: bool = False,
    max_turns: int = 0,
    state_file: str | None = None,
    reasoning_backend_cfg: dict | None = None,
    speed: int | None = None,
    is_copilot: bool = False,
) -> None:
    """Main synchronous game loop."""
    _is_copilot_vision = is_copilot or backend_cfg.get("base_url") == "https://api.githubcopilot.com"
    _is_copilot_reason = (
        reasoning_backend_cfg is not None
        and reasoning_backend_cfg.get("base_url") == "https://api.githubcopilot.com"
    )

    if _is_copilot_vision:
        vision_client = _make_copilot_client()
    else:
        vision_client = OpenAI(base_url=backend_cfg["base_url"], api_key=backend_cfg["api_key"])
    vision_model = backend_cfg["model"]

    r_cfg = reasoning_backend_cfg or backend_cfg
    if r_cfg is backend_cfg:
        reasoning_client = vision_client
    elif _is_copilot_reason:
        reasoning_client = _make_copilot_client()
    else:
        reasoning_client = OpenAI(base_url=r_cfg["base_url"], api_key=r_cfg["api_key"])
    reasoning_model = r_cfg.get("reasoning_model") or r_cfg["model"]

    def _refresh_copilot() -> None:
        """Re-read the token file and patch the client's api_key in-place."""
        nonlocal vision_client, reasoning_client
        token = _load_copilot_token()
        if _is_copilot_vision:
            vision_client = _make_copilot_client()
        if _is_copilot_reason or (r_cfg is backend_cfg and _is_copilot_vision):
            reasoning_client = vision_client if r_cfg is backend_cfg else _make_copilot_client()
        print(f"  [copilot] Token refreshed.")

    _auth_error_cb = _refresh_copilot if (_is_copilot_vision or _is_copilot_reason) else None

    system_prompt: str = game_profile["system_prompt"]
    save_sequence: list[str] | None = game_profile.get("save_sequence")
    game_name: str = game_profile.get("name", "GBC Game")

    rom_path = Path(rom)
    shots_dir = rom_path.parent / (rom_path.stem + "_shots")
    notes_path = rom_path.with_suffix(rom_path.suffix + ".pyboy_agent_notes.json")
    state_snapshot_path = rom_path.with_suffix(rom_path.suffix + ".pyboy_agent.state")
    # Drop-file for live operator overrides: write any text to this file
    # while the agent is running and it will be injected as a high-priority
    # instruction into the next turn's decide() call, then deleted.
    _message_file = Path("agent_message.txt")

    print(f"[agent] Game={game_name} | vision={vision_model} | reason={reasoning_model}")
    print(f"[agent] ROM : {rom}")
    print(f"[agent] Override: write to '{_message_file.resolve()}' to inject instructions.")
    # instruction into the next turn's decide() call, then deleted.
    _message_file = Path("agent_message.txt")

    # Persistent notes: story log, goal, memory — survives crashes
    story_log: list[str] = []
    goal_log: list[dict] = []
    current_goal: str = game_profile.get("initial_goal", "")
    memory: str = ""

    if notes_path.exists():
        try:
            saved = json.loads(notes_path.read_text(encoding="utf-8"))
            story_log = saved.get("story_log", [])
            goal_log = saved.get("goal_log", [])
            current_goal = saved.get("current_goal") or current_goal
            memory = saved.get("memory", "")
            print(
                f"[agent] Restored {len(story_log)} story log entries, "
                f"{len(goal_log)} goal changes, memory={'yes' if memory else 'none'}."
            )
        except Exception:
            pass

    # ── World map ────────────────────────────────────────────────────────────
    game_slug = game_name.lower().replace(" ", "-")
    world_map = WorldMap(game_slug)
    print(f"[agent] World map: {len(world_map.data.get('locations', {}))} location(s) — {world_map.path}")

    # ── PyBoy emulator ───────────────────────────────────────────────────────
    window = "null" if headless else "SDL2"
    # Speed: headless → unlimited (0); windowed → realtime (1) so animations
    # are visible. Either can be overridden with --speed.
    emu_speed = speed if speed is not None else (0 if headless else 1)
    print(f"[agent] Starting PyBoy ({window} window, speed={'unlimited' if emu_speed == 0 else f'{emu_speed}x'})…")
    # Suppress PyBoy's carriage-return progress output during ROM load
    _devnull = open(os.devnull, "w")
    _old_stdout = sys.stdout
    sys.stdout = _devnull
    try:
        pyboy = PyBoy(
            rom,
            window=window,
            cgb=True,
            sound_emulated=False,
            log_level="ERROR",
            no_input=False,
        )
    finally:
        sys.stdout = _old_stdout
        _devnull.close()
    pyboy.set_emulation_speed(emu_speed)

    # ── State file restore ───────────────────────────────────────────────────
    if state_file:
        sf = Path(state_file)
        if sf.exists():
            with open(sf, "rb") as f:
                pyboy.load_state(f)
            print(f"[agent] Loaded state from: {sf}")
        else:
            print(f"[agent] WARNING: state file not found: {sf} — starting fresh")
    elif state_snapshot_path.exists() and not state_file:
        # Auto-resume from last autosave snapshot
        try:
            with open(state_snapshot_path, "rb") as f:
                pyboy.load_state(f)
            print(f"[agent] Auto-resumed from last state snapshot: {state_snapshot_path}")
        except Exception as exc:
            print(f"[agent] Could not load auto-state ({exc}) — starting from .sav")

    # Let the game settle after load (boot screen / intro)
    pyboy.tick(60, render=True)

    current_b64 = capture_screenshot(pyboy, shots_dir=shots_dir)

    # When windowed, pump_fn keeps SDL2 events flowing during VLM calls.
    # render=True is required — SDL2 only flushes the Windows message queue
    # during rendering; render=False leaves the window unresponsive.
    # headless: None so VLM runs directly on the main thread (no window).
    pump_fn = (lambda: pyboy.tick(1, render=True)) if not headless else None

    # ── RAM offsets ──────────────────────────────────────────────────────────
    ram_offsets: dict = game_profile.get("ram_offsets", {})
    # has_ram is True when there are real address keys beyond the metadata "note" key
    has_ram = bool({k for k in ram_offsets if k != "note"})

    # ── Loop state ───────────────────────────────────────────────────────────
    history: list[dict] = []
    turn = 0
    start_time = time.time()
    last_button: str | None = None
    consecutive_same: int = 0
    consecutive_a: int = 0
    wall_detected: bool = False
    wall_button: str | None = None
    current_location: str = ""
    # RAM-based position tracking — used for per-tile wall keys and stuck detection
    _last_tile: tuple[int, int] | None = None
    _last_map: tuple[int, int] | None = None
    turns_at_same_tile: int = 0
    _tile_key: str = ""  # computed each turn from RAM

    def flush_notes() -> None:
        try:
            notes_path.write_text(
                json.dumps(
                    {
                        "story_log": story_log,
                        "current_goal": current_goal,
                        "goal_log": goal_log,
                        "memory": memory,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def flush_state() -> None:
        try:
            with open(state_snapshot_path, "wb") as f:
                pyboy.save_state(f)
        except Exception:
            pass

    def shutdown(save: bool = True) -> None:
        flush_notes()
        flush_state()
        pyboy.stop(save=save)
        print("[agent] Stopped.")

    # Handle Ctrl+C gracefully
    _stop_requested = [False]

    def _sigint_handler(signum: int, frame: Any) -> None:
        print("\n[agent] Ctrl+C received — shutting down gracefully…")
        _stop_requested[0] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Main game loop ───────────────────────────────────────────────────────
    try:
        while True:
            if _stop_requested[0]:
                break

            turn += 1
            elapsed = int(time.time() - start_time)
            print(f"\n[turn {turn:04d} | {elapsed//60:02d}:{elapsed%60:02d}]", end=" ")

            # ── Operator override (drop file) ─────────────────────────────────
            _operator_msg: str | None = None
            if _message_file.exists():
                try:
                    _operator_msg = _message_file.read_text(encoding="utf-8").strip()
                    _message_file.unlink()
                    if _operator_msg:
                        print(f"\n  [operator] *** OVERRIDE: {_operator_msg} ***")
                except Exception:
                    pass

            # ── Autosave ─────────────────────────────────────────────────────
            if save_sequence and turn > 1 and turn % AUTOSAVE_EVERY_N_TURNS == 0:
                current_b64 = save_game(pyboy, save_sequence)
                flush_state()

            # ── Read RAM early — used for nav-key, pre-move snapshot, and decide ──
            _ram_state: dict = read_ram_state(pyboy, ram_offsets) if has_ram else {}
            _ram_text: str | None = format_ram_state(_ram_state) if _ram_state else None

            # ── Per-tile position tracking ────────────────────────────────────
            # Wall keys are scoped to individual tiles (map + x + y) so a wall
            # recorded at tile (3,7) never blocks movement from tile (1,6).
            _mb = _ram_state.get("map_bank")
            _mn = _ram_state.get("map_number")
            _cx = _ram_state.get("x_pos")
            _cy = _ram_state.get("y_pos")
            _has_pos = _mb is not None and _mn is not None and _cx is not None and _cy is not None
            if _has_pos:
                _cur_map = (_mb, _mn)
                _cur_tile = (_cx, _cy)
                _tile_key = f"map_{_mb}_{_mn}_x{_cx}_y{_cy}"
                if _cur_map != _last_map:
                    print(f"  [map] {_last_map} -> {_cur_map} | tile ({_cx},{_cy})")
                    _last_map = _cur_map
                    _last_tile = _cur_tile
                    turns_at_same_tile = 0
                elif _cur_tile != _last_tile:
                    turns_at_same_tile = 0
                    _last_tile = _cur_tile
                else:
                    turns_at_same_tile += 1
            else:
                _tile_key = current_location
            _wall_location_key: str = _tile_key

            # ── Navigation hints ─────────────────────────────────────────────
            nav_hint: str | None = None
            # Operator override always goes first — highest priority
            if _operator_msg:
                nav_hint = f"OPERATOR INSTRUCTION (follow this immediately): {_operator_msg}"
            if wall_detected and wall_button:
                nav_hint = (
                    f"Pressing {wall_button!r} did NOT move the character — you hit a "
                    f"wall or obstacle. Choose a DIFFERENT direction immediately."
                )
            elif consecutive_same >= STUCK_BUTTON_THRESHOLD:
                nav_hint = (
                    f"You have pressed '{last_button}' {consecutive_same} times in a row. "
                    f"The character is stuck or looping. Try a completely different "
                    f"direction, or press B to close any open menus."
                )
            if consecutive_a >= 3:
                _a_hint = (
                    f"You have pressed A {consecutive_a} consecutive times. "
                    f"STOP pressing A. If stuck on same NPC: navigate away — press Down or the exit direction."
                )
                nav_hint = (_a_hint + " | " + nav_hint) if nav_hint else _a_hint

            # ── RAM-authoritative tile navigation hint ────────────────────────
            # Build navigation context from per-tile RAM data rather than VLM
            # passable_directions (which are frequently wrong for small rooms).
            _wlk = _tile_key or current_location
            _cardinal = {"Up", "Down", "Left", "Right"}
            known_walls = world_map.get_walls(_wlk) if _wlk else set()
            _tested = {d for d in _cardinal if world_map.get_walls(_wlk) or True} & set()  # placeholder
            # get_tested isn't exposed — compute untried as cardinal minus walls minus tested
            _all_tested: set[str] = set()
            if _wlk:
                _loc_entry = world_map.data.get("locations", {}).get(_wlk, {})
                _all_tested = set(k for k, v in _loc_entry.get("tested", {}).items() if v)
            _untried = _cardinal - known_walls - _all_tested
            if known_walls >= _cardinal:
                # All 4 directions blocked at this exact tile — physically impossible.
                print(f"  [wall-reset] All 4 directions blocked at tile {_wlk} — clearing.")
                if _wlk:
                    world_map.clear_walls(_wlk)
                known_walls = set()
                _untried = _cardinal.copy()
            # Build the tile hint — this is the primary navigation signal
            if _has_pos and _tile_key:
                _tile_hint_parts = [f"RAM tile ({_cx},{_cy}) on map {_mb}/{_mn}."]
                if known_walls:
                    _tile_hint_parts.append(f"Blocked from this tile: {', '.join(sorted(known_walls))}.")
                if _untried:
                    _tile_hint_parts.append(f"Not yet tried from this tile: {', '.join(sorted(_untried))}. Try one of these.")
                if turns_at_same_tile >= 2:
                    _tile_hint_parts.append(f"You have NOT moved for {turns_at_same_tile} turns — character is stuck at this tile.")
                _tile_hint = " ".join(_tile_hint_parts)
                nav_hint = (_tile_hint + " | " + nav_hint) if nav_hint else _tile_hint
            elif known_walls:
                wall_str = f"Blocked directions at current position: {', '.join(sorted(known_walls))}. Do NOT try these."
                nav_hint = (wall_str + " " + nav_hint) if nav_hint else wall_str

            # ── Party-empty critical hint ─────────────────────────────────────
            # If party_count=0 the player has no Pokémon — the only valid goal is
            # to find Prof. Elm and get a starter. Surface this prominently.
            if _ram_state:
                _party_count = _ram_state.get("party_count")
                if _party_count == 0:
                    _party_hint = (
                        "CRITICAL: RAM confirms party_count=0 — you have NO Pokémon. "
                        "Your ONLY goal right now is to reach Prof. Elm's Lab in New Bark Town "
                        "and press A in front of him to receive your starter Pokémon. "
                        "Ignore building exploration until you have a Pokémon."
                    )
                    nav_hint = (_party_hint + " | " + nav_hint) if nav_hint else _party_hint

            scene = _with_retry(lambda: perceive(vision_client, vision_model, current_b64), pump_fn=pump_fn, on_auth_error=_auth_error_cb)
            if scene:
                # Log all key fields from the scene JSON, not just the first 200 chars
                try:
                    _sp_log = json.loads(scene)
                    _loc_log = _sp_log.get("location_name", "?")
                    _type_log = _sp_log.get("screen_type", "?")
                    _out_log = "outdoor" if _sp_log.get("is_outdoor") else "indoor"
                    _face_log = _sp_log.get("player_facing", "?")
                    _pass_log = ",".join(_sp_log.get("passable_directions") or []) or "none"
                    _npc_log = f"adj={_sp_log.get('adjacent_npc_id')}" if _sp_log.get("adjacent_npc") else "no adj NPC"
                    _dlg_log = f' dlg="{_sp_log["dialogue_text"][:60]}"' if _sp_log.get("dialogue_text") else ""
                    _items_log = f' items={_sp_log["visible_items"][:40]}' if _sp_log.get("visible_items") else ""
                    _notes_log = f' notes={_sp_log["notes"][:60]}' if _sp_log.get("notes") else ""
                    print(
                        f"  [scene] {_type_log}/{_out_log} | loc={_loc_log} | "
                        f"facing={_face_log} | passable=[{_pass_log}] | {_npc_log}"
                        f"{_dlg_log}{_items_log}{_notes_log}"
                    )
                    if _sp_log.get("visible_npcs"):
                        for _vnpc in _sp_log["visible_npcs"]:
                            print(f"    NPC: {_vnpc.get('npc_id','?')} @ {_vnpc.get('position','?')}")
                    if _sp_log.get("battle_info"):
                        print(f"  [battle] {_sp_log['battle_info']}")
                except (json.JSONDecodeError, TypeError):
                    # Parse failed — show full raw response for debugging
                    print(f"  [scene/parse-fail] full raw ({len(scene)} chars):")
                    for _raw_line in scene.splitlines()[:40]:
                        print(f"    {_raw_line}")
            else:
                print("  [scene] EMPTY — check that your model supports vision/image inputs.")

            # ── Nameplate / location detection ────────────────────────────────
            _npc_hint: str | None = None
            _dialogue_text: str | None = None
            try:
                scene_parsed = json.loads(scene)
                # Normalise direction strings to title case regardless of what the model returns
                if isinstance(scene_parsed.get("passable_directions"), list):
                    scene_parsed["passable_directions"] = [
                        d.title() for d in scene_parsed["passable_directions"]
                        if isinstance(d, str)
                    ]
                if isinstance(scene_parsed.get("player_facing"), str):
                    scene_parsed["player_facing"] = scene_parsed["player_facing"].title()
                nameplate = scene_parsed.get("nameplate_text")
                if nameplate and isinstance(nameplate, str) and nameplate.strip():
                    world_map.update(nameplate.strip(), location_status="visited")
                    print(f"  [nameplate] Entered: {nameplate.strip()}")
                _new_loc = scene_parsed.get("location_name", "") or ""
                if _new_loc:
                    current_location = _best_location_key(world_map, _new_loc)
                _raw_dlg = scene_parsed.get("dialogue_text")
                if _raw_dlg and isinstance(_raw_dlg, str) and _raw_dlg.strip():
                    _dialogue_text = _raw_dlg.strip()
                    print(f'  [dialogue] "{_dialogue_text}"')
            except (json.JSONDecodeError, AttributeError):
                scene_parsed = {}

            # ── NPC re-talk guard ─────────────────────────────────────────────
            if isinstance(scene_parsed, dict) and scene_parsed.get("adjacent_npc") and current_location:
                _adj_id = (scene_parsed.get("adjacent_npc_id") or "").strip()
                _loc_key = _best_location_key(world_map, current_location)
                loc_npcs = world_map.data.get("locations", {}).get(_loc_key, {}).get("npcs", {})
                already_talked = {k for k, v in loc_npcs.items() if v.get("status") == "talked"}
                if _adj_id and _adj_id in already_talked:
                    _npc_hint = (
                        f"'{_adj_id}' is already marked 'talked' in the world map. "
                        f"Do NOT press A again. Walk to the exit — press Down."
                    )
                elif not _adj_id and already_talked:
                    _npc_hint = (
                        f"Adjacent NPC unidentified. All known NPCs in '{current_location}' "
                        f"are already 'talked': {', '.join(sorted(already_talked))}. "
                        f"Walk away, press Down toward the exit mat."
                    )
                if _npc_hint:
                    print(f"  [retalk] {_npc_hint}")
                    nav_hint = (_npc_hint + " | " + nav_hint) if nav_hint else _npc_hint
            _retalk_guard_fired = bool(_npc_hint)

            # ── Screen type hints ─────────────────────────────────────────────
            _screen_type = scene_parsed.get("screen_type", "") if isinstance(scene_parsed, dict) else ""

            if isinstance(scene_parsed, dict) and scene_parsed.get("is_outdoor") and memory:
                _mem_lower = memory.lower()
                _indoor_words = ["inside", "indoor", "trapped", "lab", "building", "house", "center", "mart", "stuck in"]
                if any(w in _mem_lower for w in _indoor_words):
                    _correction = (
                        "MEMORY CORRECTION: The current scene shows is_outdoor=TRUE — "
                        "you are OUTSIDE on the overworld, NOT inside any building. "
                        f"Location: {scene_parsed.get('location_name', 'unknown outdoor area')}."
                    )
                    nav_hint = (_correction + " | " + nav_hint) if nav_hint else _correction
                    print("  [correct] Memory says indoor but scene is outdoor — injecting correction")

            if _screen_type == "battle":
                _battle_hint = (
                    "YOU ARE IN A BATTLE. Ignore memory about overworld navigation. "
                    "Use FIGHT: navigate to it with directions (repeat=1), then press A. "
                    "Do NOT press directions with repeat>1 in a battle."
                )
                nav_hint = (_battle_hint + " | " + nav_hint) if nav_hint else _battle_hint

            # ── Decide ───────────────────────────────────────────────────────
            if _ram_state:
                _hp_cur = _ram_state.get("lead_hp_current")
                _hp_max = _ram_state.get("lead_hp_max")
                _hp_pct = _ram_state.get("lead_hp_pct")
                _badges = _ram_state.get("johto_badge_count", 0)
                _map_b = _ram_state.get("map_bank")
                _map_n = _ram_state.get("map_number")
                _x = _ram_state.get("x_pos")
                _y = _ram_state.get("y_pos")
                _money = _ram_state.get("money")
                _party = _ram_state.get("party_count")
                _hp_valid = _hp_max is not None and _hp_max > 0
                _hp_str = f"HP={_hp_cur}/{_hp_max}({_hp_pct}%)" if _hp_valid else "HP=n/a"
                _money_str = f" | ¥{_money:,}" if _money is not None else ""
                _party_str = f" | party={_party}" if _party is not None else ""
                print(
                    f"  [ram]   map={_map_b}/{_map_n} pos=({_x},{_y}) | "
                    f"{_hp_str} | badges={_badges}/8{_party_str}{_money_str}"
                )
                if _hp_valid and _hp_pct is not None and _hp_pct < 25:
                    print("  [ram]   ⚠ LOW HP — agent should seek a Pokemon Center")
                if _party == 0:
                    print("  [ram]   ⚠ PARTY EMPTY — agent must get starter from Prof. Elm")

            _think_cfg = r_cfg if reasoning_backend_cfg else backend_cfg
            _think_extra = (
                {"enable_thinking": True}
                if _think_cfg.get("base_url", "").startswith("http://localhost")
                else None
            )
            # When RAM tile data is available, strip passable_directions from the
            # scene JSON before sending to the reasoning model. The VLM frequently
            # returns wrong values (often all 4) which overrides the accurate
            # tile-level hint. RAM position delta is the authoritative source.
            _scene_for_decide = scene
            if _has_pos and scene_parsed and isinstance(scene_parsed, dict):
                _stripped = {k: v for k, v in scene_parsed.items() if k != "passable_directions"}
                try:
                    _scene_for_decide = json.dumps(_stripped)
                except Exception:
                    pass
            button, repeat, reason, event, new_goal, map_update, new_memory = _with_retry(
                lambda: decide(
                    reasoning_client, reasoning_model, _scene_for_decide, history, system_prompt,
                    current_goal=current_goal,
                    stuck_hint=nav_hint,
                    memory=memory,
                    story_log=story_log,
                    goal_log=goal_log,
                    world_map_summary=world_map.summary(),
                    dialogue_text=_dialogue_text,
                    extra_body=_think_extra,
                    ram_state_text=_ram_text,
                ),
                pump_fn=pump_fn,
                on_auth_error=_auth_error_cb,
            )

            # ── Auto-overrides ────────────────────────────────────────────────
            if _screen_type == "dialogue" and not _retalk_guard_fired:
                button = "A"
                repeat = 1
                print("  [auto]  dialogue -> forced A")
            elif _screen_type == "dialogue" and _retalk_guard_fired:
                if button == "A":
                    button = "Down"
                    repeat = 1
                    print("  [auto]  retalk+dialogue -> forced Down (walk away)")
            if _screen_type == "battle" and repeat > 1:
                repeat = 1
                print("  [auto]  battle -> clamped repeat to 1")

            if new_memory:
                memory = new_memory
                print(f"  [memory] {memory[:200]}")

            step_label = f"×{repeat}" if repeat > 1 else ""
            print(f"-> {button:6s}{step_label:4s}| {reason}")

            if event:
                story_log.append(event)
                print(f"  [story] {event}")
            if new_goal and new_goal != current_goal:
                goal_log.append({"turn": turn, "goal": new_goal})
                current_goal = new_goal
                print(f"  [goal]  {current_goal}")
            if map_update and isinstance(map_update.get("location"), str) and map_update["location"]:
                world_map.update(
                    map_update["location"],
                    location_status=map_update.get("location_status") or None,
                    npc=map_update.get("npc") or None,
                    npc_status=map_update.get("npc_status") or None,
                    note=map_update.get("note") or None,
                )
                print(f"  [map]   {map_update['location']} -> {map_update}")

            # ── Persist notes ─────────────────────────────────────────────────
            flush_notes()

            # ── Consecutive button counters ───────────────────────────────────
            if button == last_button:
                consecutive_same += 1
            else:
                consecutive_same = 1
                last_button = button
            consecutive_a = consecutive_a + 1 if button == "A" else 0

            # ── History ───────────────────────────────────────────────────────
            # User message: key scene fields (gives the model context about what was seen)
            try:
                _sp = json.loads(scene) if isinstance(scene, str) else {}
                _h_type = _sp.get("screen_type", "?")
                _h_loc = _sp.get("location_name", "?")
                _h_facing = _sp.get("player_facing", "?")
                _h_dlg = f' | dialogue="{_sp["dialogue_text"][:40]}"' if _sp.get("dialogue_text") else ""
                _h_npc = f' | adj_npc={_sp["adjacent_npc_id"]}' if _sp.get("adjacent_npc") else ""
                _h_ram = (
                    f' | HP={_ram_state.get("lead_hp_pct")}% badges={_ram_state.get("johto_badge_count")}/8'
                    if _ram_state else ""
                )
                _hist_user = f"screen={_h_type} | loc={_h_loc} | facing={_h_facing}{_h_dlg}{_h_npc}{_h_ram}"
            except Exception:
                _hist_user = "screen=?"
            history.append({"role": "user", "content": _hist_user})
            # Assistant message: button + reason (so future turns know WHY we did what we did)
            history.append({
                "role": "assistant",
                "content": json.dumps({
                    "button": button,
                    "repeat": repeat,
                    "reason": reason[:80] if reason else "",
                }),
            })

            # ── Execute button press ──────────────────────────────────────────
            if _screen_type == "dialogue" or button in {"A", "B", "Start", "Select"}:
                settle = SETTLE_FRAMES_BUTTON
            elif _screen_type in {"cutscene", "unknown"}:
                settle = SETTLE_FRAMES_CUTSCENE
            else:
                settle = SETTLE_FRAMES_MOVE

            old_hash = screenshot_hash(current_b64)
            # Snapshot RAM position BEFORE the move (already read at top of turn).
            _pre_x = _ram_state.get("x_pos") if _ram_state else None
            _pre_y = _ram_state.get("y_pos") if _ram_state else None

            if button in {"Up", "Down", "Left", "Right"} and repeat > 1:
                current_b64 = walk_steps(pyboy, button, repeat, settle, shots_dir=shots_dir)
            else:
                current_b64 = press_button(pyboy, button, settle, shots_dir=shots_dir)

            # ── Wall detection ────────────────────────────────────────────────
            new_hash = screenshot_hash(current_b64)
            if button in {"Up", "Down", "Left", "Right"}:
                # Prefer RAM position delta: if position changed → not a wall,
                # regardless of whether the screen looks the same (small rooms
                # have stationary cameras so hash match is meaningless there).
                if has_ram and _pre_x is not None and _pre_y is not None:
                    _post_ram = read_ram_state(pyboy, ram_offsets)
                    _post_x = _post_ram.get("x_pos")
                    _post_y = _post_ram.get("y_pos")
                    _moved = (_post_x != _pre_x or _post_y != _pre_y)
                    wall_detected = not _moved
                else:
                    # Fallback: hash comparison (works in large outdoor areas)
                    wall_detected = new_hash == old_hash
            else:
                wall_detected = False
            wall_button = button if wall_detected else None
            if wall_detected:
                print(f"  [wall]  {button!r} blocked — warning VLM next turn")
                # Record wall at the pre-move tile (tile_key was computed before the press)
                if _tile_key:
                    world_map.record_wall(_tile_key, button)
                    world_map.record_tested(_tile_key, button)
            elif button in {"Up", "Down", "Left", "Right"}:
                if _tile_key:
                    world_map.record_tested(_tile_key, button)

            if max_turns and turn >= max_turns:
                print(f"[agent] Reached max_turns={max_turns}, stopping.")
                break

    finally:
        shutdown(save=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous GBC game agent — drives PyBoy via a local vision model"
    )
    parser.add_argument(
        "--rom",
        default=os.getenv("ROM_PATH"),
        help="Absolute path to the GBC/GB ROM file (.gbc / .gb). "
             "Can also be set via ROM_PATH in .env",
    )
    parser.add_argument(
        "--game",
        default="pokemon-silver",
        metavar="NAME",
        help=(
            "Game profile name (loads games/<NAME>.json). "
            "Default: pokemon-silver"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="lmstudio",
        help="Vision backend (default: lmstudio)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the vision model name for the chosen backend",
    )
    parser.add_argument(
        "--reasoning-backend",
        choices=list(BACKENDS.keys()),
        default=None,
        metavar="BACKEND",
        help="Separate reasoning backend. Defaults to same as --backend.",
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        metavar="MODEL",
        help="Override the reasoning model name.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a display window (null window, max speed). Default shows SDL2 window.",
    )
    parser.add_argument(
        "--state",
        default=None,
        metavar="FILE",
        help="Path to a PyBoy state file to load (.pyboy_agent.state). "
             "Omit to auto-resume from the last autosave snapshot next to the ROM.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Stop after this many turns (0 = run forever)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        metavar="N",
        help="Emulation speed multiplier (0=unlimited, 1=realtime, 2=2x, …). "
             "Default: 1 with SDL2 window, 0 with --headless.",
    )
    args = parser.parse_args()

    if not args.rom:
        parser.error(
            "ROM path is required. Pass --rom <path> or set ROM_PATH in your .env file.\n"
            "  Example: ROM_PATH=H:/Games/GBC/Pokemon Silver.gbc"
        )

    game_profile = load_game_profile(args.game)

    vision_cfg = dict(BACKENDS[args.backend])
    if args.model:
        vision_cfg["model"] = args.model

    if args.backend == "openai" and not vision_cfg["api_key"]:
        print("ERROR: Set OPENAI_API_KEY in your environment or .env file.")
        sys.exit(1)

    if args.backend == "copilot":
        try:
            vision_cfg["api_key"] = _load_copilot_token()
            print(f"[agent] Copilot backend: token loaded from {_COPILOT_TOKEN_PATH}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    reasoning_cfg: dict | None = None
    if args.reasoning_backend and args.reasoning_backend != args.backend:
        reasoning_cfg = dict(BACKENDS[args.reasoning_backend])
        if args.reasoning_model:
            reasoning_cfg["reasoning_model"] = args.reasoning_model
        if args.reasoning_backend == "openai" and not reasoning_cfg["api_key"]:
            print("ERROR: Set OPENAI_API_KEY in your environment or .env file.")
            sys.exit(1)
        if args.reasoning_backend == "copilot":
            try:
                reasoning_cfg["api_key"] = _load_copilot_token()
                print(f"[agent] Copilot reasoning backend: token loaded.")
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                sys.exit(1)
    elif args.reasoning_model:
        vision_cfg["reasoning_model"] = args.reasoning_model

    run_agent(
        rom=args.rom,
        game_profile=game_profile,
        backend_cfg=vision_cfg,
        headless=args.headless,
        max_turns=args.max_turns,
        state_file=args.state,
        reasoning_backend_cfg=reasoning_cfg,
        speed=args.speed,
        is_copilot=args.backend == "copilot",
    )


if __name__ == "__main__":
    main()
