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
import datetime
import hashlib
import json
import os
import re
import signal
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import uuid

from dotenv import load_dotenv
import openai
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
        # Vision model — handles image → scene description.
        # Gemma 4 E4B: compact multimodal, fast on 8 GB VRAM.
        "model": os.getenv("LMS_MODEL", "google/gemma-4-e4b"),
        # Reasoning model — text-only, handles strategy + button decision.
        # Override with LMS_REASON_MODEL; falls back to same model if not set.
        "reasoning_model": os.getenv("LMS_REASON_MODEL", ""),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        "reasoning_model": os.getenv("OLLAMA_REASON_MODEL", ""),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "reasoning_model": os.getenv("OPENAI_REASON_MODEL", ""),
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

# How often (in turns) the goal-tracker VLM call re-assesses the situation.
GOAL_UPDATE_EVERY_N_TURNS = 10

# After this many consecutive presses of the same button the agent is
# considered stuck. A warning is injected into the next VLM prompt and
# a goal-tracker update is triggered immediately.
STUCK_BUTTON_THRESHOLD = 5


# Fallback system prompt used when no game profile is loaded.
GENERIC_SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous game-playing AI controlling a game running in the
    mGBA emulator. Study each screenshot carefully and decide the single best
    button to press next.

    Available buttons: A, B, Up, Down, Left, Right, Start, L, R
    Do NOT use Select — it does nothing in this game.

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

async def _with_retry(fn: Any, *, retries: int = 6, base_delay: float = 10.0) -> Any:
    """Call fn() in a thread executor, retrying on OpenAI API errors.

    Runs the blocking OpenAI call in the default thread pool so the asyncio
    event loop stays responsive. On failure waits base_delay * attempt seconds
    (non-blocking) — handles model swaps / reloads in LM Studio / Ollama.
    """
    loop = asyncio.get_event_loop()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await loop.run_in_executor(None, fn)
        except (
            openai.APIConnectionError,
            openai.APIStatusError,
            openai.APITimeoutError,
        ) as exc:
            last_exc = exc
            wait = base_delay * attempt
            print(
                f"  [llm] API error (attempt {attempt}/{retries}): {exc}. "
                f"Model may be loading — retrying in {wait:.0f}s…"
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"VLM call failed after {retries} attempts: {last_exc}")


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


def _screenshot_hash(b64: str) -> str:
    """MD5 of the raw base64 string — fast way to detect identical frames (wall hit)."""
    return hashlib.md5(b64.encode()).hexdigest()


# Structured prompt sent to the vision model each turn.
# The output is plain text consumed by the reasoning model — no image needed there.
_PERCEIVE_PROMPT = """\
Look at this Pokemon GBA screenshot. Describe what you see as JSON.

Classification help:
- If you see FIGHT / BAG / POKEMON / RUN → screen_type is "battle"
- If you see a text box with dialogue → screen_type is "dialogue"
- If you see a menu list (not battle) → screen_type is "menu"
- Otherwise → screen_type is "overworld"

Reply with ONLY this JSON (no markdown fences, no extra text):
{
  "screen_type": "overworld" | "dialogue" | "battle" | "menu",
  "dialogue_text": "<exact text in any dialogue/text box, or null>",
  "menu_options": ["<option1>", "<option2>"] or null,
  "battle_info": "<Pokemon names, levels, HP bars if in battle, or null>",
  "player_facing": "up" | "down" | "left" | "right" | "unknown",
  "adjacent_npc": true | false,
  "surroundings": {
    "up": "<what is directly above the player: open grass, trees, wall, NPC, door, path, water, etc.>",
    "down": "<what is directly below the player>",
    "left": "<what is directly to the left>",
    "right": "<what is directly to the right>"
  },
  "location_name": "<your best guess: 'Route 101', 'Oldale Town', 'Littleroot Town - Prof. Birch Lab', etc.>",
  "notes": "<anything else notable: items on ground, doors, trainers, nameplate banners>"
}
"""


def perceive(
    vision_client: OpenAI,
    vision_model: str,
    screenshot_b64: str,
) -> str:
    """Ask the vision model to describe the current screen as structured text.

    Returns a JSON string (or a plain-text fallback on parse failure) describing
    what's visible. This is fed as context to the reasoning model in decide().
    """
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
        max_tokens=4096,
        temperature=0.1,
        timeout=120.0,
        extra_body={"enable_thinking": False},
    )
    raw = (response.choices[0].message.content or "").strip()
    # Thinking models may put output in reasoning_content with content empty.
    # Extract the JSON block that contains 'screen_type' from the trace.
    if not raw:
        rc = getattr(response.choices[0].message, "reasoning_content", None) or ""
        if rc:
            print(f"  [perceive] content was empty, extracting JSON from reasoning_content ({len(rc)} chars)")
            # Look specifically for a JSON object containing screen_type — the real output
            _m = list(re.finditer(r'\{[^{}]*"screen_type"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', rc, re.DOTALL))
            raw = _m[-1].group(0).strip() if _m else ""
    if not raw:
        # Log the full finish_reason and token counts to help diagnose
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        print(f"  [perceive] WARNING: empty response. finish_reason={choice.finish_reason!r} usage={usage}")
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Strip <think>...</think> blocks that some reasoning models prefix to their output
    if "<think>" in raw:
        # Remove everything between <think> and </think> (the reasoning scratchpad)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Validate — return raw string either way; reasoning model receives it as text
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        pass  # pass the raw string through; reasoning model can still use it
    return raw


# ---------------------------------------------------------------------------
# World map — persistent cross-session location and NPC tracker
# ---------------------------------------------------------------------------

class WorldMap:
    """Tracks visited buildings and NPCs across all sessions for a given game.

    Stored at ~/.mgba-live-mcp/world_maps/<game-slug>.json so knowledge
    survives crashes, restarts, and --session resumes indefinitely.

    Location statuses: "visited" (entered but not fully explored),
                       "fully_explored" (all rooms, NPCs, items checked).
    NPC statuses:      "talked", "quest_active", "quest_complete".
    """

    # Regex that matches real Pokemon location names. Prevents VLM hallucinations
    # like "Unknown grassy area" from persisting in the world map across sessions.
    _REAL_LOCATION_RE = re.compile(
        r'^(Route \d|Littleroot|Oldale|Petalburg|Rustboro|Dewford|Slateport|Mauville|'
        r'Verdanturf|Fallarbor|Lavaridge|Fortree|Lilycove|Mossdeep|Sootopolis|'
        r'Pacifidlog|Ever Grande|Pokemon|Pokémon|Poké|Prof\.|map_\d)',
        re.IGNORECASE,
    )

    def __init__(self, game_slug: str) -> None:
        maps_dir = Path.home() / ".mgba-live-mcp" / "world_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        self.path = maps_dir / f"{game_slug}.json"
        self._summary_cache: str | None = None  # invalidated on every update()
        if self.path.exists():
            try:
                raw_data: dict = json.loads(self.path.read_text(encoding="utf-8"))
                # Strip hallucinated location names on load — keeps disk file clean
                clean_locs = {
                    k: v for k, v in raw_data.get("locations", {}).items()
                    if self._REAL_LOCATION_RE.search(k)
                }
                clean_order = [
                    x for x in raw_data.get("visited_order", [])
                    if x in clean_locs
                ]
                removed = len(raw_data.get("locations", {})) - len(clean_locs)
                if removed:
                    print(f"[world_map] Pruned {removed} non-canonical location(s) on load.")
                self.data: dict = {"locations": clean_locs, "visited_order": clean_order}
            except Exception:
                self.data = {"locations": {}, "visited_order": []}
        else:
            self.data = {"locations": {}, "visited_order": []}
        # Ensure visited_order exists in older saved files
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
        # Reject hallucinated location names silently — only store real places.
        if not self._REAL_LOCATION_RE.search(location):
            return
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
        self._summary_cache = None  # invalidate cache
        self.save()

    def record_wall(self, location: str, direction: str) -> None:
        """Record that walking in `direction` hits a wall at `location`.

        Only writes to disk when a direction is newly discovered, so repeated
        wall hits in the same direction cost nothing after the first.
        """
        locs = self.data.setdefault("locations", {})
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})
        walls = entry.setdefault("walls", {})
        if not walls.get(direction):
            walls[direction] = True
            self._summary_cache = None
            self.save()

    def get_walls(self, location: str) -> set[str]:
        """Return the set of confirmed wall directions for `location`."""
        entry = self.data.get("locations", {}).get(location, {})
        return {d for d, v in entry.get("walls", {}).items() if v}

    def record_tested(self, location: str, direction: str) -> None:
        """Record that `direction` was attempted at `location` (wall or open move).

        Combined with record_wall(), this lets get_untested_directions() return
        only directions the agent has never tried yet — powering the boundary scan.
        Only writes to disk when newly discovered.
        """
        locs = self.data.setdefault("locations", {})
        entry = locs.setdefault(location, {"status": "visited", "npcs": {}})
        tested = entry.setdefault("tested", {})
        if not tested.get(direction):
            tested[direction] = True
            self._summary_cache = None
            self.save()

    def get_untested_directions(self, location: str) -> set[str]:
        """Return cardinal directions not yet attempted at `location`.

        A direction is 'done' once it appears in either `walls` or `tested`.
        The boundary scan is complete when this returns an empty set.
        """
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


def _best_location_key(world_map: "WorldMap", location: str) -> str:
    """Return the closest matching key already in world_map, handling vision model name drift.

    Handles cases like 'Prof. Birch's House (1F)' vs 'Prof. Birch's Lab' — the vision
    model sometimes hallucinates building names. Priority:
      1. Exact match
      2. Strip floor suffix  ' (1F)', ' (2F)', etc.
      3. Case-insensitive exact
      4. Same town prefix + 2+ shared content words (e.g. both contain 'Birch')
    Falls back to the original string (creates a new world map entry).
    """
    locs = world_map.data.get("locations", {})
    if not locs or not location:
        return location
    if location in locs:
        return location
    # Strip floor suffix
    base = re.sub(r"\s*\([^)]*\)\s*$", "", location).strip()
    if base and base in locs:
        return base
    # Case-insensitive exact
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
) -> tuple[str, int, str, str | None, str | None, dict | None, str]:
    """Ask the reasoning model what button to press next, given a text scene description.

    Returns (button, repeat, reason, event, new_goal, map_update, new_memory).
    """
    # Build the user text: memory → scene → recent events → world map → goal → nav warnings
    user_parts: list[str] = [
        "CURRENT SCREEN (from vision model):\n" + scene_description,
    ]
    if memory:
        user_parts.insert(0, "YOUR GAME DIARY (your own synthesis — trust this):\n" + memory)
    elif story_log:
        # No memory yet — show raw events as bootstrap context
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
        # Rolling window of last 6 turns as proper user/assistant pairs.
        # Each pair: compact user summary of what was on screen, then the button chosen.
        # This gives the model a valid alternating message structure so it can
        # detect repeated states (stuck-loop) without excessive token cost.
        *history[-6:],
        {"role": "user", "content": user_text},
    ]

    response = reasoning_client.chat.completions.create(
        model=reasoning_model,
        messages=messages,
        max_tokens=4096,
        temperature=0.2,
        timeout=180.0,
        extra_body={"enable_thinking": False},
    )

    raw = (response.choices[0].message.content or "").strip()
    # If content is empty (thinking model), extract JSON with 'button' key from reasoning_content
    if not raw:
        rc = getattr(response.choices[0].message, "reasoning_content", None) or ""
        if rc:
            _matches = list(re.finditer(r'\{[^{}]*"button"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', rc, re.DOTALL))
            raw = _matches[-1].group(0).strip() if _matches else rc.strip()
    # Some models wrap JSON in ```json fences despite the system prompt telling
    # them not to. Strip fences before parsing.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Strip <think>...</think> reasoning blocks before JSON parsing
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(raw)
        thinking_raw = parsed.get("thinking")
        if thinking_raw:
            print(f"  [think] {str(thinking_raw).strip()}")
        button = str(parsed.get("button", "A")).strip()
        # repeat: how many steps to take in this direction (directional buttons only).
        # Clamped to 1-3 so the VLM re-evaluates frequently enough to catch
        # doors, NPCs, and item balls before overshooting them.
        _repeat_raw = parsed.get("repeat", 1)
        try:
            repeat = max(1, min(3, int(_repeat_raw)))
        except (TypeError, ValueError):
            repeat = 1
        # Only allow repeat > 1 for directional buttons — not for A/B/Start etc.
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
        button, repeat, reason, event, new_goal, map_update, new_memory = "B", 1, f"(parse error — defaulted to B) raw={raw[:80]}", None, None, None, ""

    return button, repeat, reason, event, new_goal, map_update, new_memory


# ---------------------------------------------------------------------------
# Bridge IPC client (direct file IPC — no MCP/session registry needed)
# ---------------------------------------------------------------------------

def _to_lua_value(value: Any) -> str:
    """Serialize a Python value to a Lua literal (mirrors live_cli.py's to_lua_value)."""
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(_to_lua_value(v) for v in value) + "}"
    if isinstance(value, dict):
        parts: list[str] = []
        for k in sorted(value.keys(), key=str):
            ks = str(k)
            if ks.isidentifier():
                parts.append(f"{ks} = {_to_lua_value(value[k])}")
            else:
                parts.append(f'["{ks}"] = {_to_lua_value(value[k])}')
        return "{ " + ", ".join(parts) + " }"
    raise TypeError(f"Unsupported value type: {type(value)}")


class BridgeClient:
    """Talks directly to mgba_live_bridge.lua via file IPC.

    Bypasses the mgba-live-mcp CLI entirely so there is no session.json,
    no PID tracking, and no prune_dead_sessions() interference.
    """

    def __init__(self, ipc_dir: Path) -> None:
        self.ipc_dir = ipc_dir
        self.command_path = ipc_dir / "command.lua"
        self.response_path = ipc_dir / "response.json"
        self.heartbeat_path = ipc_dir / "heartbeat.json"
        shots = ipc_dir / "screenshots"
        shots.mkdir(exist_ok=True)
        self._shots_dir = shots

    async def send(
        self, kind: str, payload: dict | None = None, timeout: float = 15.0
    ) -> dict:
        """Send one command to the bridge and return its response dict."""
        payload = payload or {}
        req_id = uuid.uuid4().hex
        command = {"id": req_id, "kind": kind, **payload}

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout

        # Wait for bridge to consume any previous command
        while self.command_path.exists():
            if loop.time() > deadline:
                raise TimeoutError("Bridge busy — command.lua not consumed in time")
            await asyncio.sleep(0.02)

        # Clear stale response — retry on Windows file-locking errors
        for _ in range(10):
            try:
                self.response_path.unlink(missing_ok=True)
                break
            except PermissionError:
                await asyncio.sleep(0.05)

        # Write command atomically
        tmp = self.command_path.with_suffix(".tmp")
        tmp.write_text("return " + _to_lua_value(command) + "\n", encoding="utf-8")
        tmp.replace(self.command_path)

        # Wait for matching response
        while loop.time() < deadline:
            if self.response_path.exists():
                try:
                    resp = json.loads(self.response_path.read_text(encoding="utf-8"))
                    if resp.get("id") == req_id:
                        return resp
                except json.JSONDecodeError:
                    pass
            await asyncio.sleep(0.02)

        # Clean up command if bridge never consumed it
        if self.command_path.exists():
            self.command_path.unlink(missing_ok=True)
        raise TimeoutError(f"Bridge timeout waiting for response to '{kind}'")

    async def screenshot(self) -> str:
        """Capture a screenshot and return it as base64-encoded PNG."""
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shot_path = self._shots_dir / f"shot-{ts}.png"
        resp = await self.send("screenshot", {"path": shot_path.as_posix()})
        if not resp.get("ok"):
            raise RuntimeError(f"Bridge screenshot error: {resp.get('error', resp)}")
        raw = shot_path.read_bytes()
        return base64.b64encode(raw).decode()

    async def tap_and_screenshot(
        self, key: str, duration: int = 2, wait_frames: int = SETTLE_FRAMES
    ) -> str | None:
        """Press a button and return a screenshot taken after wait_frames settle."""
        tap_resp = await self.send("tap_key", {"key": key, "duration": duration})
        if not tap_resp.get("ok"):
            raise RuntimeError(f"Bridge tap error: {tap_resp.get('error', tap_resp)}")

        # Wait for the tap + settle frames to pass, using the heartbeat frame counter.
        tap_frame = tap_resp.get("frame", 0)
        target_frame = tap_frame + duration + wait_frames
        for _ in range(300):  # up to 6 s at 20 ms polls
            try:
                hb = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
                if hb.get("frame", 0) >= target_frame:
                    break
            except (json.JSONDecodeError, OSError):
                pass
            await asyncio.sleep(0.02)

        try:
            return await self.screenshot()
        except Exception:
            return None  # screenshot is best-effort; caller can retry

    async def read_range(self, start: int, length: int) -> list[int]:
        """Read `length` bytes starting at `start` address. Returns list of byte values."""
        if length <= 0:
            return []
        resp = await self.send("read_range", {"start": start, "length": length})
        if resp.get("error"):
            raise RuntimeError(f"Bridge read_range error: {resp['error']}")
        inner = resp.get("data", {})
        # Bridge wraps the Lua read_range result {start,length,data} as the "data" field
        if isinstance(inner, dict):
            data = inner.get("data", [])
        else:
            data = inner  # fallback: bare list (defensive)
        if not isinstance(data, list):
            raise TypeError(f"read_range expected list, got {type(data).__name__}: {inner!r:.120}")
        return data

    async def read_u8(self, addr: int) -> int:
        """Read a single unsigned byte from `addr`."""
        data = await self.read_range(addr, 1)
        return data[0] if data else 0

    async def read_u16(self, addr: int) -> int:
        """Read a little-endian unsigned 16-bit value from `addr`."""
        data = await self.read_range(addr, 2)
        return (data[1] << 8 | data[0]) if len(data) >= 2 else 0

    async def read_u32(self, addr: int) -> int:
        """Read a little-endian unsigned 32-bit value from `addr`."""
        data = await self.read_range(addr, 4)
        if len(data) >= 4:
            return data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)
        return 0


# ---------------------------------------------------------------------------
# High-level game loop helpers (thin wrappers over BridgeClient)
# ---------------------------------------------------------------------------

async def walk_steps(
    bridge: BridgeClient,
    button: str,
    repeat: int,
    settle_frames: int = SETTLE_FRAMES,
) -> str | None:
    """Press a directional button `repeat` times without re-querying the VLM.

    All steps except the last are fired as quick taps with minimal settle time
    so the character walks continuously. The final tap waits settle_frames and
    returns a screenshot for replanning.
    """
    for _ in range(repeat - 1):
        await bridge.send("tap_key", {"key": button, "duration": 2})
        await asyncio.sleep(0.1)  # ~6 frames at 60 fps — enough for one tile
    return await bridge.tap_and_screenshot(button, wait_frames=settle_frames)


async def capture_screenshot(bridge: BridgeClient, retries: int = 3) -> str:
    """Return a base64 PNG from the bridge, retrying on transient errors."""
    for attempt in range(1, retries + 1):
        try:
            return await bridge.screenshot()
        except Exception as exc:
            if attempt < retries:
                print(f"  [screenshot] error (attempt {attempt}/{retries}): {exc}, retrying…")
                await asyncio.sleep(1.0)
    raise RuntimeError("Failed to capture screenshot after all retries.")


async def press_button(
    bridge: BridgeClient,
    button: str,
    wait_frames: int = SETTLE_FRAMES,
) -> str | None:
    """Tap a button and return the post-settle screenshot (or None)."""
    return await bridge.tap_and_screenshot(button, wait_frames=wait_frames)


async def save_game(bridge: BridgeClient, save_sequence: list[str]) -> str | None:
    """Execute the game-specific save_sequence and return the final screenshot."""
    print("  [autosave] running save sequence…")
    for key in save_sequence:
        await bridge.tap_and_screenshot(key, wait_frames=4)
        await asyncio.sleep(0.15)
    print("  [autosave] done.")
    return await capture_screenshot(bridge)


# ---------------------------------------------------------------------------
# RAM-based game state reader
# ---------------------------------------------------------------------------

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
    reasoning_backend_cfg: dict | None = None,
) -> None:
    # Vision client: handles screenshot → scene description (needs multimodal support).
    vision_client = OpenAI(base_url=backend_cfg["base_url"], api_key=backend_cfg["api_key"])
    vision_model = backend_cfg["model"]

    # Reasoning client: handles scene description + context → button decision (text only).
    # Falls back to the same backend/model as vision if no separate backend is configured.
    r_cfg = reasoning_backend_cfg or backend_cfg
    reasoning_client = (
        vision_client
        if r_cfg is backend_cfg
        else OpenAI(base_url=r_cfg["base_url"], api_key=r_cfg["api_key"])
    )
    reasoning_model = r_cfg.get("reasoning_model") or r_cfg["model"]

    system_prompt: str = game_profile["system_prompt"]
    save_sequence: list[str] | None = game_profile.get("save_sequence")
    game_name: str = game_profile.get("name", "game")

    print(f"[agent] Game={game_name} | vision={vision_model} | reason={reasoning_model} | rom={rom}")

    runtime_dir = Path.home() / ".mgba-live-mcp" / "runtime"

    # ── Session management ──────────────────────────────────────────────────
    if session_id:
        # Resume path: the bridge is already running in mGBA with an existing
        # session directory.  Just point BridgeClient at its IPC files.
        ipc_dir = runtime_dir / session_id
        if not (ipc_dir / "heartbeat.json").exists():
            raise RuntimeError(
                f"No heartbeat.json found for session {session_id}. "
                "Is mGBA still running with the bridge script loaded?"
            )
        print(f"[agent] Resuming session {session_id}")
    else:
        # New session path: generate mgba_launcher.lua then wait for the user
        # to load it in mGBA's Scripting window.
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ipc_dir = runtime_dir / session_id
        ipc_dir.mkdir(parents=True, exist_ok=True)

        bridge_path = Path(__file__).parent / "mgba_live_bridge.lua"
        launcher_path = Path(__file__).parent / "mgba_launcher.lua"

        sdir_lua = ipc_dir.as_posix()
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
        print(f"[agent] Session dir: {ipc_dir}")
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

        heartbeat = ipc_dir / "heartbeat.json"
        poll = 0
        while not heartbeat.exists():
            poll += 1
            if poll % 5 == 0:
                print(f"[agent] Waiting for mGBA bridge (poll {poll})…")
            await asyncio.sleep(2.0)
        print("[agent] Bridge ready! Starting game loop…")

    # ── Bridge client ───────────────────────────────────────────────────────
    bridge = BridgeClient(ipc_dir)

    # ── RAM state reader ────────────────────────────────────────────────────
    ram_offsets = game_profile.get("ram_offsets", {})
    game_state = GameState(ram_offsets) if ram_offsets else None
    if game_state:
        print(f"[agent] RAM state reader: {len(game_state._addr)} address(es) configured")

    # ── World map (persistent cross-session knowledge base) ─────────────────
    game_slug = game_name.lower().replace(" ", "-")
    world_map = WorldMap(game_slug)
    print(f"[agent] World map: {len(world_map.data.get('locations', {}))} location(s) — {world_map.path}")

    # ── Initial screenshot ──────────────────────────────────────────────────
    await asyncio.sleep(1.0)
    current_b64 = await capture_screenshot(bridge, retries=6)

    # ── Persistent notes (survives crashes / resumes) ─────────────────────
    # notes.json is written to the session IPC directory after every turn so
    # that story_log and current_goal are restored when --session is used.
    notes_path = ipc_dir / "notes.json"
    if notes_path.exists():
        try:
            saved = json.loads(notes_path.read_text(encoding="utf-8"))
            story_log: list[str] = saved.get("story_log", [])
            goal_log: list[dict] = saved.get("goal_log", [])
            current_goal: str = saved.get("current_goal") or game_profile.get("initial_goal", "")
            memory: str = saved.get("memory", "")
            print(f"[agent] Restored {len(story_log)} story log entries, {len(goal_log)} goal changes, memory={'yes' if memory else 'none'} from previous run.")
        except Exception:
            story_log = []
            goal_log = []
            current_goal = game_profile.get("initial_goal", "")
            memory = ""
    else:
        story_log = []
        goal_log = []
        current_goal = game_profile.get("initial_goal", "")
        memory = ""

    history: list[dict] = []
    turn = 0
    start_time = time.time()
    last_button: str | None = None
    consecutive_same: int = 0
    recent_buttons: list[str] = []  # last N buttons for oscillation detection
    wall_detected: bool = False
    wall_button: str | None = None
    current_location: str = ""  # updated each turn from scene; used for wall tracking
    last_map_id: int | None = None  # for detecting map transitions via RAM

    while True:
        turn += 1
        elapsed = int(time.time() - start_time)
        print(f"\n[turn {turn:04d} | {elapsed//60:02d}:{elapsed%60:02d}]", end=" ")

        # ── Autosave ────────────────────────────────────────────────────────
        if save_sequence and turn > 1 and turn % AUTOSAVE_EVERY_N_TURNS == 0:
            current_b64 = await save_game(bridge, save_sequence)

        # ── Navigation hints: wall hit or stuck loop ────────────────────────
        nav_hint: str | None = None
        if wall_detected and wall_button:
            nav_hint = f"'{wall_button}' hit a wall. Try a different direction."
        elif consecutive_same >= STUCK_BUTTON_THRESHOLD:
            nav_hint = f"Pressed '{last_button}' {consecutive_same} times — stuck. Try something different."

        # Oscillation detection: check recent_buttons for repeating patterns
        # e.g. Down,Right,Down,Right or Down,Right,Down,Right,Down,Right
        if not nav_hint and len(recent_buttons) >= 6:
            tail = recent_buttons[-8:]  # check last 8
            for pattern_len in (2, 3):
                pattern = tail[-pattern_len:]
                repeats = 0
                for i in range(len(tail) - pattern_len, -1, -pattern_len):
                    chunk = tail[i:i + pattern_len]
                    if chunk == pattern:
                        repeats += 1
                    else:
                        break
                if repeats >= 3:
                    nav_hint = (
                        f"OSCILLATING: repeating {' → '.join(pattern)} pattern "
                        f"({repeats}× in a row). Break the loop — try B, Start, "
                        f"A, or a completely different direction."
                    )
                    break
        # Known walls for this location
        known_walls = world_map.get_walls(current_location) if current_location else set()
        if known_walls:
            wall_str = f"KNOWN WALLS in {current_location}: {', '.join(sorted(known_walls))}. Do NOT try these directions."
            nav_hint = (wall_str + " " + nav_hint) if nav_hint else wall_str

        # ── RAM state read (instant, no VLM needed) ───────────────────────────
        ram_state: dict[str, Any] = {}
        ram_summary: str = ""
        if game_state:
            try:
                ram_state = await game_state.read(bridge)
                ram_summary = game_state.summary(ram_state)
                # Compact header: money + badges on the turn line
                _money = ram_state.get("money")
                _badges = ram_state.get("badges")
                _map_id = ram_state.get("map_id")
                _header_parts = []
                if _money is not None:
                    _header_parts.append(f"¥{_money:,}")
                if _badges is not None:
                    _header_parts.append(f"{_badges}★")
                if _map_id is not None:
                    _header_parts.append(f"map:{_map_id}")
                if _header_parts:
                    print(f"  [ram] {' '.join(_header_parts)}")
                # Party/enemy info on separate line if present
                _party = ram_state.get("party", [])
                if _party:
                    _p_parts = [f"{p['nickname']} {p['hp']}/{p['max_hp']}" for p in _party]
                    print(f"  [party] {' | '.join(_p_parts)}")
            except Exception as exc:
                print(f"  [ram] read error: {exc}")

        # ── Two-stage decision: perceive → reason ───────────────────────────
        processed = process_screenshot(current_b64)
        scene = await _with_retry(lambda: perceive(vision_client, vision_model, processed))
        if scene:
            print(f"  [scene] {scene[:200].replace(chr(10), ' ')}")
        else:
            print("  [scene] EMPTY — vision model not describing the screen. Check that your model supports vision/image inputs.")

        # cur_map_id from RAM — initialized here so it's always defined even if
        # the try block below raises before reaching its internal assignment.
        cur_map_id: int | None = ram_state.get("map_id")

        # ── Nameplate detection: record location transition immediately ──────
        # The location nameplate in Gen 3 lasts ~60 frames. If the vision model
        # spotted it, log the new location to the world map before reasoning so
        # the reasoning model already sees it in the world map summary.
        try:
            scene_parsed = json.loads(scene)
            nameplate = scene_parsed.get("nameplate_text")
            if nameplate and isinstance(nameplate, str) and nameplate.strip():
                nameplate = nameplate.strip()
                world_map.update(nameplate, location_status="visited")
                print(f"  [nameplate] Entered: {nameplate}")
            # Use map_id from RAM as the primary location key for wall tracking.
            # VLM-guessed location_name creates hundreds of garbage entries like
            # "Unknown grassy area" — map_id is stable and accurate.
            cur_map_id = ram_state.get("map_id")
            if cur_map_id is not None:
                current_location = f"map_{cur_map_id}"
            else:
                # Fallback: use VLM location name only if it looks like a real
                # Pokemon location (contains Town, City, Route, Lab, etc.)
                _new_loc = scene_parsed.get("location_name", "") or ""
                if _new_loc and re.search(
                    r'\b(Town|City|Route|Lab|Center|Mart|Cave|Forest|Island|'
                    r'Mountain|Tower|Gym|League|Falls|Tunnel|Base)\b',
                    _new_loc, re.IGNORECASE
                ):
                    current_location = _best_location_key(world_map, _new_loc)
        except (json.JSONDecodeError, AttributeError):
            scene_parsed = {}

        # ── Screen type detection (Python-side mechanical correction) ────────
        _screen_type = scene_parsed.get("screen_type", "") if isinstance(scene_parsed, dict) else ""
        # Fix misclassification using menu_options keywords — objective, not model-dependent
        if isinstance(scene_parsed, dict):
            _mopts = [str(o).upper() for o in (scene_parsed.get("menu_options") or [])]
            if "FIGHT" in _mopts:
                if _screen_type != "battle":
                    print(f"  [fix] menu_options contains FIGHT → correcting screen_type to 'battle'")
                _screen_type = "battle"
                scene_parsed["screen_type"] = "battle"
            elif scene_parsed.get("dialogue_text") and not _mopts:
                if _screen_type != "dialogue":
                    print(f"  [fix] dialogue_text present → correcting screen_type to 'dialogue'")
                _screen_type = "dialogue"
                scene_parsed["screen_type"] = "dialogue"

        # RAM-based screen type correction — battle flag is ground truth
        if ram_state.get("in_battle") is True and _screen_type != "battle":
            print(f"  [ram-fix] gBattleTypeFlags non-zero → correcting screen_type to 'battle'")
            _screen_type = "battle"
            if isinstance(scene_parsed, dict):
                scene_parsed["screen_type"] = "battle"
        elif ram_state.get("in_battle") is False and _screen_type == "battle":
            print(f"  [ram-fix] gBattleTypeFlags is 0 → correcting screen_type from 'battle'")
            _screen_type = "overworld"
            if isinstance(scene_parsed, dict):
                scene_parsed["screen_type"] = "overworld"

        # Combine scene + RAM state for the reasoning model
        scene_with_ram = scene
        if ram_summary:
            scene_with_ram = scene + "\n\nGAME STATE (from RAM — accurate):\n" + ram_summary

        # Always call decide() so the model can update memory/goal/events.
        button, repeat, reason, event, new_goal, map_update, new_memory = await _with_retry(
            lambda: decide(
                reasoning_client, reasoning_model, scene_with_ram, history, system_prompt,
                current_goal=current_goal,
                stuck_hint=nav_hint,
                memory=memory,
                story_log=story_log,
                goal_log=goal_log,
                world_map_summary=world_map.summary(),
            )
        )

        # Force A during dialogue ONLY when there are no choices shown.
        # If menu_options are present (YES/NO etc.) let the model's button stand.
        if _screen_type == "dialogue" and not _mopts:
            button = "A"
            repeat = 1
            print(f"  [auto]  dialogue → forced A")
        # In battle: clamp repeat to 1 — menu navigation is single-step.
        if _screen_type == "battle" and repeat > 1:
            repeat = 1
            print(f"  [auto]  battle → clamped repeat to 1")
        if new_memory:
            memory = new_memory
            print(f"  [memory] {memory[:200]}")
        step_label = f"×{repeat}" if repeat > 1 else ""
        print(f"→ {button:6s}{step_label:4s}| {reason}")
        if event:
            # Deduplicate: skip if same as the last 1-2 story entries
            if not story_log or event not in story_log[-2:]:
                story_log.append(event)
                print(f"  [story] {event}")
        if new_goal and new_goal != current_goal:
            goal_log.append({"turn": turn, "goal": new_goal})
            current_goal = new_goal
            print(f"  [goal]  {current_goal}")
        if map_update and isinstance(map_update.get("location"), str) and map_update["location"]:
            _mu_loc = map_update["location"]
            # Only accept map updates with recognizable location names to prevent
            # VLM hallucinations like "Unknown grassy area" from polluting the map.
            if re.search(
                r'\b(Town|City|Route|Lab|Center|Mart|Cave|Forest|Island|'
                r'Mountain|Tower|Gym|League|Falls|Tunnel|Base|House)\b',
                _mu_loc, re.IGNORECASE
            ):
                world_map.update(
                    _mu_loc,
                    location_status=map_update.get("location_status") or None,
                    npc=map_update.get("npc") or None,
                    npc_status=map_update.get("npc_status") or None,
                    note=map_update.get("note") or None,
                )
                print(f"  [map]   {_mu_loc} \u2192 {map_update}")

        # ── Persist notes ────────────────────────────────────────────────────
        try:
            notes_path.write_text(
                json.dumps({"story_log": story_log, "current_goal": current_goal, "goal_log": goal_log, "memory": memory}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # non-fatal — next turn will retry

        # Update consecutive same-button counter and recent button history.
        if button == last_button:
            consecutive_same += 1
        else:
            consecutive_same = 1
            last_button = button
        recent_buttons.append(button)
        if len(recent_buttons) > 12:
            recent_buttons = recent_buttons[-12:]

        # Build a compact user-turn summary for history (screen_type + location only).
        # This keeps history valid (alternating user/assistant) without token bloat.
        try:
            _sp = json.loads(scene) if isinstance(scene, str) else {}
            _hist_user = f"screen={_sp.get('screen_type','?')} loc={_sp.get('location_name','?')}"
        except Exception:
            _hist_user = "screen=?"
        history.append({"role": "user", "content": _hist_user})
        history.append({
            "role": "assistant",
            "content": json.dumps({"button": button, "repeat": repeat}),
        })

        # Tap button (repeat times for directional moves) then capture screenshot.
        # Record screenshot hash BEFORE pressing to detect wall collisions after.
        # Adaptive settle frames: dialogue needs only ~3 frames; overworld movement
        # needs 8; cutscenes/transitions need 20+ to avoid reading mid-fade screens.
        if _screen_type == "dialogue" or button in {"A", "B", "Start", "Select"}:
            settle = 3
        elif _screen_type in {"cutscene", "unknown"}:
            settle = 20
        else:
            settle = SETTLE_FRAMES
        old_hash = _screenshot_hash(current_b64)
        if repeat > 1:
            next_b64 = await walk_steps(bridge, button, repeat, settle_frames=settle)
        else:
            next_b64 = await press_button(bridge, button, wait_frames=settle)
        if next_b64 is not None:
            current_b64 = next_b64
        else:
            current_b64 = await capture_screenshot(bridge)

        # Wall detection: directional press that left the screenshot unchanged = blocked.
        new_hash = _screenshot_hash(current_b64)
        wall_detected = button in {"Up", "Down", "Left", "Right"} and new_hash == old_hash
        wall_button = button if wall_detected else None
        if wall_detected:
            print(f"  [wall]  {button!r} blocked — warning VLM next turn")
            if current_location:
                world_map.record_wall(current_location, button)
                world_map.record_tested(current_location, button)
        elif button in {"Up", "Down", "Left", "Right"} and current_location:
            # Successful directional move — mark direction as tested for boundary scan.
            world_map.record_tested(current_location, button)

        # Map transition detection via RAM map_id (replaces expensive VLM nameplate check).
        # cur_map_id was already read from ram_state earlier in this turn.
        if cur_map_id is not None and last_map_id is not None and cur_map_id != last_map_id:
            print(f"  [map]   map_id changed: {last_map_id} → {cur_map_id}")
        if cur_map_id is not None:
            last_map_id = cur_map_id

        if max_turns and turn >= max_turns:
            print(f"[agent] Reached max_turns={max_turns}, stopping.")
            break

        await asyncio.sleep(0.05)


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
        help="Vision backend — handles screenshot → scene description (default: lmstudio)",
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
        help=(
            "Reasoning backend — handles scene description → button decision. "
            "Defaults to the same backend as --backend. "
            "Example: --backend lmstudio --reasoning-backend ollama"
        ),
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        metavar="MODEL",
        help=(
            "Override the reasoning model name. "
            "Can also be set via LMS_REASON_MODEL / OLLAMA_REASON_MODEL / OPENAI_REASON_MODEL in .env"
        ),
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

    vision_cfg = dict(BACKENDS[args.backend])
    if args.model:
        vision_cfg["model"] = args.model

    if args.backend == "openai" and not vision_cfg["api_key"]:
        print("ERROR: Set OPENAI_API_KEY in your environment or .env file.")
        sys.exit(1)

    # Reasoning backend — separate config when --reasoning-backend is given.
    if args.reasoning_backend and args.reasoning_backend != args.backend:
        reasoning_cfg: dict | None = dict(BACKENDS[args.reasoning_backend])
        if args.reasoning_model:
            reasoning_cfg["reasoning_model"] = args.reasoning_model
        if args.reasoning_backend == "openai" and not reasoning_cfg["api_key"]:
            print("ERROR: Set OPENAI_API_KEY in your environment or .env file.")
            sys.exit(1)
    else:
        # Same backend — just override reasoning_model if given
        reasoning_cfg = None
        if args.reasoning_model:
            vision_cfg["reasoning_model"] = args.reasoning_model

    asyncio.run(run_agent(
        rom=args.rom,
        session_id=args.session,
        backend_cfg=vision_cfg,
        game_profile=game_profile,
        max_turns=args.max_turns,
        mgba_path=args.mgba_path,
        reasoning_backend_cfg=reasoning_cfg,
    ))


if __name__ == "__main__":
    main()
