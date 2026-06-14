"""
mgba_agent/loop.py — Main agent turn loop.

run_agent() is the async entry point for a full game session. It handles:
  - Session management (new / resume / auto-resume)
  - mgba_launcher.lua generation and mGBA setup instructions
  - Heartbeat polling until the bridge is ready
  - Per-turn: RAM read → nav hints → perceive → decide → press → wall detection
  - Persistent notes (story_log, goal_log, memory) saved to notes.json each turn
  - Autosave every AUTOSAVE_EVERY_N_TURNS turns
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import (
    AUTOSAVE_EVERY_N_TURNS,
    SETTLE_FRAMES,
    STUCK_BUTTON_THRESHOLD,
)
from .bridge.client import BridgeClient
from .bridge.emulator import (
    capture_screenshot,
    press_button,
    save_game,
    walk_steps,
)
from .llm.decide import decide
from .llm.retry import with_retry
from .navigation.world_map import WorldMap, best_location_key
from .ram.reader import GameState
from .vision.perceive import perceive, process_screenshot, screenshot_hash


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
    # Three paths:
    #   A) --session <id> supplied → resume that specific session
    #   B) no --session but a recent heartbeat (<5 min) found → offer auto-resume
    #   C) neither → new session (generate launcher, wait for mGBA)

    def _validate_session(sid: str) -> Path:
        """Return ipc_dir if heartbeat exists, else raise."""
        d = runtime_dir / sid
        if not (d / "heartbeat.json").exists():
            raise RuntimeError(
                f"No heartbeat.json found for session {sid}. "
                "Is mGBA still running with the bridge script loaded?"
            )
        return d

    _new_session = False
    if session_id:
        # Path A: explicit --session
        ipc_dir = _validate_session(session_id)
        print(f"[agent] Resuming session {session_id}")
    else:
        # Path B: auto-resume scan
        _resume_candidate: str | None = None
        if runtime_dir.exists():
            _cutoff = time.time() - 300  # 5 minutes
            _candidates: list[tuple[float, str]] = [
                ((_d / "heartbeat.json").stat().st_mtime, _d.name)
                for _d in runtime_dir.iterdir()
                if _d.is_dir() and (_d / "heartbeat.json").exists()
                and (_d / "heartbeat.json").stat().st_mtime >= _cutoff
            ]
            if _candidates:
                _candidates.sort(reverse=True)
                _resume_candidate = _candidates[0][1]

        if _resume_candidate:
            print(f"[agent] Found recent session: {_resume_candidate}")
            print("[agent] Press Enter to resume, 'n' for new session, or type a session ID: ", end="", flush=True)
            _ans = sys.stdin.readline().strip()
            if _ans.lower() in ("", "y", "yes"):
                session_id = _resume_candidate
            elif _ans.lower() in ("n", "no", "new"):
                session_id = None  # fall through to new session
            else:
                session_id = _ans  # user typed a custom session ID

        if session_id:
            # Resume (either auto-candidate or user-supplied custom ID)
            ipc_dir = _validate_session(session_id)
            print(f"[agent] Resuming session {session_id}")
        else:
            # Path C: brand-new session
            _new_session = True
            session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ipc_dir = runtime_dir / session_id
            ipc_dir.mkdir(parents=True, exist_ok=True)

    if _new_session:
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
        scene = await with_retry(lambda: perceive(vision_client, vision_model, processed))
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
                    current_location = best_location_key(world_map, _new_loc)
        except (json.JSONDecodeError, AttributeError):
            scene_parsed = {}

        # ── Screen type detection (Python-side mechanical correction) ────────
        _screen_type = scene_parsed.get("screen_type", "") if isinstance(scene_parsed, dict) else ""
        # Fix misclassification using menu_options keywords — objective, not model-dependent
        _mopts: list[str] = []
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
        button, repeat, reason, event, new_goal, map_update, new_memory = await with_retry(
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
        old_hash = screenshot_hash(current_b64)
        if repeat > 1:
            next_b64 = await walk_steps(bridge, button, repeat, settle_frames=settle)
        else:
            next_b64 = await press_button(bridge, button, wait_frames=settle)
        if next_b64 is not None:
            current_b64 = next_b64
        else:
            current_b64 = await capture_screenshot(bridge)

        # Wall detection: directional press that left the screenshot unchanged = blocked.
        new_hash = screenshot_hash(current_b64)
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
