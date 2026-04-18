"""
pyboy_agent.loop
================
Main agent loop — wires all modules together into a single ``run_agent()`` call.

This module is intentionally short.  All domain logic lives in the submodules
it imports.  The loop's job is sequencing: read RAM → build hints → perceive →
decide → press button → detect wall → record → repeat.

Session state that lives here
------------------------------
- ``turn``                  : monotonically increasing turn counter
- ``history``               : last N message pairs for the reasoning model
- ``last_button``           : previous turn's button (for stuck detection)
- ``consecutive_same``      : how many turns in a row the same button was pressed
- ``consecutive_a``         : how many consecutive A presses
- ``wall_detected``         : was the previous directional press blocked?
- ``wall_button``           : which direction was blocked (or None)
- ``_last_tile``            : RAM (x,y) from last turn (for stuck-tile counter)
- ``_last_map``             : RAM (bank, number) from last turn
- ``turns_at_same_tile``    : turns spent on the same tile without moving
- ``_hp_valid_turns``       : turns HP has read a non-zero hp_max (stabilisation)

Autosave
---------
Every ``AUTOSAVE_EVERY_N_TURNS`` turns the game-profile's ``save_sequence`` is
executed and a PyBoy state snapshot is written next to the ROM.  Autosave is
skipped when RAM confirms the player is in a battle (to avoid pressing the save
sequence into the battle menu).

Operator override
------------------
Two channels for real-time human input:
1. Lines typed into the terminal (daemon stdin-reader thread → ``_stdin_queue``)
2. A drop file at ``./agent_message.txt`` (read and deleted each turn)

Both are merged into ``operator_msg`` and injected as the highest-priority
navigation hint.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from pyboy_agent.config import (
    AUTOSAVE_EVERY_N_TURNS,
    SETTLE_FRAMES_BUTTON,
    SETTLE_FRAMES_CUTSCENE,
    SETTLE_FRAMES_MOVE,
    STUCK_TILE_THRESHOLD,
)
from pyboy_agent.emulator import (
    capture_screenshot,
    create_pyboy,
    press_button,
    save_game,
    walk_steps,
)
from pyboy_agent.llm.decide import decide
from pyboy_agent.llm.retry import with_retry
from pyboy_agent.navigation.hints import build_nav_hints
from pyboy_agent.navigation.wall_tracker import detect_and_record_wall
from pyboy_agent.navigation.world_map import WorldMap, best_location_key
from pyboy_agent.ram.formatter import format_ram_state
from pyboy_agent.ram.reader import read_ram_state
from pyboy_agent.vision.perceive import perceive


def run_agent(
    rom: str,
    game_profile: dict,
    vision_client: OpenAI,
    vision_model: str,
    reasoning_client: OpenAI,
    reasoning_model: str,
    *,
    headless: bool = False,
    max_turns: int = 0,
    state_file: str | None = None,
    speed: int | None = None,
    is_local_vision: bool = False,
    is_local_reason: bool = False,
    on_auth_error: Any = None,
) -> None:
    """Run the agent loop until stopped (Ctrl+C or max_turns).

    Args:
        rom: Absolute path to the ROM file.
        game_profile: Loaded game profile dict (from ``profiles.load_game_profile``).
        vision_client: OpenAI-compatible client for the vision model.
        vision_model: Vision model name string.
        reasoning_client: OpenAI-compatible client for the reasoning model.
        reasoning_model: Reasoning model name string.
        headless: Run without SDL2 window (null window, max speed).
        max_turns: Stop after this many turns (0 = run forever).
        state_file: Path to a PyBoy state file to load on startup.
        speed: Emulation speed multiplier (0=unlimited, 1=realtime).
        is_local_vision: True if the vision backend is on localhost (enables thinking).
        is_local_reason: True if the reasoning backend is on localhost.
        on_auth_error: Callback called on 401 Unauthorized (for Copilot token refresh).
    """
    system_prompt:  str = game_profile["system_prompt"]
    save_sequence:  list[str] | None = game_profile.get("save_sequence")
    game_name:      str = game_profile.get("name", "GBC Game")
    ram_offsets:    dict = game_profile.get("ram_offsets", {})
    has_ram = bool({k for k in ram_offsets if k != "note"})

    # Extra-body for local backends that support enable_thinking.
    _reason_extra = {"enable_thinking": True} if is_local_reason else None
    _vision_extra = {"enable_thinking": True} if is_local_vision else None

    # ── File paths ────────────────────────────────────────────────────────
    rom_path = Path(rom)
    notes_path       = rom_path.with_suffix(rom_path.suffix + ".pyboy_agent_notes.json")
    state_snap_path  = rom_path.with_suffix(rom_path.suffix + ".pyboy_agent.state")
    shots_dir        = rom_path.parent / (rom_path.stem + "_shots")
    _message_file    = Path("agent_message.txt")

    # ── Persistent notes ──────────────────────────────────────────────────
    from pyboy_agent.goals.tracker import NotesTracker
    notes = NotesTracker(notes_path, initial_goal=game_profile.get("initial_goal", ""))

    # ── World map ─────────────────────────────────────────────────────────
    game_slug = game_name.lower().replace(" ", "-")
    world_map = WorldMap(game_slug)
    print(
        f"[agent] World map: {len(world_map.data.get('locations', {}))} location(s) — "
        f"{world_map.path}"
    )

    # ── Emulator startup ──────────────────────────────────────────────────
    print(f"[agent] Game={game_name} | vision={vision_model} | reason={reasoning_model}")
    print(f"[agent] ROM : {rom}")

    pyboy = create_pyboy(rom, headless=headless, speed=speed)

    # Load state snapshot.
    if state_file:
        sf = Path(state_file)
        if sf.exists():
            with open(sf, "rb") as f:
                pyboy.load_state(f)
            print(f"[agent] Loaded state from: {sf}")
        else:
            print(f"[agent] WARNING: state file not found: {sf} — starting fresh")
    elif state_snap_path.exists():
        try:
            with open(state_snap_path, "rb") as f:
                pyboy.load_state(f)
            print(f"[agent] Auto-resumed from snapshot: {state_snap_path}")
        except Exception as exc:
            print(f"[agent] Could not load auto-state ({exc}) — starting from .sav")

    # Let the game settle after load (boot screen / intro).
    pyboy.tick(60, render=True)
    current_b64 = capture_screenshot(pyboy, shots_dir=shots_dir)

    # pump_fn keeps SDL2 events flowing during VLM calls in windowed mode.
    pump_fn = (lambda: pyboy.tick(1, render=True)) if not headless else None

    # ── Stdin operator-override reader ────────────────────────────────────
    _stdin_queue: queue.Queue[str] = queue.Queue()

    def _stdin_reader() -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    _stdin_queue.put(line)
        except Exception:
            pass

    _stdin_thread = threading.Thread(target=_stdin_reader, daemon=True)
    _stdin_thread.start()
    print(
        f"[agent] Override: type a message here and press Enter, "
        f"or write to '{_message_file.resolve()}'."
    )

    # ── Helper closures ───────────────────────────────────────────────────
    def flush_state() -> None:
        try:
            with open(state_snap_path, "wb") as f:
                pyboy.save_state(f)
        except Exception:
            pass

    def shutdown(save: bool = True) -> None:
        notes.flush()
        flush_state()
        pyboy.stop(save=save)
        print("[agent] Stopped.")

    _stop_requested = [False]

    def _sigint_handler(signum: int, frame: Any) -> None:
        print("\n[agent] Ctrl+C received — shutting down gracefully…")
        _stop_requested[0] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Turn-level state ──────────────────────────────────────────────────
    history:           list[dict] = []
    turn               = 0
    start_time         = time.time()
    last_button:       str | None = None
    consecutive_same   = 0
    consecutive_a      = 0
    wall_detected      = False
    wall_button:       str | None = None
    current_location   = ""
    _last_tile:        tuple[int, int] | None = None
    _last_map:         tuple[int, int] | None = None
    turns_at_same_tile = 0
    _tile_key          = ""
    _hp_valid_turns    = 0
    _ram_state:        dict[str, Any] = {}

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            if _stop_requested[0]:
                break

            turn += 1
            elapsed = int(time.time() - start_time)
            print(f"\n[turn {turn:04d} | {elapsed//60:02d}:{elapsed%60:02d}]", end=" ")

            # ── Operator override ─────────────────────────────────────────
            _operator_msg: str | None = None
            _stdin_lines: list[str] = []
            while not _stdin_queue.empty():
                try:
                    _stdin_lines.append(_stdin_queue.get_nowait())
                except Exception:
                    break
            if _stdin_lines:
                _operator_msg = " | ".join(_stdin_lines)
            if _message_file.exists():
                try:
                    _file_msg = _message_file.read_text(encoding="utf-8").strip()
                    _message_file.unlink()
                    if _file_msg:
                        _operator_msg = (
                            (_operator_msg + " | " + _file_msg) if _operator_msg else _file_msg
                        )
                except Exception:
                    pass
            if _operator_msg:
                print(f"\n  [operator] *** OVERRIDE: {_operator_msg} ***")

            # ── Autosave ──────────────────────────────────────────────────
            if save_sequence and turn > 1 and turn % AUTOSAVE_EVERY_N_TURNS == 0:
                _in_battle = _ram_state.get("in_battle", False)
                if _in_battle:
                    print(f"  [autosave] Skipping turn {turn} — in battle")
                else:
                    current_b64 = save_game(pyboy, save_sequence)
                    flush_state()

            # ── Read RAM ──────────────────────────────────────────────────
            _ram_state = read_ram_state(pyboy, ram_offsets) if has_ram else {}
            if _ram_state:
                _cur_hp_max = _ram_state.get("lead_hp_max") or 0
                _hp_valid_turns = (_hp_valid_turns + 1) if _cur_hp_max > 0 else 0
                _ram_state["hp_stabilised"] = _hp_valid_turns >= 2
            _ram_text: str | None = format_ram_state(_ram_state) if _ram_state else None

            # ── Position tracking ─────────────────────────────────────────
            _mb = _ram_state.get("map_bank")
            _mn = _ram_state.get("map_number")
            _cx = _ram_state.get("x_pos")
            _cy = _ram_state.get("y_pos")
            _has_pos = all(v is not None for v in (_mb, _mn, _cx, _cy))

            if _has_pos:
                _cur_map  = (_mb, _mn)
                _cur_tile = (_cx, _cy)
                _tile_key = f"map_{_mb}_{_mn}_x{_cx}_y{_cy}"
                if _cur_map != _last_map:
                    print(f"  [map] {_last_map} -> {_cur_map} | tile ({_cx},{_cy})")
                    _last_map  = _cur_map
                    _last_tile = _cur_tile
                    turns_at_same_tile = 0
                elif _cur_tile != _last_tile:
                    turns_at_same_tile = 0
                    _last_tile = _cur_tile
                else:
                    turns_at_same_tile += 1
            else:
                _tile_key = current_location

            # ── Log RAM summary ───────────────────────────────────────────
            if _ram_state:
                _hp_cur = _ram_state.get("lead_hp_current")
                _hp_max = _ram_state.get("lead_hp_max")
                _hp_pct = _ram_state.get("lead_hp_pct")
                _badges = _ram_state.get("johto_badge_count", 0)
                _money  = _ram_state.get("money")
                _party  = _ram_state.get("party_count")
                _hp_s   = _ram_state.get("hp_stabilised", False)
                _hp_str = (
                    f"HP={_hp_cur}/{_hp_max}({_hp_pct}%)" if _hp_s
                    else ("HP=n/a" if not _hp_max else "HP=stabilising…")
                )
                _money_str = f" | ¥{_money:,}" if _money is not None else ""
                _party_str = f" | party={_party}" if _party is not None else ""
                print(
                    f"  [ram]   map={_mb}/{_mn} pos=({_cx},{_cy}) | "
                    f"{_hp_str} | badges={_badges}/8{_party_str}{_money_str}"
                )
                if _hp_s and _hp_pct is not None and _hp_pct < 25:
                    print("  [ram]   ⚠ LOW HP — agent should seek a Pokemon Center")
                if _party == 0:
                    print("  [ram]   ⚠ PARTY EMPTY — agent must get starter from Prof. Elm")

            # ── Wall-reset guard ──────────────────────────────────────────
            # If all 4 directions are confirmed walls at the current tile,
            # this is physically impossible — a cutscene or invisible dialogue
            # is likely freezing movement.  Press B×5 and clear the tile data.
            _wlk = _tile_key or current_location
            known_walls = world_map.get_walls(_wlk) if _wlk else set()
            _cardinal = {"Up", "Down", "Left", "Right"}

            if known_walls >= _cardinal:
                print(
                    f"  [wall-reset] All 4 directions blocked at tile {_wlk} — "
                    "dialogue freeze suspected. Pressing B×5 then clearing walls."
                )
                for _ in range(5):
                    press_button(pyboy, "B", SETTLE_FRAMES_BUTTON, shots_dir=shots_dir)
                press_button(pyboy, "A", SETTLE_FRAMES_BUTTON, shots_dir=shots_dir)
                if _wlk:
                    world_map.clear_walls(_wlk)
                known_walls = set()

            # ── Stuck-tile auto-recovery ──────────────────────────────────
            if turns_at_same_tile >= STUCK_TILE_THRESHOLD and turns_at_same_tile % 4 == 0:
                print(
                    f"  [stuck-recovery] {turns_at_same_tile} turns at same tile — "
                    "pressing B×3 to clear frozen state"
                )
                for _ in range(3):
                    press_button(pyboy, "B", SETTLE_FRAMES_BUTTON, shots_dir=shots_dir)

            # ── RAM fast-paths (skip VLM for obvious states) ──────────────
            if has_ram and _ram_state:
                # Dialogue open → advance immediately with A.
                if _ram_state.get("dialogue_open"):
                    _ram_fast_press(
                        pyboy, "A", "RAM->dialogue",
                        consecutive_same, consecutive_a, last_button,
                        history, shots_dir=shots_dir,
                    )
                    current_b64, button, consecutive_same, consecutive_a, last_button = (
                        press_button(pyboy, "A", SETTLE_FRAMES_BUTTON, shots_dir=shots_dir),
                        "A", consecutive_same + 1 if last_button == "A" else 1,
                        consecutive_a + 1, "A",
                    )
                    continue

                # Menu open (not battle) → close with B.
                if _ram_state.get("menu_open") and not _ram_state.get("in_battle"):
                    current_b64 = press_button(pyboy, "B", SETTLE_FRAMES_BUTTON, shots_dir=shots_dir)
                    button = "B"
                    consecutive_same = consecutive_same + 1 if last_button == "B" else 1
                    consecutive_a = 0
                    last_button = "B"
                    print("  [auto]  RAM->menu_open -> forced B (skip VLM)")
                    continue

                # In battle on odd turns → advance/confirm with A.
                if _ram_state.get("in_battle") and turn % 2 == 1:
                    current_b64 = press_button(pyboy, "A", SETTLE_FRAMES_BUTTON, shots_dir=shots_dir)
                    button = "A"
                    consecutive_same = consecutive_same + 1 if last_button == "A" else 1
                    consecutive_a += 1
                    last_button = "A"
                    print("  [auto]  RAM->battle -> forced A (skip VLM)")
                    continue

            # ── Vision model call ─────────────────────────────────────────
            # RAM-first policy: only call vision when needed.
            need_vision = True
            if has_ram and _ram_state:
                if wall_detected or _operator_msg or not current_location or turn % 5 == 0:
                    need_vision = True
                elif _ram_state.get("in_battle"):
                    need_vision = (turn % 2 == 0)
                else:
                    need_vision = False

            if need_vision:
                scene = with_retry(
                    lambda: perceive(
                        vision_client, vision_model, current_b64,
                        extra_body=_vision_extra,
                    ),
                    pump_fn=pump_fn,
                    on_auth_error=on_auth_error,
                )
            else:
                scene = "{}"

            # Parse and log scene.
            scene_parsed: dict = {}
            _dialogue_text: str | None = None

            if scene:
                try:
                    scene_parsed = json.loads(scene)
                    # Normalise direction capitalisation.
                    if isinstance(scene_parsed.get("passable_directions"), list):
                        scene_parsed["passable_directions"] = [
                            d.title() for d in scene_parsed["passable_directions"]
                            if isinstance(d, str)
                        ]
                    if isinstance(scene_parsed.get("player_facing"), str):
                        scene_parsed["player_facing"] = scene_parsed["player_facing"].title()

                    # Location tracking.
                    _new_loc = scene_parsed.get("location_name", "") or ""
                    if _new_loc:
                        current_location = best_location_key(world_map, _new_loc)

                    # Nameplate.
                    nameplate = scene_parsed.get("nameplate_text")
                    if nameplate and isinstance(nameplate, str) and nameplate.strip():
                        world_map.update(nameplate.strip(), location_status="visited")
                        print(f"  [nameplate] Entered: {nameplate.strip()}")

                    # Dialogue text.
                    _raw_dlg = scene_parsed.get("dialogue_text")
                    if _raw_dlg and isinstance(_raw_dlg, str) and _raw_dlg.strip():
                        _dialogue_text = _raw_dlg.strip()
                        print(f'  [dialogue] "{_dialogue_text}"')

                    # Brief scene log.
                    _s = scene_parsed
                    print(
                        f"  [scene] {_s.get('screen_type','?')} | "
                        f"{'outdoor' if _s.get('is_outdoor') else 'indoor'} | "
                        f"loc={_s.get('location_name','?')} | "
                        f"facing={_s.get('player_facing','?')}"
                    )
                    if _s.get("battle_info"):
                        print(f"  [battle] {_s['battle_info']}")
                except (json.JSONDecodeError, TypeError):
                    print(f"  [scene/parse-fail] raw ({len(scene)} chars): {scene[:200]}")

            # When RAM position is available, strip passable_directions from the
            # scene JSON — VLM values are frequently wrong, RAM delta is authoritative.
            _scene_for_decide = scene
            if _has_pos and scene_parsed and isinstance(scene_parsed, dict):
                try:
                    _scene_for_decide = json.dumps(
                        {k: v for k, v in scene_parsed.items() if k != "passable_directions"}
                    )
                except Exception:
                    pass

            # ── Navigation hints ──────────────────────────────────────────
            nav_hint = build_nav_hints(
                operator_msg=_operator_msg,
                wall_detected=wall_detected,
                wall_button=wall_button,
                last_button=last_button,
                consecutive_same=consecutive_same,
                consecutive_a=consecutive_a,
                tile_key=_tile_key,
                has_pos=_has_pos,
                cx=_cx,
                cy=_cy,
                map_bank=_mb,
                map_number=_mn,
                turns_at_same_tile=turns_at_same_tile,
                world_map=world_map,
                ram_state=_ram_state,
                scene_parsed=scene_parsed,
                current_location=current_location,
                memory=notes.memory,
            )

            # ── Decide ────────────────────────────────────────────────────
            (
                button,
                repeat,
                reason,
                event,
                new_goal,
                map_update,
                new_memory,
                _thinking,
            ) = with_retry(
                lambda: decide(
                    reasoning_client,
                    reasoning_model,
                    _scene_for_decide,
                    history,
                    system_prompt,
                    current_goal=notes.current_goal,
                    stuck_hint=nav_hint or None,
                    memory=notes.memory,
                    story_log=notes.story_log,
                    goal_log=notes.goal_log,
                    world_map_summary=world_map.summary(),
                    dialogue_text=_dialogue_text,
                    extra_body=_reason_extra,
                    ram_state_text=_ram_text,
                ),
                pump_fn=pump_fn,
                on_auth_error=on_auth_error,
            )

            # ── Auto-overrides ────────────────────────────────────────────
            _screen_type = scene_parsed.get("screen_type", "")
            _retalk_fired = nav_hint and "already marked 'talked'" in nav_hint

            if _screen_type == "dialogue" and not _retalk_fired:
                button = "A"
                repeat = 1
                print("  [auto]  dialogue -> forced A")
            elif _screen_type == "dialogue" and _retalk_fired:
                if button == "A":
                    button = "Down"
                    repeat = 1
                    print("  [auto]  retalk+dialogue -> forced Down (walk away)")
            if _screen_type == "battle" and repeat > 1:
                repeat = 1
                print("  [auto]  battle -> clamped repeat to 1")

            step_label = f"×{repeat}" if repeat > 1 else ""
            print(f"-> {button:6s}{step_label:4s}| {reason}")

            # ── Update notes ──────────────────────────────────────────────
            if event:
                notes.append_event(event)
            if new_goal:
                notes.update_goal(new_goal, turn)
            if new_memory:
                notes.update_memory(new_memory)
            if map_update and isinstance(map_update.get("location"), str) and map_update["location"]:
                world_map.update(
                    map_update["location"],
                    location_status=map_update.get("location_status") or None,
                    npc=map_update.get("npc") or None,
                    npc_status=map_update.get("npc_status") or None,
                    note=map_update.get("note") or None,
                )
                print(f"  [map]   {map_update['location']} -> {map_update}")

            notes.flush()

            # ── Consecutive counters ──────────────────────────────────────
            if button == last_button:
                consecutive_same += 1
            else:
                consecutive_same = 1
                last_button = button
            consecutive_a = (consecutive_a + 1) if button == "A" else 0

            # ── History ───────────────────────────────────────────────────
            try:
                _sp = json.loads(scene) if isinstance(scene, str) else {}
                _h_type   = _sp.get("screen_type", "?")
                _h_loc    = _sp.get("location_name", "?")
                _h_facing = _sp.get("player_facing", "?")
                _h_dlg    = f' | dialogue="{_sp["dialogue_text"][:40]}"' if _sp.get("dialogue_text") else ""
                _h_npc    = f' | adj_npc={_sp["adjacent_npc_id"]}' if _sp.get("adjacent_npc") else ""
                _h_ram    = (
                    f' | HP={_ram_state.get("lead_hp_pct")}% badges={_ram_state.get("johto_badge_count")}/8'
                    if _ram_state else ""
                )
                _hist_user = f"screen={_h_type} | loc={_h_loc} | facing={_h_facing}{_h_dlg}{_h_npc}{_h_ram}"
            except Exception:
                _hist_user = "screen=?"

            history.append({"role": "user", "content": _hist_user})
            history.append({
                "role": "assistant",
                "content": json.dumps({
                    "button": button,
                    "repeat": repeat,
                    "reason": reason[:80] if reason else "",
                }),
            })

            # ── Execute button press ──────────────────────────────────────
            if _screen_type == "dialogue" or button in {"A", "B", "Start", "Select"}:
                settle = SETTLE_FRAMES_BUTTON
            elif _screen_type in {"cutscene", "unknown"}:
                settle = SETTLE_FRAMES_CUTSCENE
            else:
                settle = SETTLE_FRAMES_MOVE

            old_b64 = current_b64

            # Save pre-move map values for warp detection.
            _pre_mb = _mb
            _pre_mn = _mn

            if button in {"Up", "Down", "Left", "Right"} and repeat > 1:
                current_b64 = walk_steps(pyboy, button, repeat, settle, shots_dir=shots_dir)
            else:
                current_b64 = press_button(pyboy, button, settle, shots_dir=shots_dir)

            # ── Wall detection ────────────────────────────────────────────
            wall_detected, wall_button = detect_and_record_wall(
                pyboy,
                button=button,
                old_screenshot_b64=old_b64,
                new_screenshot_b64=current_b64,
                pre_ram_state=_ram_state,
                has_ram=has_ram,
                ram_offsets=ram_offsets,
                tile_key=_tile_key,
                world_map=world_map,
                pre_map_bank=_pre_mb,
                pre_map_number=_pre_mn,
            )

            if max_turns and turn >= max_turns:
                print(f"[agent] Reached max_turns={max_turns}, stopping.")
                break

    finally:
        shutdown(save=True)


def _ram_fast_press(
    pyboy: Any,
    button: str,
    label: str,
    consecutive_same: int,
    consecutive_a: int,
    last_button: str | None,
    history: list[dict],
    *,
    shots_dir: Any = None,
) -> None:
    """Log a RAM-shortcut button press into history (used by fast-path branches)."""
    reason = f"RAM-shortcut: {label}"
    print(f"  [auto]  {label} -> forced {button} (skip VLM)")
    try:
        history.append({"role": "user", "content": f"screen=? | {label}"})
        history.append({
            "role": "assistant",
            "content": json.dumps({"button": button, "repeat": 1, "reason": reason[:80]}),
        })
    except Exception:
        pass
