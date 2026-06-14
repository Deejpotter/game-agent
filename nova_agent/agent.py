"""
nova_agent.agent
================
Main agent loop — single-model tool-calling architecture.

Architecture summary
--------------------
Each turn:
  1. Read RAM → detect screen type → capture screenshot with overlay
  2. Build prompt: system + KB + screen hint + recent events
  3. Call LLM (single vision-capable model) with tools defined
  4. Dispatch tool calls in order returned by model
  5. Post-move: update tile graph from RAM delta (if a directional was pressed)
  6. Autosave every N turns
  7. Summarize history when it grows too long

Key design decisions vs pyboy_agent:
  - One LLM call per turn (not two — perceive was removed)
  - Model sees screenshot directly (not a JSON scene description)
  - Tools replace the rigid JSON return schema
  - Tile graph replaces 12-source nav hints
  - KB replaces free-form memory string
  - Summarization handles long context (not just capping history)

Stdin operator override still works: type into the terminal during a run
to inject a message as the next user turn.
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

from nova_agent.config import (
    AUTOSAVE_EVERY_N_TURNS,
    BUTTON_MAP,
    MAX_HISTORY_MESSAGES,
    SETTLE_FRAMES_CUTSCENE,
)
from nova_agent.emulator import (
    capture_frame,
    create_pyboy,
    load_state,
    press_sequence,
    save_state,
)
from nova_agent.memory import KnowledgeBase
from nova_agent.pathfinder import TileGraph, TileKey
from nova_agent.state import (
    SCREEN_HINTS,
    ScreenType,
    detect_screen_type,
    read_ram,
)
from nova_agent.summarizer import summarize_and_reset
from nova_agent.tools import TOOLS, ToolDispatcher


# ---------------------------------------------------------------------------
# Backend / token helpers
# ---------------------------------------------------------------------------

def _load_copilot_token() -> str:
    """Load a GitHub Copilot token from the VS Code credential store."""
    try:
        import subprocess, json as _json
        result = subprocess.run(
            ["node", "-e",
             "const {execSync}=require('child_process');"
             "const out=execSync('gh auth token').toString().trim();"
             "console.log(out)"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return os.getenv("GITHUB_TOKEN", "")


def _make_client(backend: str, backends: dict[str, Any]) -> tuple[OpenAI, str, bool]:
    """Return (client, model_name, is_local)."""
    cfg = backends.get(backend)
    if cfg is None:
        raise ValueError(f"Unknown backend: {backend!r}")

    api_key = cfg["api_key"]
    if api_key == "_copilot_":
        api_key = _load_copilot_token()

    client = OpenAI(base_url=cfg["base_url"], api_key=api_key or "no-key")
    model = cfg["model"]
    is_local = "localhost" in (cfg.get("base_url") or "")
    return client, model, is_local


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_agent(
    rom: str,
    game_profile: dict[str, Any],
    client: OpenAI,
    model: str,
    *,
    headless: bool = False,
    max_turns: int = 0,
    state_file: str | None = None,
    speed: int | None = None,
    is_local: bool = False,
) -> None:
    """Run the nova_agent loop until Ctrl+C or max_turns.

    Args:
        rom: Path to the ROM file.
        game_profile: Loaded game profile dict.
        client: OpenAI-compatible client (single model for vision + reasoning).
        model: Model name string.
        headless: Run without SDL2 window.
        max_turns: Stop after this many turns (0 = forever).
        state_file: Optional path to a PyBoy state file to load on startup.
        speed: Emulation speed multiplier.
        is_local: True if the backend is a local server (enables thinking).
    """
    _base_prompt: str = game_profile["system_prompt"]
    save_sequence: list[str] = game_profile.get("save_sequence", [])
    game_name: str = game_profile.get("name", "Unknown")
    ram_offsets: dict = game_profile.get("ram_offsets", {})
    has_ram = bool({k for k in ram_offsets if k != "note"})

    # Prepend tool-calling preamble so the model always uses tools instead of
    # returning raw JSON text (the old pyboy_agent schema).
    _TOOL_PREAMBLE = """\
You control a Game Boy Color game by calling tools. NEVER output raw JSON or text \
as your response — always use one or more tool calls to act.

Available tools:
  press_buttons   — press a sequence of buttons (e.g. "A", "Right Right Down A")
  navigate_to     — walk to a specific tile coordinate using learned pathfinding
  update_knowledge — update a section of your persistent knowledge base
  add_event        — record a notable story event
  set_goal         — update your current short-term goal

You MUST call at least one tool every turn. Do NOT respond with plain text or JSON.
"""
    system_prompt = _TOOL_PREAMBLE + "\n" + _base_prompt

    extra_body: dict | None = {"enable_thinking": True} if is_local else None

    # ── File paths ────────────────────────────────────────────────────────
    rom_path = Path(rom)
    slug = game_name.lower().replace(" ", "-")
    kb_path      = rom_path.with_suffix(rom_path.suffix + ".nova_kb.json")
    state_snap   = rom_path.with_suffix(rom_path.suffix + ".nova.state")
    graph_path   = rom_path.with_suffix(rom_path.suffix + ".nova_graph.json")
    train_log    = rom_path.with_suffix(rom_path.suffix + ".nova_train.jsonl")

    # ── Persistent state ──────────────────────────────────────────────────
    kb = KnowledgeBase(kb_path, initial_goal=game_profile.get("initial_goal", ""))
    tile_graph = TileGraph(graph_path)

    # ── Emulator startup ──────────────────────────────────────────────────
    print(f"[nova] Game={game_name} | model={model} | backend={'local' if is_local else 'remote'}")
    print(f"[nova] ROM : {rom}")
    pyboy = create_pyboy(rom, headless=headless, speed=speed)

    if state_file and load_state(pyboy, state_file):
        print(f"[nova] Loaded state from: {state_file}")
    elif load_state(pyboy, state_snap):
        print(f"[nova] Auto-resumed from snapshot: {state_snap}")

    # Let the game settle.
    pyboy.tick(60, render=not headless)

    pump_fn = (lambda: pyboy.tick(1, render=True)) if not headless else None

    # ── Operator override via stdin ───────────────────────────────────────
    _stdin_q: queue.Queue[str] = queue.Queue()

    def _stdin_reader() -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    _stdin_q.put(line)
        except Exception:
            pass

    threading.Thread(target=_stdin_reader, daemon=True).start()
    print("[nova] Type a message and press Enter to override the agent's next action.")

    # ── Shutdown handler ──────────────────────────────────────────────────
    _stop = threading.Event()

    def _handle_sigint(sig, frame):  # noqa: ANN001
        print("\n[nova] Interrupted — saving state…")
        save_state(pyboy, state_snap)
        _stop.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Conversation history ──────────────────────────────────────────────
    history: list[dict[str, Any]] = []
    turn = 0

    print(f"[nova] Starting loop (max_turns={max_turns or 'unlimited'})")

    while not _stop.is_set():
        if max_turns and turn >= max_turns:
            print(f"[nova] Reached max_turns={max_turns}")
            break

        turn += 1
        print(f"\n─── Turn {turn} ───")

        # ── 1. RAM read ───────────────────────────────────────────────────
        ram_state: dict[str, Any] = {}
        if has_ram:
            ram_state = read_ram(pyboy, ram_offsets)
            # Store offsets inside state so navigate_to can re-read RAM mid-path.
            ram_state["_offsets"] = ram_offsets

        screen_type = detect_screen_type(ram_state) if has_ram else ScreenType.UNKNOWN

        # ── 2. Screenshot with overlay ────────────────────────────────────
        frame_b64 = capture_frame(pyboy, ram_state if has_ram else None)

        # ── 3. Check for operator override message ────────────────────────
        operator_msg: str | None = None
        try:
            operator_msg = _stdin_q.get_nowait()
            print(f"[nova] Operator: {operator_msg}")
        except queue.Empty:
            pass

        # Also check drop file.
        msg_file = Path("nova_message.txt")
        if msg_file.exists():
            try:
                operator_msg = msg_file.read_text(encoding="utf-8").strip()
                msg_file.unlink()
                print(f"[nova] Drop file: {operator_msg}")
            except Exception:
                pass

        # ── 4. Build user message ─────────────────────────────────────────
        screen_hint = SCREEN_HINTS.get(screen_type, "")
        kb_block = kb.to_prompt_block()
        recent = kb.recent_events(12)
        events_text = "\n".join(f"  • {e}" for e in recent) if recent else "(none)"

        # RAM facts summary (brief — full data is on the overlay).
        ram_summary = _format_ram_summary(ram_state) if has_ram else ""

        user_text_parts = []
        if ram_summary:
            user_text_parts.append(ram_summary)
        if screen_hint:
            user_text_parts.append(f"SCREEN STATE: {screen_hint}")
        if operator_msg:
            user_text_parts.append(
                f"⚠ OPERATOR OVERRIDE: {operator_msg}\n"
                "This is a message from the human supervising you. Follow it."
            )
        user_text_parts.append(
            f"RECENT EVENTS:\n{events_text}"
        )
        user_text_parts.append(kb_block)
        user_text_parts.append(
            "Look at the screenshot and decide what to do next. "
            "Use the tools available to you. You may call multiple tools in sequence."
        )

        user_text = "\n\n".join(user_text_parts)

        user_message: dict[str, Any] = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame_b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        }

        # ── 5. Build messages for LLM call ────────────────────────────────
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            user_message,
        ]

        # ── 6. LLM call with tool-calling ─────────────────────────────────
        dispatcher = ToolDispatcher(
            pyboy=pyboy,
            kb=kb,
            tile_graph=tile_graph,
            ram_state=ram_state,
            pump_fn=pump_fn,
        )

        try:
            assistant_message, tool_results = _run_llm_turn(
                client=client,
                model=model,
                messages=messages,
                dispatcher=dispatcher,
                pump_fn=pump_fn,
                extra_body=extra_body,
                train_log=train_log,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            print(f"[nova] LLM call failed: {exc}")
            # Fallback: press B to try to unstick any UI.
            press_sequence(pyboy, ["B"], pump_fn=pump_fn)
            continue

        # ── 7. Post-turn: wall detection from RAM delta ───────────────────
        if has_ram and dispatcher.pressed_this_turn and not dispatcher.used_pathfinder:
            _detect_walls_post_turn(
                pyboy=pyboy,
                ram_offsets=ram_offsets,
                pre_ram=ram_state,
                buttons=dispatcher.pressed_this_turn,
                tile_graph=tile_graph,
            )

        # ── 8. Update conversation history ────────────────────────────────
        history.append(user_message)
        history.append(assistant_message)
        # Append tool result messages.
        history.extend(tool_results)

        # Summarize if history is getting long.
        if len(history) > MAX_HISTORY_MESSAGES:
            history = summarize_and_reset(
                client=client,
                model=model,
                history=history,
                kb=kb,
                extra_body=extra_body,
            )

        # ── 9. Autosave ───────────────────────────────────────────────────
        if save_sequence and turn % AUTOSAVE_EVERY_N_TURNS == 0:
            _in_battle = ram_state.get("in_battle", False)
            if not _in_battle:
                print("[nova] Autosaving…")
                press_sequence(pyboy, save_sequence, pump_fn=pump_fn)

        save_state(pyboy, state_snap)

    print("[nova] Loop ended.")
    pyboy.stop()


# ---------------------------------------------------------------------------
# LLM turn execution (handles tool-call loop)
# ---------------------------------------------------------------------------

def _run_llm_turn(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    dispatcher: ToolDispatcher,
    pump_fn,
    extra_body: dict | None,
    train_log: Path,
    system_prompt: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Call the LLM, execute tool calls, and return the assistant message + tool results."""
    call_messages = list(messages)
    all_tool_results: list[dict[str, Any]] = []

    for _attempt in range(8):  # Max tool-call rounds per turn.
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": call_messages,
            "tools": TOOLS,
            "tool_choice": "required",
            "max_tokens": 2048,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # Convert to dict for history storage.
        assistant_dict: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]

        call_messages.append(assistant_dict)

        if not msg.tool_calls:
            # tool_choice=required should prevent this, but handle it gracefully.
            if msg.content:
                print(f"[nova] Model (no tool): {msg.content[:200]}")
            return assistant_dict, all_tool_results

        # Execute each tool call and collect results.
        _known_tools = {"press_buttons", "navigate_to", "update_knowledge", "add_event", "set_goal"}
        any_action = False
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tool_name not in _known_tools:
                result = (
                    f"Unknown tool '{tool_name}'. "
                    f"Valid tools: {', '.join(sorted(_known_tools))}. "
                    "Use press_buttons to press buttons."
                )
                print(f"[nova] Unknown tool: {tool_name!r}")
            else:
                print(f"[nova] Tool: {tool_name}({_format_args(args)})")
                result = dispatcher.dispatch(tool_name, args)
                any_action = True
                print(f"[nova]  → {result}")
                # Write training record (screenshot + context → tool call).
                _log_training_record(
                    log_path=train_log,
                    system_prompt=system_prompt,
                    user_message=messages[-1],
                    tool_name=tool_name,
                    tool_args=args,
                )

            tool_result_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            call_messages.append(tool_result_msg)
            all_tool_results.append(tool_result_msg)

        # If real actions were taken, end the turn (don't loop back to model).
        # Only loop back if the model only called unknown tools and needs correction.
        if any_action:
            return assistant_dict, all_tool_results

        # All tool calls were unknown — loop back so model can correct itself.

    # Exceeded max rounds — return what we have.
    return assistant_dict, all_tool_results


# ---------------------------------------------------------------------------
# Post-move wall detection
# ---------------------------------------------------------------------------

def _detect_walls_post_turn(
    *,
    pyboy,
    ram_offsets: dict,
    pre_ram: dict[str, Any],
    buttons: list[str],
    tile_graph: TileGraph,
) -> None:
    """Detect walls from pre/post RAM position delta and record in tile graph.

    Only called when navigate_to was NOT used (navigate_to does its own
    wall recording per step).  Here we handle press_buttons calls that
    included directional buttons.
    """
    post_ram = read_ram(pyboy, ram_offsets)

    pre_mb = pre_ram.get("map_bank")
    pre_mn = pre_ram.get("map_number")
    pre_x  = pre_ram.get("x_pos")
    pre_y  = pre_ram.get("y_pos")
    post_mb = post_ram.get("map_bank")
    post_mn = post_ram.get("map_number")
    post_x  = post_ram.get("x_pos")
    post_y  = post_ram.get("y_pos")

    if None in (pre_mb, pre_mn, pre_x, pre_y, post_mb, post_mn, post_x, post_y):
        return

    # Only care about directional buttons; find the last one pressed.
    _DIRECTIONAL = {"Up", "Down", "Left", "Right"}
    dir_buttons = [b for b in buttons if b in _DIRECTIONAL]
    if not dir_buttons:
        return

    # We only have pre/post positions for the whole sequence — we don't know
    # which individual button caused a wall.  We record the LAST directional
    # if the net position didn't change from pre to post (common case for a
    # single directional button press).
    last_dir = dir_buttons[-1]
    warped = (post_mb != pre_mb or post_mn != pre_mn)
    moved = warped or (post_x != pre_x or post_y != pre_y)

    from_tile: TileKey = (pre_mb, pre_mn, pre_x, pre_y)
    tile_graph.record_move(from_tile, last_dir, success=moved)

    if moved and not warped:
        _pos_label = f"({post_x},{post_y})"
    elif warped:
        _pos_label = f"map{post_mb}:{post_mn}"
    else:
        _pos_label = f"wall going {last_dir}"

    print(f"[nova] Wall detect: {_pos_label} (moved={moved})")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_ram_summary(state: dict[str, Any]) -> str:
    """Brief RAM fact summary to prepend to the user message."""
    parts: list[str] = []

    x = state.get("x_pos")
    y = state.get("y_pos")
    mb = state.get("map_bank")
    mn = state.get("map_number")
    if x is not None:
        parts.append(f"Position: ({x},{y}) on map {mb}:{mn}")

    hp_cur = state.get("lead_hp_current")
    hp_max = state.get("lead_hp_max")
    if hp_cur is not None and hp_max and hp_max > 0:
        pct = int(hp_cur * 100 / hp_max)
        parts.append(f"Lead HP: {hp_cur}/{hp_max} ({pct}%)")
    elif hp_max == 0:
        parts.append("Lead HP: (not yet initialised)")

    johto = state.get("johto_badge_count", 0)
    if johto:
        parts.append(f"Johto badges: {johto}/8")

    party_count = state.get("party_count")
    if party_count is not None:
        parts.append(f"Party size: {party_count}")

    if state.get("in_battle"):
        parts.append("⚔ In battle")

    if state.get("all_fainted"):
        parts.append("⚠ ALL POKÉMON FAINTED — go to last Pokémon Center")

    return "RAM STATE: " + " | ".join(parts) if parts else ""


def _format_args(args: dict[str, Any]) -> str:
    """Compact repr of tool call args for logging."""
    items = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 60:
            sv = sv[:57] + "…"
        items.append(f"{k}={sv!r}")
    return ", ".join(items)


# ---------------------------------------------------------------------------
# Training data logging
# ---------------------------------------------------------------------------

def _log_training_record(
    *,
    log_path: Path,
    system_prompt: str,
    user_message: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
) -> None:
    """Append one (prompt → tool_call) pair to the training JSONL file.

    Format is OpenAI chat format with a single tool_call in the assistant
    message — ready for SFT with Unsloth or HuggingFace TRL.

    The image is included as a base64 image_url content part so the
    vision encoder is trained alongside the language model.

    Args:
        log_path: Path to the .nova_train.jsonl file (appended, not overwritten).
        system_prompt: The system prompt used this turn.
        user_message: The full user message dict (may include image_url content).
        tool_name: Name of the tool that was called.
        tool_args: Arguments passed to the tool.
    """
    record = {
        "messages": [
            {"role": "system", "content": system_prompt},
            user_message,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            },
        ]
    }
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[nova] Training log write failed (non-fatal): {exc}")
