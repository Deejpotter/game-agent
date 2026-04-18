"""
pyboy_agent.main
================
CLI entry point for the modular pyboy_agent package.

Usage
-----
    python -m pyboy_agent [options]
    python -m pyboy_agent --rom "H:/Games/GBC/Pokemon Silver.gbc"
    python -m pyboy_agent --headless --max-turns 10 --backend ollama

Options
-------
--rom PATH          Path to the .gbc/.gb ROM file (overrides ROM_PATH in .env).
--backend NAME      Backend name: lmstudio (default), ollama, openai, copilot.
--headless          Run without an SDL2 window (faster, for automated testing).
--max-turns N       Stop after N turns (0 = run forever).
--state PATH        Load a specific PyBoy state file on startup.
--speed N           Emulation speed multiplier (0=unlimited, 1=real-time).
--game NAME         Game profile name under games/ (auto-detected from ROM name).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _resolve_rom(flag_value: str | None) -> str:
    """Return the ROM path from the flag, or fall back to ROM_PATH env var."""
    if flag_value:
        return flag_value
    env_rom = os.environ.get("ROM_PATH", "")
    if env_rom:
        return env_rom
    print(
        "ERROR: No ROM specified.  "
        "Use --rom PATH or set ROM_PATH in your .env file.",
        file=sys.stderr,
    )
    sys.exit(1)


def _detect_game_name(rom_path: str) -> str:
    """Guess the game profile name from the ROM file stem.

    Examples:
        ``Pokemon Silver.gbc`` -> ``pokemon-silver``
        ``Pokemon Crystal.gbc`` -> ``pokemon-crystal``
    """
    stem = Path(rom_path).stem.lower().replace(" ", "-").replace("_", "-")
    return stem


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and start the agent."""
    parser = argparse.ArgumentParser(
        prog="pyboy_agent",
        description="Autonomous AI agent for GBC/GB games using PyBoy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--rom",       metavar="PATH", help="Path to the ROM file.")
    parser.add_argument("--backend",   default="lmstudio", choices=["lmstudio", "ollama", "openai", "copilot"], help="API backend (default: lmstudio).")
    parser.add_argument("--headless",  action="store_true", help="Run without SDL2 window.")
    parser.add_argument("--max-turns", type=int, default=0, dest="max_turns", help="Stop after N turns (0=forever).")
    parser.add_argument("--state",     metavar="PATH", help="PyBoy state file to load on startup.")
    parser.add_argument("--speed",     type=int, default=None, help="Emulation speed multiplier (0=unlimited).")
    parser.add_argument("--game",      metavar="NAME", help="Game profile name (auto-detected from ROM stem if omitted).")

    args = parser.parse_args(argv)

    rom_path  = _resolve_rom(args.rom)
    game_name = args.game or _detect_game_name(rom_path)

    print(f"[main] ROM     : {rom_path}")
    print(f"[main] Backend : {args.backend}")
    print(f"[main] Game    : {game_name}")
    print(f"[main] Headless: {args.headless}")
    if args.max_turns:
        print(f"[main] Max turns: {args.max_turns}")

    # ── Load game profile ─────────────────────────────────────────────────
    from pyboy_agent.profiles import load_game_profile
    profile = load_game_profile(game_name)

    # ── Build clients ─────────────────────────────────────────────────────
    from pyboy_agent.backends import is_copilot_backend, is_local_backend, make_client
    from pyboy_agent.config import BACKENDS

    backend_cfg = BACKENDS.get(args.backend)
    if backend_cfg is None:
        print(f"ERROR: Unknown backend '{args.backend}'.", file=sys.stderr)
        sys.exit(1)

    is_local  = is_local_backend(backend_cfg)
    is_copilot = is_copilot_backend(backend_cfg)

    vision_client  = make_client(backend_cfg)
    reason_client  = make_client(backend_cfg)

    vision_model   = backend_cfg["vision_model"]
    reasoning_model = backend_cfg["reasoning_model"]

    print(f"[main] Vision  : {vision_model}")
    print(f"[main] Reason  : {reasoning_model}")

    # ── Start loop ────────────────────────────────────────────────────────
    from pyboy_agent.loop import run_agent

    def _on_auth_error() -> None:
        """Refresh Copilot token when a 401 is encountered."""
        if is_copilot:
            from pyboy_agent.backends import make_copilot_client
            nonlocal vision_client, reason_client
            vision_client = make_copilot_client()
            reason_client = make_copilot_client()
            print("[auth] Copilot token refreshed.")
        else:
            print("[auth] 401 Unauthorized — check your API key.", file=sys.stderr)

    run_agent(
        rom=rom_path,
        game_profile=profile,
        vision_client=vision_client,
        vision_model=vision_model,
        reasoning_client=reason_client,
        reasoning_model=reasoning_model,
        headless=args.headless,
        max_turns=args.max_turns,
        state_file=args.state,
        speed=args.speed,
        is_local_vision=is_local,
        is_local_reason=is_local,
        on_auth_error=_on_auth_error,
    )
