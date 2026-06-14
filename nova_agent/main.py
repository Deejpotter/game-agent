"""
nova_agent.main
===============
CLI argument parsing and agent startup.

Usage::

    python -m nova_agent --rom "H:/Games/GBC/Pokemon Silver.gbc"
    python -m nova_agent --rom ... --backend openai --max-turns 10
    python -m nova_agent --rom ... --headless --game pokemon-silver
    python -m nova_agent --list-profiles
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m nova_agent",
        description="nova_agent — single-model tool-calling GBC game agent",
    )

    parser.add_argument(
        "--rom",
        default=os.getenv("ROM_PATH", ""),
        help="Path to the GBC/GB ROM file. Defaults to ROM_PATH env var.",
    )
    parser.add_argument(
        "--game",
        default="",
        help=(
            "Game profile slug (e.g. 'pokemon-silver') or path to a JSON file. "
            "Auto-detected from ROM filename if omitted."
        ),
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("NOVA_BACKEND", os.getenv("BACKEND", "lmstudio")),
        choices=["lmstudio", "ollama", "openai", "copilot"],
        help="LLM backend to use. Default: lmstudio.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override the model name from the backend config.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without SDL2 window (faster, no display required).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Stop after this many turns (default: run forever).",
    )
    parser.add_argument(
        "--state",
        default="",
        help="Path to a PyBoy .state file to load on startup.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        help="Emulation speed multiplier (0=unlimited, 1=real-time).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available game profiles and exit.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from nova_agent.profiles import list_profiles, load_game_profile
    from nova_agent.config import BACKENDS
    from nova_agent.agent import _make_client, run_agent

    if args.list_profiles:
        profiles = list_profiles()
        print("Available game profiles:")
        for p in profiles:
            print(f"  {p}")
        return

    # ── Validate ROM ──────────────────────────────────────────────────────
    if not args.rom:
        print("[nova] ERROR: --rom is required (or set ROM_PATH in .env)")
        sys.exit(1)

    rom_path = Path(args.rom)
    if not rom_path.exists():
        print(f"[nova] ERROR: ROM not found: {rom_path}")
        sys.exit(1)

    # ── Load game profile ─────────────────────────────────────────────────
    game_slug = args.game
    if not game_slug:
        # Auto-detect from ROM filename.
        stem = rom_path.stem.lower()
        if "silver" in stem:
            game_slug = "pokemon-silver"
        elif "gold" in stem:
            game_slug = "pokemon-silver"  # same RAM layout
        elif "crystal" in stem:
            game_slug = "pokemon-silver"
        elif "firered" in stem or "fire_red" in stem:
            game_slug = "pokemon-firered"
        else:
            game_slug = "pokemon-silver"  # fallback
        print(f"[nova] Auto-detected profile: {game_slug}")

    try:
        profile = load_game_profile(game_slug)
    except FileNotFoundError as exc:
        print(f"[nova] ERROR: {exc}")
        sys.exit(1)

    # ── Build LLM client ──────────────────────────────────────────────────
    try:
        client, model, is_local = _make_client(args.backend, BACKENDS)
    except Exception as exc:
        print(f"[nova] ERROR: Could not create LLM client: {exc}")
        sys.exit(1)

    if args.model:
        model = args.model

    print(f"[nova] Backend: {args.backend} | Model: {model}")

    # ── Run ───────────────────────────────────────────────────────────────
    run_agent(
        rom=str(rom_path),
        game_profile=profile,
        client=client,
        model=model,
        headless=args.headless,
        max_turns=args.max_turns,
        state_file=args.state or None,
        speed=args.speed,
        is_local=is_local,
    )
