"""
mgba_agent/main.py — CLI entry point for the mGBA game agent.

Parses command-line arguments, resolves backend configuration, loads the game
profile, and starts run_agent(). Can be invoked as:

    python -m mgba_agent --rom "path/to/game.gba" --game pokemon-sapphire
    python mgba_agent/agent.py --rom "path/to/game.gba"
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import Any

from dotenv import load_dotenv

from .config import BACKENDS
from .loop import run_agent
from .profiles import load_game_profile

load_dotenv()


def _handle_sigint(signum: int, frame: Any) -> None:
    print("\n[agent] Ctrl+C received — shutting down gracefully…")
    sys.exit(0)


def main() -> None:
    import argparse

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
            'Example: "C:/Program Files/mGBA/mGBA.exe"'
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
    reasoning_cfg: dict | None
    if args.reasoning_backend and args.reasoning_backend != args.backend:
        reasoning_cfg = dict(BACKENDS[args.reasoning_backend])
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
