"""
mgba_agent/bridge/emulator.py — High-level emulator helpers (thin wrappers over BridgeClient).

These functions provide the primary action API used by the game loop:
  - press_button()       — tap one button and return the next screenshot
  - walk_steps()         — press a directional button N times efficiently
  - capture_screenshot() — get a screenshot with retry logic
  - save_game()          — run the game-specific save_sequence
"""

from __future__ import annotations

import asyncio

from ..config import SETTLE_FRAMES
from .client import BridgeClient


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
