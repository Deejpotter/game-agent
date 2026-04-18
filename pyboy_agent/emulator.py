"""
pyboy_agent.emulator
====================
Thin wrappers around the PyBoy API.

This module owns all direct interaction with the emulator:
- Screenshot capture and hashing
- Button press and walk helpers
- In-game save sequence execution

All functions take a ``PyBoy`` instance as the first argument so they remain
testable and don't hold global state.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import io
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from pyboy_agent.config import (
    BUTTON_MAP,
    SCREENSHOT_SCALE,
    SETTLE_FRAMES_BUTTON,
    SETTLE_FRAMES_MOVE,
)

if TYPE_CHECKING:
    from pyboy import PyBoy


# ---------------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------------

def capture_screenshot(
    pyboy: "PyBoy",
    scale: int = SCREENSHOT_SCALE,
    *,
    shots_dir: Path | None = None,
) -> str:
    """Capture the current screen and return it as a base64-encoded PNG string.

    Converts RGBA → RGB (PyBoy returns RGBA), applies nearest-neighbour upscaling
    for better VLM legibility, and optionally saves the raw file to disk for
    debugging or replay.

    Args:
        pyboy: Running PyBoy instance.
        scale: Integer upscale factor applied with nearest-neighbour resampling.
        shots_dir: If given, each frame is saved as a timestamped PNG here.

    Returns:
        Base64-encoded PNG string suitable for an OpenAI image_url content block.
    """
    raw_img = pyboy.screen.image
    if raw_img is None:
        # Edge case: image not ready yet — tick one frame to force a render.
        pyboy.tick(1, render=True)
        raw_img = pyboy.screen.image
    img: Image.Image = raw_img.convert("RGB")  # type: ignore[union-attr]
    if scale != 1:
        img = img.resize(
            (img.width * scale, img.height * scale),
            Image.Resampling.NEAREST,
        )
    if shots_dir:
        shots_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        img.save(shots_dir / f"shot-{ts}.png")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def screenshot_hash(b64: str) -> str:
    """Return an MD5 hex digest of the base64 screenshot string.

    Used as a fast way to detect identical frames (potential wall hit) without
    decoding the image. Only reliable in large outdoor areas where the camera
    scrolls; indoor rooms with a fixed camera produce identical hashes even
    when the player moves — always prefer RAM position delta for wall detection.
    """
    return hashlib.md5(b64.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Button input helpers
# ---------------------------------------------------------------------------

def _tick_button(pyboy: "PyBoy", vlm_button: str, settle_frames: int, render: bool = True) -> None:
    """Press one button and advance the emulator by settle_frames.

    Args:
        pyboy: Running PyBoy instance.
        vlm_button: Button name in VLM format (e.g. ``"A"``, ``"Up"``).
        settle_frames: How many frames to tick after the press.
        render: If True, render each frame (required for SDL2 window responsiveness).
    """
    gbc_key = BUTTON_MAP.get(vlm_button)
    if gbc_key:
        pyboy.button(gbc_key)
    pyboy.tick(settle_frames, render)


def press_button(
    pyboy: "PyBoy",
    button: str,
    settle_frames: int = SETTLE_FRAMES_BUTTON,
    shots_dir: Path | None = None,
) -> str:
    """Press a single button, wait, and return a fresh screenshot.

    Use this for A, B, Start, Select, and single directional taps.

    Returns:
        Base64-encoded PNG of the screen after the press.
    """
    _tick_button(pyboy, button, settle_frames, render=True)
    return capture_screenshot(pyboy, shots_dir=shots_dir)


def walk_steps(
    pyboy: "PyBoy",
    button: str,
    repeat: int,
    settle_frames: int = SETTLE_FRAMES_MOVE,
    shots_dir: Path | None = None,
) -> str:
    """Hold a directional button for ``repeat`` tile-steps.

    Only the final step renders a frame — intermediate steps skip rendering
    for speed (the SDL2 window stays responsive because the main loop pumps
    events between VLM calls, not during walks).

    Args:
        pyboy: Running PyBoy instance.
        button: Directional button (``"Up"`` / ``"Down"`` / ``"Left"`` / ``"Right"``).
        repeat: Number of tile-steps to take before re-evaluating.
        settle_frames: Frames to tick per step.
        shots_dir: Optional directory for debug screenshot dumps.

    Returns:
        Base64-encoded PNG of the screen after the final step.
    """
    for i in range(repeat):
        is_last = (i == repeat - 1)
        _tick_button(pyboy, button, settle_frames, render=is_last)
    return capture_screenshot(pyboy, shots_dir=shots_dir)


def save_game(pyboy: "PyBoy", save_sequence: list[str]) -> str:
    """Execute the game-profile's in-game save sequence.

    The save_sequence is a list of button names (VLM format) defined in
    the game profile JSON. It navigates the in-game pause menu to the SAVE
    option and confirms, then waits for the save animation to finish.

    Returns:
        Base64-encoded PNG of the screen after saving.
    """
    print("  [autosave] running save sequence…")
    for button in save_sequence:
        _tick_button(pyboy, button, SETTLE_FRAMES_BUTTON, render=True)
        # Small real-time gap so the GBC can process button-release timing.
        time.sleep(0.08)
    print("  [autosave] done.")
    # Wait for the save-complete animation to finish before taking the screenshot.
    pyboy.tick(30, render=True)
    return capture_screenshot(pyboy)


# ---------------------------------------------------------------------------
# PyBoy startup helper
# ---------------------------------------------------------------------------

def create_pyboy(rom: str, *, headless: bool = False, speed: int | None = None) -> "PyBoy":
    """Create and return a configured PyBoy instance.

    Suppresses PyBoy's carriage-return progress output during ROM load so it
    doesn't corrupt the agent's own log output.

    Args:
        rom: Absolute path to the GBC/GB ROM file.
        headless: If True, uses the ``null`` window (max speed, no display).
        speed: Emulation speed multiplier. Defaults to 0 (unlimited) when
               headless, 1 (realtime) when windowed.

    Returns:
        Running PyBoy instance with the ROM loaded.
    """
    from pyboy import PyBoy  # type: ignore[import-untyped]

    window = "null" if headless else "SDL2"
    emu_speed = speed if speed is not None else (0 if headless else 1)

    print(
        f"[emulator] Starting PyBoy ({window} window, "
        f"speed={'unlimited' if emu_speed == 0 else f'{emu_speed}x'})…"
    )

    # Suppress PyBoy's carriage-return ROM-load progress bar.
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
    return pyboy
