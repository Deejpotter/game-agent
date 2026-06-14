"""
nova_agent.emulator
===================
PyBoy lifecycle management, button execution, and screenshot capture
with a RAM-state overlay drawn directly onto the image.

Key difference from pyboy_agent:
- ``capture_frame()`` draws an info bar onto the screenshot so the LLM
  sees RAM facts (position, HP, badges) visually alongside the pixels —
  eliminating the need for a separate text block in the prompt.
- ``press_sequence()`` accepts a list of button names and presses them all
  in one call, allowing the LLM to plan multi-step actions per turn.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

from nova_agent.config import (
    BUTTON_MAP,
    BUTTON_ALIASES,
    HP_LOW_PCT,
    OVERLAY_BG_COLOR,
    OVERLAY_HEIGHT,
    OVERLAY_TEXT_COLOR,
    OVERLAY_WARN_COLOR,
    SCREENSHOT_SCALE,
    SETTLE_FRAMES_BUTTON,
    SETTLE_FRAMES_MOVE,
)

if TYPE_CHECKING:
    from pyboy import PyBoy

_DIRECTIONAL = {"Up", "Down", "Left", "Right"}


# ---------------------------------------------------------------------------
# PyBoy lifecycle
# ---------------------------------------------------------------------------

def create_pyboy(rom: str, *, headless: bool = True, speed: int | None = None) -> "PyBoy":
    """Create and return a running PyBoy instance."""
    import os as _os, sys as _sys
    from pyboy import PyBoy as _PyBoy

    window = "null" if headless else "SDL2"
    emu_speed = speed if speed is not None else (0 if headless else 1)

    # Suppress PyBoy's ROM-load progress bar (carriage-return spam).
    _devnull = open(_os.devnull, "w")
    _old_stdout = _sys.stdout
    _sys.stdout = _devnull
    try:
        pyboy = _PyBoy(
            rom,
            window=window,
            cgb=True,
            sound_emulated=False,
            log_level="ERROR",
        )
    finally:
        _sys.stdout = _old_stdout
        _devnull.close()

    pyboy.set_emulation_speed(emu_speed)
    return pyboy


def load_state(pyboy: "PyBoy", path: str | Path) -> bool:
    """Load a saved state. Returns True on success."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        with open(p, "rb") as f:
            pyboy.load_state(f)
        return True
    except Exception:
        return False


def save_state(pyboy: "PyBoy", path: str | Path) -> bool:
    """Save current state. Returns True on success."""
    try:
        with open(path, "wb") as f:
            pyboy.save_state(f)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Button execution
# ---------------------------------------------------------------------------

def _resolve_button(name: str) -> str | None:
    """Normalise a button name to title-case (Up/Down/Left/Right/A/B/Start/Select)."""
    title = name.strip().title()
    if title in BUTTON_MAP:
        return title
    upper = name.strip().upper()
    if upper in BUTTON_ALIASES:
        return BUTTON_ALIASES[upper]
    return None


def press_sequence(pyboy: "PyBoy", buttons: list[str], pump_fn=None) -> list[str]:
    """Press a sequence of buttons, settling after each one.

    Args:
        pyboy: Running PyBoy instance.
        buttons: Ordered list of button names (e.g. ["Right", "Right", "A"]).
                 Unknown names are skipped with a warning.
        pump_fn: Optional callable; called between each settle tick to keep
                 the SDL2 window responsive in windowed mode.

    Returns:
        List of button names that were actually pressed.
    """
    pressed: list[str] = []
    for raw in buttons:
        name = _resolve_button(raw)
        if name is None:
            print(f"[emulator] Unknown button: {raw!r} — skipped")
            continue

        pyboy_key = BUTTON_MAP[name]
        pyboy.button_press(pyboy_key)
        settle = SETTLE_FRAMES_MOVE if name in _DIRECTIONAL else SETTLE_FRAMES_BUTTON

        if pump_fn is not None:
            for _ in range(settle):
                pump_fn()
        else:
            pyboy.tick(settle, render=False)

        pyboy.button_release(pyboy_key)
        pressed.append(name)

    return pressed


# ---------------------------------------------------------------------------
# Screenshot capture with overlay
# ---------------------------------------------------------------------------

def capture_frame(pyboy: "PyBoy", ram_state: dict[str, Any] | None = None) -> str:
    """Capture the current screen and return a base64-encoded PNG.

    A narrow info bar is drawn at the bottom of the screenshot showing
    key RAM facts (player position, lead HP, badge count, battle indicator).
    This means the LLM sees ground-truth numbers visually without needing a
    separate text block in the prompt.

    Args:
        pyboy: Running PyBoy instance.
        ram_state: Optional dict from ``read_ram()``.  If None, the bar
                   shows only the raw pixel frame.

    Returns:
        Base64-encoded PNG string.
    """
    # Raw screenshot from PyBoy.
    raw: Image.Image = pyboy.screen.image.convert("RGB")

    # 2× nearest-neighbour upscale so the LLM can read small text on screen.
    w, h = raw.size
    raw = raw.resize((w * SCREENSHOT_SCALE, h * SCREENSHOT_SCALE), Image.NEAREST)

    if ram_state:
        raw = _draw_overlay(raw, ram_state)

    buf = io.BytesIO()
    raw.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


def _draw_overlay(img: Image.Image, state: dict[str, Any]) -> Image.Image:
    """Draw a semi-transparent info bar below the game frame."""
    w, h = img.size
    bar_h = OVERLAY_HEIGHT

    # Create a new image with extra space at the bottom.
    canvas = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    # Dark background for the bar.
    draw.rectangle([(0, h), (w, h + bar_h)], fill=(20, 20, 20))

    # Build info text segments.
    segments: list[tuple[str, tuple[int, int, int]]] = []

    # Position
    x = state.get("x_pos")
    y = state.get("y_pos")
    mb = state.get("map_bank")
    mn = state.get("map_number")
    if x is not None and y is not None:
        segments.append((f"({x},{y})", OVERLAY_TEXT_COLOR))
    if mb is not None and mn is not None:
        segments.append((f" map{mb}:{mn}", (160, 160, 160)))

    # Lead HP
    hp_cur = state.get("lead_hp_current")
    hp_max = state.get("lead_hp_max")
    if hp_cur is not None and hp_max is not None and hp_max > 0:
        pct = int(hp_cur * 100 / hp_max)
        color = OVERLAY_WARN_COLOR if pct <= HP_LOW_PCT else OVERLAY_TEXT_COLOR
        segments.append((f"  HP {hp_cur}/{hp_max} ({pct}%)", color))

    # Badges
    johto = state.get("johto_badge_count", 0)
    kanto = state.get("kanto_badge_count", 0)
    if johto or kanto:
        segments.append((f"  J:{johto} K:{kanto}", OVERLAY_TEXT_COLOR))

    # Battle flag
    if state.get("in_battle"):
        segments.append(("  [BATTLE]", (255, 200, 50)))

    # Render segments left-to-right.
    try:
        font = ImageFont.load_default(size=11)
    except Exception:
        font = ImageFont.load_default()

    cursor_x = 4
    text_y = h + (bar_h - 12) // 2
    for text, color in segments:
        draw.text((cursor_x, text_y), text, fill=color, font=font)
        # Estimate width (default font is monospaced ~7px per char).
        cursor_x += len(text) * 7

    return canvas


def screenshot_md5(pyboy: "PyBoy") -> str:
    """Return MD5 hash of the current frame (for hash-based change detection)."""
    import hashlib
    raw = pyboy.screen.image.tobytes()
    return hashlib.md5(raw).hexdigest()
