"""
pyboy_agent.navigation.wall_tracker
=====================================
Post-move wall detection and per-tile wall/tested recording.

Wall detection strategy
------------------------
After every directional button press we compare pre-move and post-move
position.  If the position changed → not a wall; otherwise → wall.

**RAM delta is the primary check.**
Screenshot hash comparison is unreliable in small indoor rooms where the
camera doesn't scroll — the hash is identical whether the player moved one
tile or was blocked by a wall.  RAM position is always correct.

Hash fallback
~~~~~~~~~~~~~
When ``has_ram`` is False (game has no RAM offsets), we fall back to MD5
hash comparison of the screenshot.  This is only used for games without
profiles.

Map warp detection
~~~~~~~~~~~~~~~~~~~
If map_bank or map_number changes after the button press, a warp happened —
the character definitely moved, regardless of what x/y say.  Warp transitions
take many frames so x/y may not have settled yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyboy_agent.ram.reader import read_ram_state
from pyboy_agent.emulator import screenshot_hash
from pyboy_agent.navigation.world_map import WorldMap

if TYPE_CHECKING:
    from pyboy import PyBoy


def detect_and_record_wall(
    pyboy: "PyBoy",
    *,
    button: str,
    old_screenshot_b64: str,
    new_screenshot_b64: str,
    pre_ram_state: dict[str, Any],
    has_ram: bool,
    ram_offsets: dict[str, str],
    tile_key: str,
    world_map: WorldMap,
    pre_map_bank: int | None = None,
    pre_map_number: int | None = None,
) -> tuple[bool, str | None]:
    """Detect whether a directional button press was blocked by a wall.

    Must be called immediately after the button press and screenshot capture,
    before any other emulator ticks.

    Args:
        pyboy: Running PyBoy instance (used to re-read post-move RAM).
        button: The directional button that was pressed.
        old_screenshot_b64: Screenshot before the button press.
        new_screenshot_b64: Screenshot after the button press.
        pre_ram_state: RAM state read before the button press.
        has_ram: True if RAM offsets are available.
        ram_offsets: Game profile RAM offset dict.
        tile_key: Per-tile wall key for the pre-move tile.
        world_map: WorldMap instance to record wall/tested data.
        pre_map_bank: Map bank value before the press (for warp detection).
        pre_map_number: Map number value before the press (for warp detection).

    Returns:
        (wall_detected, wall_button) — wall_button is None when no wall.
    """
    if button not in {"Up", "Down", "Left", "Right"}:
        return False, None

    wall_detected: bool

    if has_ram:
        pre_x = pre_ram_state.get("x_pos")
        pre_y = pre_ram_state.get("y_pos")

        if pre_x is not None and pre_y is not None:
            post_ram = read_ram_state(pyboy, ram_offsets)
            post_x = post_ram.get("x_pos")
            post_y = post_ram.get("y_pos")
            post_mb = post_ram.get("map_bank")
            post_mn = post_ram.get("map_number")

            # A map change means a warp fired — definitely moved.
            map_changed = (post_mb != pre_map_bank or post_mn != pre_map_number)
            moved = map_changed or (post_x != pre_x or post_y != pre_y)
            wall_detected = not moved
        else:
            # RAM present but position bytes couldn't be read — fall back to hash.
            wall_detected = screenshot_hash(new_screenshot_b64) == screenshot_hash(old_screenshot_b64)
    else:
        # No RAM offsets — use screenshot hash (works in large outdoor areas only).
        wall_detected = screenshot_hash(new_screenshot_b64) == screenshot_hash(old_screenshot_b64)

    # Record the result in the world map.
    if tile_key:
        if wall_detected:
            world_map.record_wall(tile_key, button)
        world_map.record_tested(tile_key, button)

    if wall_detected:
        print(f"  [wall]  {button!r} blocked — warning VLM next turn")
        return True, button

    return False, None
