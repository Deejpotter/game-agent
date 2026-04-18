---
description: "Use when editing pyboy_agent.py — the main game loop, VLM integration, WorldMap, RAM reader, or CLI args. Covers two-stage perception, wall detection, windowed threading, and extension patterns."
applyTo: "pyboy_agent.py,pyboy_agent/loop.py"
---

# pyboy_agent — Agent Loop Editing Guide

Applies to both **`pyboy_agent.py`** (legacy monolith) and **`pyboy_agent/loop.py`** (package `run_agent()`). The package is the canonical version — prefer editing it. Do not break the monolith; it shares `games/` profiles and `.pyboy_agent.state` snapshot files.

## Turn Loop Structure

Each turn runs in this order:

1. **RAM read** — `read_ram_state()` at the TOP of the turn. Result (`_ram_state`) is used for the nav-hint wall key, the pre-move position snapshot, and the decide prompt. Do not read RAM twice per turn.
2. **Nav hints** — `build_nav_hints()` (package) / inline code (monolith) — wall warnings, stuck-loop warnings, retalk guards, badge-phase guide, all assembled into `nav_hint`.
3. **RAM fast-paths** — skip VLM entirely for obvious states: `dialogue_open → A`, `menu_open (not battle) → B`, `in_battle (odd turns) → A`.
4. **`perceive()`** — vision model call: screenshot → JSON scene description. Skipped when RAM fast-paths fire.
5. **Scene parse** — `json.loads(scene)`, normalise directions to title case, strip `passable_directions` when RAM position is available (VLM values are unreliable).
6. **`decide()`** — reasoning model call: scene + RAM + nav hint + history → 8-tuple.
7. **Button press** — `press_button()` or `walk_steps()`, then read post-move RAM for wall detection.
8. **Wall detection** — `detect_and_record_wall()` — compare pre/post RAM `x_pos`/`y_pos`. Record wall under per-tile key.
9. **History + notes** — append to `history` (capped at `MAX_HISTORY_MESSAGES`), flush `NotesTracker` + state snapshot.

## Two-Stage VLM Pipeline

```
perceive(vision_client, vision_model, screenshot_b64)
    └── returns JSON string (screen_type, player_facing, location_name, etc.)
        Note: passable_directions is omitted from the prompt — RAM delta is authoritative.

decide(reason_client, reason_model, scene_json, history, system_prompt, ...)
    └── returns 8-tuple:
        (button, repeat, reason, event, new_goal, map_update, new_memory, thinking)
        thinking = model chain-of-thought — logged only, never acted on
```

- `perceive()` must NOT include strategy — describe only. `max_tokens=2048`.
- `decide()` never receives the raw image. `max_tokens=4096`, `timeout=120s`.
- Both calls wrapped in `with_retry()` (6 retries, exponential backoff).
- `enable_thinking: True` injected via `extra_body` for any `http://localhost` backend.

## Wall Detection (critical — read before touching)

**Primary:** RAM position delta. After pressing a directional button, read `x_pos`/`y_pos` again. If unchanged → wall. This works in small indoor rooms where the camera doesn't scroll (screenshot hash would falsely match).

**Fallback:** Screenshot hash comparison, used only when `has_ram` is False.

**Wall keys:** Always use `map_{bank}_{number}_x{cx}_y{cy}` (per-tile, from RAM), never the VLM's fuzzy location name. Per-tile keys mean a wall in one doorway doesn't block the entire room.

```python
# Package pattern (detect_and_record_wall in navigation/wall_tracker.py):
_tile_key = f"map_{_mb}_{_mn}_x{_cx}_y{_cy}"
wall_detected, wall_button = detect_and_record_wall(
    pyboy, button=button, old_screenshot_b64=old_b64,
    new_screenshot_b64=current_b64, pre_ram_state=_ram_state,
    has_ram=has_ram, ram_offsets=ram_offsets,
    tile_key=_tile_key, world_map=world_map,
    pre_map_bank=_pre_mb, pre_map_number=_pre_mn,
)
```

**Wall-reset guard:** If all 4 cardinal directions are blocked at the current tile, it is physically impossible — a frozen dialogue or cutscene is the likely cause. Press B×5 + A, then `world_map.clear_walls(tile_key)`.

**Stuck-tile recovery:** If `turns_at_same_tile >= STUCK_TILE_THRESHOLD` (8), press B×3 every 4 turns to break out of frozen states.

## Windowed Mode Threading

When `headless=False`:

- `pump_fn = lambda: pyboy.tick(1, render=True)` runs on the **main thread** at ~60 fps
- VLM calls run in a `ThreadPoolExecutor` background thread via `_with_retry(fn, pump_fn=pump_fn)`
- SDL2 requires all rendering on the main thread — never call `pyboy.tick()` from a worker thread

## Direction Normalisation

VLM models may return directions in any case. Always normalise to title case immediately after parsing perceive output:

```python
scene_parsed["passable_directions"] = [d.title() for d in scene_parsed["passable_directions"]]
scene_parsed["player_facing"] = scene_parsed["player_facing"].title()
```

`BUTTON_MAP` keys are title case: `"Up"`, `"Down"`, `"Left"`, `"Right"`.

## Key Constants

| Constant                 | Default | Effect                                                    |
| ------------------------ | ------- | --------------------------------------------------------- |
| `SETTLE_FRAMES_MOVE`     | 16      | Frames after directional button (one tile walk animation) |
| `SETTLE_FRAMES_BUTTON`   | 8       | Frames after A/B/Start/Select                             |
| `SETTLE_FRAMES_CUTSCENE` | 30      | Frames after screen transitions                           |
| `SCREENSHOT_SCALE`       | 2       | GBC 160×144 → 320×288 for VLM                             |
| `AUTOSAVE_EVERY_N_TURNS` | 60      | How often `save_sequence` fires                           |
| `STUCK_BUTTON_THRESHOLD` | 5       | Consecutive same button triggers hint                     |

## RAM State (`has_ram`)

`has_ram = bool({k for k in ram_offsets if k != "note"})` — True only when real address keys exist beyond the metadata `"note"` key. Always gate RAM reads behind `if has_ram`.

`read_ram_state()` returns a dict with: `map_bank`, `map_number`, `x_pos`, `y_pos`, `lead_hp_current`, `lead_hp_max`, `lead_hp_pct`, `johto_badge_count`, `money`, etc.

HP is big-endian 16-bit. Money is 3-byte BCD. Guard against `hp_max == 0` (RAM not initialised) before computing percentages.

## WorldMap

Persistent cross-session tracker stored at `~/.pyboy-agent/world_maps/<game-slug>.json`.

- `world_map.record_wall(key, direction)` — mark direction as impassable at that tile
- `world_map.get_walls(key)` — returns set of blocked directions for that tile
- `world_map.clear_walls(key)` — remove all walls for a tile (used by wall-reset guard)
- `world_map.record_tested(key, direction)` — mark a direction as tried (passable or not)
- `world_map.update(name, location_status=...)` — update location status
- `best_location_key(world_map, name)` — fuzzy match a VLM location name to a known key

## Do Not Touch

- `mgba_live_bridge.lua` — owned by `mgba-live-mcp` package
- `mgba_launcher.lua` — auto-generated at runtime; changes are overwritten on next run
