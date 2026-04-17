---
description: "Use when editing pyboy_agent.py — the main game loop, VLM integration, WorldMap, RAM reader, or CLI args. Covers two-stage perception, wall detection, windowed threading, and extension patterns."
applyTo: "pyboy_agent.py"
---

# pyboy_agent.py — Editing Guide

## Turn Loop Structure

Each turn runs in this order:

1. **RAM read** — `read_ram_state()` at the TOP of the turn. Result (`_ram_state`) is used for the nav-hint wall key, the pre-move position snapshot, and the decide prompt. Do not read RAM twice per turn.
2. **Nav hints** — wall warnings, stuck-loop warnings, retalk guards, all assembled into `nav_hint`.
3. **`perceive()`** — vision model call: screenshot → JSON scene description.
4. **Scene parse** — `json.loads(scene)`, normalise directions to title case, extract nameplate/location/dialogue.
5. **`decide()`** — reasoning model call: scene + RAM + nav hint + history → `(button, repeat, reason, event, new_goal, map_update, new_memory)`.
6. **Button press** — `press_button()` or `walk_steps()`, then read post-move RAM for wall detection.
7. **Wall detection** — compare pre/post RAM `x_pos`/`y_pos`. Record wall under `map_{bank}_{number}` key.
8. **History + notes** — append to `history` (capped at 10), flush `notes.json` + state snapshot.

## Two-Stage VLM Pipeline

```
perceive(vision_client, vision_model, screenshot_b64)
    └── returns JSON string (screen_type, passable_directions, player_facing, etc.)

decide(reason_client, reason_model, scene_json, ram_text, nav_hint, history, ...)
    └── returns (button, repeat, reason, event, new_goal, map_update, new_memory)
```

- `perceive()` must NOT include strategy — describe only. `max_tokens=2048`.
- `decide()` never receives the raw image. `max_tokens=4096`, `timeout=120s`.
- Both calls wrapped in `_with_retry()` (6 retries, exponential backoff).
- `enable_thinking: True` injected via `extra_body` for any `http://localhost` backend.

## Wall Detection (critical — read before touching)

**Primary:** RAM position delta. After pressing a directional button, read `x_pos`/`y_pos` again. If unchanged → wall. This works in small indoor rooms where the camera doesn't scroll (screenshot hash would falsely match).

**Fallback:** Screenshot hash comparison, used only when `has_ram` is False.

**Wall keys:** Always use `map_{bank}_{number}` (from RAM), never the VLM's fuzzy location name. This prevents walls from bleeding between rooms that the VLM names similarly.

```python
# Correct pattern:
_wall_location_key = f"map_{_ram_state['map_bank']}_{_ram_state['map_number']}"
world_map.record_wall(_wall_location_key, button)
known_walls = world_map.get_walls(_wall_location_key)
```

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

- `world_map.record_wall(key, direction)` — mark direction as impassable
- `world_map.get_walls(key)` — returns set of blocked directions
- `world_map.record_tested(key, direction)` — mark a direction as tried (passable or not)
- `world_map.update(name, location_status=...)` — update location status

## Do Not Touch

- `mgba_live_bridge.lua` — owned by `mgba-live-mcp` package
- `mgba_launcher.lua` — auto-generated at runtime; changes are overwritten on next run
