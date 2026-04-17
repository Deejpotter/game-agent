---
description: "Use when creating or editing game profiles in games/*.json. Covers required fields, system prompt format, save sequences, and RAM offsets."
applyTo: "games/**/*.json"
---

# Game Profile Guidelines

Game profiles live at `games/<name>.json` where `<name>` matches the `--game` CLI arg.

## Required Fields

```json
{
	"name": "Human-readable game name",
	"console": "gba",
	"system_prompt": "...",
	"save_sequence": ["Start", "Down", "A"],
	"ram_offsets": {}
}
```

## `system_prompt` Rules

- Must end with: `Reply with a JSON object ONLY — no markdown, no explanation:\n{"button": "<one button string>", "reason": "<one sentence explaining why>"}`
- Include game-specific knowledge: story path, type chart, healing threshold, stuck-loop recovery
- Available buttons: `A`, `B`, `Up`, `Down`, `Left`, `Right`, `Start`, `Select`, `L`, `R`
- Include an explicit stuck-loop escape: "If stuck on same screen 3+ turns: press B, try a different direction"

See `games/pokemon-sapphire.json` for the canonical example including gym order, story path, and memory addresses.

## `save_sequence`

The button list is sent every 60 turns (`AUTOSAVE_EVERY_N_TURNS`). Navigate from the overworld title bar to the save dialog. For Pokemon Sapphire: `["Start","Down","Down","Down","A","A"]`.

## `ram_offsets`

Keys are human names, values are hex address strings (`"0x03004BD8"`). All offsets assume the US/UE ROM revision — note this explicitly in the `"note"` field. Addresses are read via `mgba_live_read_memory` MCP tool.
