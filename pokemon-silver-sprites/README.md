# Pokemon Silver Sprites — Research Archive

A research archive of all 251 Pokemon battle sprites from **Pokémon Silver Version** (Game Boy Color, 2000), with structured metadata for each Pokemon.

---

## About the Game

**Pokémon Silver Version** (ポケットモンスター 銀, *Pocket Monsters Silver*) is a role-playing game developed by Game Freak and published by Nintendo for the Game Boy Color. Released in Japan in November 1999 and internationally in 2000, it is one of the Generation II titles alongside Pokémon Gold Version. The game introduced 100 new Pokemon (totalling 251) and was set in the Johto region.

Key facts:
- Platform: Game Boy Color
- Developer: Game Freak / Nintendo
- Release: 1999 (JP) / 2000 (INT)
- Generation: II (Johto)
- Total Pokemon: 251 (Gen I: 1–151, Gen II: 152–251)

---

## Folder Structure

```
pokemon-silver-sprites/
├── README.md                   ← This file
├── index.json                  ← Master index of all 251 Pokemon
├── download_sprites.py         ← Script used to generate this archive
├── sprites/                    ← PNG sprite images (56×56 px)
│   ├── 001-bulbasaur.png
│   ├── 002-ivysaur.png
│   └── ... (251 sprites)
└── data/                       ← JSON metadata files (one per Pokemon)
    ├── 001-bulbasaur.json
    ├── 002-ivysaur.json
    └── ... (251 JSON files)
```

---

## Sprite Details

- **Source**: PokeAPI/sprites (GitHub) — `sprites/pokemon/versions/generation-ii/silver/`
- **Format**: PNG, 56×56 pixels
- **Naming**: `NNN-name.png` where NNN is the zero-padded National Pokédex number
- **Style**: Front-facing battle sprites as they appeared in Pokemon Silver's battle system
- **Palette**: Silver version color palette (slight differences from Gold version)

---

## Metadata JSON Structure

Each file in `data/` follows this schema:

```json
{
  "pokedex_number": 1,
  "name": "Bulbasaur",
  "api_name": "bulbasaur",
  "category": "Seed Pokémon",
  "types": ["Grass", "Poison"],
  "abilities": ["Overgrow"],
  "hidden_abilities": ["Chlorophyll"],
  "height_dm": 7,
  "weight_hg": 69,
  "base_stats": {
    "hp": 45,
    "attack": 49,
    "defense": 49,
    "special_attack": 65,
    "special_defense": 65,
    "speed": 45
  },
  "base_experience": 64,
  "capture_rate": 45,
  "base_happiness": 70,
  "growth_rate": "medium-slow",
  "flavor_text": "A strange seed was planted on its back at birth...",
  "flavor_text_version": "silver",
  "evolves_from": null,
  "is_legendary": false,
  "is_mythical": false,
  "habitat": "grassland",
  "sprite_file": "../sprites/001-bulbasaur.png",
  "sprite_url": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/silver/1.png",
  "sprite_available": true,
  "game": "Pokémon Silver Version",
  "platform": "Game Boy Color",
  "release_year": 2000,
  "generation": "Generation II"
}
```

### Field Descriptions

| Field | Description |
|---|---|
| `pokedex_number` | National Pokédex entry number (1–251) |
| `name` | Display name |
| `category` | Species category (e.g. "Seed Pokémon") |
| `types` | Elemental types (1 or 2) |
| `abilities` | Non-hidden battle abilities |
| `height_dm` | Height in decimetres (÷10 = metres) |
| `weight_hg` | Weight in hectograms (÷10 = kilograms) |
| `base_stats` | Six base stat values used in combat |
| `capture_rate` | Catch difficulty (0–255; lower = rarer) |
| `base_happiness` | Starting friendship value |
| `growth_rate` | EXP curve for levelling up |
| `flavor_text` | In-game Pokédex entry (Silver version preferred) |
| `flavor_text_version` | Which game version the flavor text is from |
| `evolves_from` | Previous evolution name, or null for base forms |
| `is_legendary` | Legendary status |
| `is_mythical` | Mythical/event-only status |
| `sprite_file` | Relative path to the PNG sprite |
| `sprite_available` | Whether a Silver-specific sprite was found |

---

## Master Index (`index.json`)

`index.json` contains a lightweight summary list of all Pokemon for quick lookups:

```json
{
  "project": "Pokemon Silver Sprites Research",
  "total_pokemon": 251,
  "pokemon": [
    {
      "pokedex_number": 1,
      "name": "Bulbasaur",
      "types": ["Grass", "Poison"],
      "sprite_file": "sprites/001-bulbasaur.png",
      "data_file": "data/001-bulbasaur.json",
      "sprite_available": true
    }
  ]
}
```

---

## Data Sources

| Source | URL | Usage |
|---|---|---|
| PokeAPI REST v2 | https://pokeapi.co | Pokemon metadata, flavor text, stats |
| PokeAPI/sprites (GitHub) | https://github.com/PokeAPI/sprites | Sprite PNG images |
| Bulbagarden Archives | https://archives.bulbagarden.net/wiki/Category:Silver_sprites | Reference / verification |

---

## Generation II Pokemon (New in Silver)

Pokemon #152–251 were introduced in Generation II and appear only in the Johto region of Pokemon Gold/Silver/Crystal:

| Range | Region | Starters |
|---|---|---|
| 001–151 | Kanto (Gen I) | Bulbasaur, Charmander, Squirtle |
| 152–251 | Johto (Gen II) | Chikorita, Cyndaquil, Totodile |

Notable Gen II Pokemon in Silver:
- **Starters**: Chikorita (#152), Cyndaquil (#155), Totodile (#158)
- **Johto Legendaries**: Raikou (#243), Entei (#244), Suicune (#245), Lugia (#249, Silver mascot), Ho-Oh (#250)
- **Mythical**: Celebi (#251)

---

## Notes on Silver-Specific Sprites

- Some Gen II Pokemon used Gold-palette sprites in the Bulbagarden archives (`Spr_2g_NNN.png` vs `Spr_2s_NNN.png`) — the PokeAPI silver folder provides the Silver-palette version where available.
- Sprites are 56×56 pixels in the GBC color palette.
- The Japanese release used slightly different sprites for a small number of Pokemon (e.g. #079 Slowpoke, #160 Feraligatr, #171 Lanturn, #172 Pichu, #190 Aipom).
