"""
Pokemon Silver Sprite Downloader
=================================
Downloads all 251 Pokemon battle sprites from Pokemon Silver (GBC, 2000)
and generates JSON metadata files for each Pokemon.

Sources:
- Sprites:  PokeAPI/sprites on GitHub (generation-ii/silver)
- Metadata: PokeAPI REST (pokeapi.co)

Output:
- sprites/NNN-name.png   - front battle sprite from Pokemon Silver
- data/NNN-name.json     - metadata JSON for each Pokemon
- index.json             - master index of all sprites
"""

import json
import os
import time
import urllib.request
import urllib.error

# ── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITES_DIR = os.path.join(BASE_DIR, "sprites")
DATA_DIR = os.path.join(BASE_DIR, "data")

POKEAPI_BASE = "https://pokeapi.co/api/v2"
SPRITE_BASE = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master"
    "/sprites/pokemon/versions/generation-ii/silver"
)

# Pokemon Silver (GBC) contains all 251 Gen I + Gen II Pokemon
TOTAL_POKEMON = 251

REQUEST_DELAY = 0.5   # seconds between Pokemon to avoid rate limits
API_DELAY = 0.25      # seconds between the two API calls per Pokemon


# ── Helpers ─────────────────────────────────────────────────────────────────

def fetch_json(url: str, retries: int = 2) -> dict | None:
    """Fetch a JSON endpoint, return parsed dict or None on error."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PokemonSilverResearch/1.0 (research project)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # resource does not exist
            print(f"  HTTP {e.code} on {url} (attempt {attempt + 1})")
        except Exception as e:
            print(f"  Error fetching {url}: {e} (attempt {attempt + 1})")
        if attempt < retries - 1:
            time.sleep(1)
    return None


def download_binary(url: str, dest_path: str, retries: int = 2) -> bool:
    """Download a binary file (PNG), return True on success."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PokemonSilverResearch/1.0 (research project)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            print(f"  HTTP {e.code} downloading {url} (attempt {attempt + 1})")
        except Exception as e:
            print(f"  Error downloading {url}: {e} (attempt {attempt + 1})")
        if attempt < retries - 1:
            time.sleep(1)
    return False


def pick_flavor_text(flavor_entries: list, preferred_versions: list) -> tuple[str, str]:
    """
    Return (text, version_name) for the first matching preferred version.
    Falls back to any English entry if none match.
    """
    english = [e for e in flavor_entries if e.get("language", {}).get("name") == "en"]
    for version in preferred_versions:
        for entry in english:
            if entry.get("version", {}).get("name") == version:
                raw = entry["flavor_text"].replace("\n", " ").replace("\f", " ")
                return raw, version
    # fallback: first english entry
    if english:
        raw = english[0]["flavor_text"].replace("\n", " ").replace("\f", " ")
        return raw, english[0].get("version", {}).get("name", "unknown")
    return "", "none"


def capitalize_name(name: str) -> str:
    """Convert api slug 'mr-mime' -> 'Mr. Mime'."""
    replacements = {
        "mr-mime": "Mr. Mime",
        "farfetchd": "Farfetch'd",
        "nidoran-f": "Nidoran♀",
        "nidoran-m": "Nidoran♂",
        "ho-oh": "Ho-Oh",
    }
    if name in replacements:
        return replacements[name]
    return " ".join(part.capitalize() for part in name.split("-"))


# ── Main ────────────────────────────────────────────────────────────────────

def process_pokemon(pokemon_id: int) -> dict | None:
    """
    Fetch data, download sprite, write JSON for a single Pokemon.
    Returns the index entry dict, or None if skipped.
    """
    # 1) Fetch basic pokemon data
    poke_data = fetch_json(f"{POKEAPI_BASE}/pokemon/{pokemon_id}")
    if poke_data is None:
        print(f"  [{pokemon_id:03d}] Could not fetch pokemon data — skipping.")
        return None

    time.sleep(API_DELAY)

    # 2) Fetch species data (for flavor text, categories, etc.)
    species_data = fetch_json(f"{POKEAPI_BASE}/pokemon-species/{pokemon_id}")
    if species_data is None:
        print(f"  [{pokemon_id:03d}] Could not fetch species data — skipping.")
        return None

    # Extract name
    api_name = poke_data["name"]
    display_name = capitalize_name(api_name)
    safe_name = api_name.replace("'", "").replace(".", "").replace(" ", "-")
    file_stem = f"{pokemon_id:03d}-{safe_name}"

    # 3) Download sprite
    sprite_url = f"{SPRITE_BASE}/{pokemon_id}.png"
    sprite_filename = f"{file_stem}.png"
    sprite_path = os.path.join(SPRITES_DIR, sprite_filename)
    sprite_downloaded = download_binary(sprite_url, sprite_path)
    if not sprite_downloaded:
        print(f"  [{pokemon_id:03d}] Sprite not found at {sprite_url} — recording as missing.")

    # 4) Parse metadata
    types = [t["type"]["name"].capitalize() for t in poke_data.get("types", [])]
    abilities = [
        a["ability"]["name"].replace("-", " ").title()
        for a in poke_data.get("abilities", [])
        if not a["is_hidden"]
    ]
    hidden_abilities = [
        a["ability"]["name"].replace("-", " ").title()
        for a in poke_data.get("abilities", [])
        if a["is_hidden"]
    ]
    stats = {
        s["stat"]["name"].replace("-", "_"): s["base_stat"]
        for s in poke_data.get("stats", [])
    }

    # Genus (e.g. "Seed Pokémon")
    genus = ""
    for g in species_data.get("genera", []):
        if g.get("language", {}).get("name") == "en":
            genus = g["genus"]
            break

    # Flavor text — prefer Silver, then Gold, then Crystal, then any Gen II
    flavor_text, flavor_version = pick_flavor_text(
        species_data.get("flavor_text_entries", []),
        ["silver", "gold", "crystal"]
    )

    # Evolution and lineage
    evolves_from = None
    if species_data.get("evolves_from_species"):
        evolves_from = capitalize_name(species_data["evolves_from_species"]["name"])

    is_legendary = species_data.get("is_legendary", False)
    is_mythical = species_data.get("is_mythical", False)
    habitat = (species_data.get("habitat") or {}).get("name", "unknown")
    capture_rate = species_data.get("capture_rate")
    base_happiness = species_data.get("base_happiness")
    growth_rate = (species_data.get("growth_rate") or {}).get("name", "unknown")

    # 5) Build JSON record
    record = {
        "pokedex_number": pokemon_id,
        "name": display_name,
        "api_name": api_name,
        "category": genus,
        "types": types,
        "abilities": abilities,
        "hidden_abilities": hidden_abilities,
        "height_dm": poke_data.get("height"),
        "weight_hg": poke_data.get("weight"),
        "base_stats": {
            "hp": stats.get("hp"),
            "attack": stats.get("attack"),
            "defense": stats.get("defense"),
            "special_attack": stats.get("special_attack"),
            "special_defense": stats.get("special_defense"),
            "speed": stats.get("speed"),
        },
        "base_experience": poke_data.get("base_experience"),
        "capture_rate": capture_rate,
        "base_happiness": base_happiness,
        "growth_rate": growth_rate,
        "flavor_text": flavor_text,
        "flavor_text_version": flavor_version,
        "evolves_from": evolves_from,
        "is_legendary": is_legendary,
        "is_mythical": is_mythical,
        "habitat": habitat,
        "sprite_file": f"../sprites/{sprite_filename}",
        "sprite_url": sprite_url,
        "sprite_available": sprite_downloaded,
        "game": "Pokémon Silver Version",
        "platform": "Game Boy Color",
        "release_year": 2000,
        "generation": "Generation II",
    }

    # 6) Write individual JSON file
    json_path = os.path.join(DATA_DIR, f"{file_stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(
        f"  [{pokemon_id:03d}] {display_name:<16} "
        f"types={','.join(types):<16} "
        f"sprite={'OK' if sprite_downloaded else 'MISSING'}"
    )

    return {
        "pokedex_number": pokemon_id,
        "name": display_name,
        "types": types,
        "sprite_file": f"sprites/{sprite_filename}",
        "data_file": f"data/{file_stem}.json",
        "sprite_available": sprite_downloaded,
    }


def main():
    os.makedirs(SPRITES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Pokemon Silver Sprite Downloader")
    print(f"Downloading {TOTAL_POKEMON} Pokemon sprites + metadata...")
    print(f"Output: {BASE_DIR}")
    print("-" * 60)

    index_entries = []
    failed = []

    for pokemon_id in range(1, TOTAL_POKEMON + 1):
        entry = process_pokemon(pokemon_id)
        if entry:
            index_entries.append(entry)
        else:
            failed.append(pokemon_id)
        time.sleep(REQUEST_DELAY)

    # Write master index.json
    index_path = os.path.join(BASE_DIR, "index.json")
    index_data = {
        "project": "Pokemon Silver Sprites Research",
        "game": "Pokémon Silver Version",
        "platform": "Game Boy Color",
        "release_year": 2000,
        "generation": "Generation II",
        "total_pokemon": TOTAL_POKEMON,
        "downloaded_count": len(index_entries),
        "failed_ids": failed,
        "sprite_source": "PokeAPI/sprites (GitHub) — generation-ii/silver",
        "metadata_source": "PokéAPI REST v2 (pokeapi.co)",
        "pokemon": index_entries,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"Done! {len(index_entries)}/{TOTAL_POKEMON} Pokemon processed.")
    if failed:
        print(f"Failed IDs: {failed}")
    print(f"Master index written to: index.json")


if __name__ == "__main__":
    main()
