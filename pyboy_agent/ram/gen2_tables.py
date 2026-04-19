"""
pyboy_agent.ram.gen2_tables
===========================
Static lookup tables for Pokemon Gold/Silver (Generation 2).

All data is ROM-derived or documented at https://github.com/pret/pokegold.

Contents
--------
- _GEN2_CHAR    : byte → char mapping for Gen 2 text encoding (null-terminated)
- _GEN2_MOVE    : move ID 1-251 → move name
- _GEN2_TYPE    : type byte → type name (with Gen 2's gap at 9-19)
- _GEN2_SPECIES : National Dex ID 1-251 → species name (+ 253 = Egg)
- _JOHTO_BADGES : bitmask definitions for the 8 Johto gym badges
- _KANTO_BADGES : bitmask definitions for the 8 Kanto gym badges

Public helpers
--------------
- decode_gen2_name(pyboy, start_addr, length) → str
- read_bcd(pyboy, start_addr, length) → int
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyboy import PyBoy


# ---------------------------------------------------------------------------
# Gen 2 character encoding (subset for player names and location strings)
# ---------------------------------------------------------------------------

# Byte → ASCII-printable character.  0x50 is the string terminator (empty string).
# 0x7F is a space.  Digits 0-9 are encoded at 0xF6-0xFF.
_GEN2_CHAR: dict[int, str] = {
    0x80: "A", 0x81: "B", 0x82: "C", 0x83: "D", 0x84: "E", 0x85: "F", 0x86: "G",
    0x87: "H", 0x88: "I", 0x89: "J", 0x8A: "K", 0x8B: "L", 0x8C: "M", 0x8D: "N",
    0x8E: "O", 0x8F: "P", 0x90: "Q", 0x91: "R", 0x92: "S", 0x93: "T", 0x94: "U",
    0x95: "V", 0x96: "W", 0x97: "X", 0x98: "Y", 0x99: "Z",
    0xA0: "a", 0xA1: "b", 0xA2: "c", 0xA3: "d", 0xA4: "e", 0xA5: "f", 0xA6: "g",
    0xA7: "h", 0xA8: "i", 0xA9: "j", 0xAA: "k", 0xAB: "l", 0xAC: "m", 0xAD: "n",
    0xAE: "o", 0xAF: "p", 0xB0: "q", 0xB1: "r", 0xB2: "s", 0xB3: "t", 0xB4: "u",
    0xB5: "v", 0xB6: "w", 0xB7: "x", 0xB8: "y", 0xB9: "z",
    0xF6: "0", 0xF7: "1", 0xF8: "2", 0xF9: "3", 0xFA: "4",
    0xFB: "5", 0xFC: "6", 0xFD: "7", 0xFE: "8", 0xFF: "9",
    0x50: "",   # string terminator
    0x7F: " ",  # space
}


# ---------------------------------------------------------------------------
# Gen 2 moves (1-251, Gold/Silver)
# ---------------------------------------------------------------------------
# HM moves for reference:
#   Cut=15, Fly=19, Surf=57, Strength=70, Flash=148,
#   Whirlpool=250, Waterfall=127, Rock Smash=249

_GEN2_MOVE: dict[int, str] = {
    1: "Pound", 2: "Karate Chop", 3: "DoubleSlap", 4: "Comet Punch", 5: "Mega Punch",
    6: "Pay Day", 7: "Fire Punch", 8: "Ice Punch", 9: "ThunderPunch", 10: "Scratch",
    11: "ViceGrip", 12: "Guillotine", 13: "Razor Wind", 14: "Swords Dance", 15: "Cut",
    16: "Gust", 17: "Wing Attack", 18: "Whirlwind", 19: "Fly", 20: "Bind",
    21: "Slam", 22: "Vine Whip", 23: "Stomp", 24: "Double Kick", 25: "Mega Kick",
    26: "Jump Kick", 27: "Rolling Kick", 28: "Sand Attack", 29: "Headbutt", 30: "Horn Attack",
    31: "Fury Attack", 32: "Horn Drill", 33: "Tackle", 34: "Body Slam", 35: "Wrap",
    36: "Take Down", 37: "Thrash", 38: "Double-Edge", 39: "Tail Whip", 40: "Poison Sting",
    41: "Twineedle", 42: "Pin Missile", 43: "Leer", 44: "Bite", 45: "Growl",
    46: "Roar", 47: "Sing", 48: "Supersonic", 49: "SonicBoom", 50: "Disable",
    51: "Acid", 52: "Ember", 53: "Flamethrower", 54: "Mist", 55: "Water Gun",
    56: "Hydro Pump", 57: "Surf", 58: "Ice Beam", 59: "Blizzard", 60: "Psybeam",
    61: "BubbleBeam", 62: "Aurora Beam", 63: "Hyper Beam", 64: "Peck", 65: "Drill Peck",
    66: "Submission", 67: "Low Kick", 68: "Counter", 69: "Seismic Toss", 70: "Strength",
    71: "Absorb", 72: "Mega Drain", 73: "Leech Seed", 74: "Growth", 75: "Razor Leaf",
    76: "SolarBeam", 77: "PoisonPowder", 78: "Stun Spore", 79: "Sleep Powder", 80: "Petal Dance",
    81: "String Shot", 82: "Dragon Rage", 83: "Fire Spin", 84: "ThunderShock", 85: "Thunderbolt",
    86: "Thunder Wave", 87: "Thunder", 88: "Rock Throw", 89: "Earthquake", 90: "Fissure",
    91: "Dig", 92: "Toxic", 93: "Confusion", 94: "Psychic", 95: "Hypnosis",
    96: "Meditate", 97: "Agility", 98: "Quick Attack", 99: "Rage", 100: "Teleport",
    101: "Night Shade", 102: "Mimic", 103: "Screech", 104: "Double Team", 105: "Recover",
    106: "Harden", 107: "Minimize", 108: "Smokescreen", 109: "Confuse Ray", 110: "Withdraw",
    111: "Defense Curl", 112: "Barrier", 113: "Light Screen", 114: "Haze", 115: "Reflect",
    116: "Focus Energy", 117: "Bide", 118: "Metronome", 119: "Mirror Move", 120: "Selfdestruct",
    121: "Egg Bomb", 122: "Lick", 123: "Smog", 124: "Sludge", 125: "Bone Club",
    126: "Fire Blast", 127: "Waterfall", 128: "Clamp", 129: "Swift", 130: "Skull Bash",
    131: "Spike Cannon", 132: "Constrict", 133: "Amnesia", 134: "Kinesis", 135: "Softboiled",
    136: "Hi Jump Kick", 137: "Glare", 138: "Dream Eater", 139: "Poison Gas", 140: "Barrage",
    141: "Leech Life", 142: "Lovely Kiss", 143: "Sky Attack", 144: "Transform", 145: "Bubble",
    146: "Dizzy Punch", 147: "Spore", 148: "Flash", 149: "Psywave", 150: "Splash",
    151: "Acid Armor", 152: "Crabhammer", 153: "Explosion", 154: "Fury Swipes", 155: "Bonemerang",
    156: "Rest", 157: "Rock Slide", 158: "Hyper Fang", 159: "Sharpen", 160: "Conversion",
    161: "Tri Attack", 162: "Super Fang", 163: "Slash", 164: "Substitute", 165: "Struggle",
    166: "Sketch", 167: "Triple Kick", 168: "Thief", 169: "Spider Web", 170: "Mind Reader",
    171: "Nightmare", 172: "Flame Wheel", 173: "Snore", 174: "Curse", 175: "Flail",
    176: "Conversion 2", 177: "Aeroblast", 178: "Cotton Spore", 179: "Reversal", 180: "Spite",
    181: "Powder Snow", 182: "Protect", 183: "Mach Punch", 184: "Scary Face", 185: "Faint Attack",
    186: "Sweet Kiss", 187: "Belly Drum", 188: "Sludge Bomb", 189: "Mud-Slap", 190: "Octazooka",
    191: "Spikes", 192: "Zap Cannon", 193: "Foresight", 194: "Destiny Bond", 195: "Perish Song",
    196: "Icy Wind", 197: "Detect", 198: "Bone Rush", 199: "Lock-On", 200: "Outrage",
    201: "Sandstorm", 202: "Giga Drain", 203: "Endure", 204: "Charm", 205: "Rollout",
    206: "False Swipe", 207: "Swagger", 208: "Milk Drink", 209: "Spark", 210: "Fury Cutter",
    211: "Steel Wing", 212: "Mean Look", 213: "Attract", 214: "Sleep Talk", 215: "Heal Bell",
    216: "Return", 217: "Present", 218: "Frustration", 219: "Safeguard", 220: "Pain Split",
    221: "Sacred Fire", 222: "Magnitude", 223: "DynamicPunch", 224: "Megahorn", 225: "DragonBreath",
    226: "Baton Pass", 227: "Encore", 228: "Pursuit", 229: "Rapid Spin", 230: "Sweet Scent",
    231: "Iron Tail", 232: "Metal Claw", 233: "Vital Throw", 234: "Morning Sun", 235: "Synthesis",
    236: "Moonlight", 237: "Hidden Power", 238: "Cross Chop", 239: "Twister", 240: "Rain Dance",
    241: "Sunny Day", 242: "Crunch", 243: "Mirror Coat", 244: "Psych Up", 245: "ExtremeSpeed",
    246: "AncientPower", 247: "Shadow Ball", 248: "Future Sight", 249: "Rock Smash",
    250: "Whirlpool", 251: "Beat Up",
}


# ---------------------------------------------------------------------------
# Gen 2 types (note: IDs 9-19 are unused in Gen 2 — intentional gap)
# ---------------------------------------------------------------------------

_GEN2_TYPE: dict[int, str] = {
    0: "Normal", 1: "Fighting", 2: "Flying", 3: "Poison",
    4: "Ground", 5: "Rock", 6: "Bug", 7: "Ghost", 8: "Steel",
    # IDs 9-19 do not exist in Gen 2 (Gen 1 used Bird/??? types that were removed)
    20: "Fire", 21: "Water", 22: "Grass", 23: "Electric",
    24: "Psychic", 25: "Ice", 26: "Dragon", 27: "Dark",
}


# ---------------------------------------------------------------------------
# Gen 2 species table
# ---------------------------------------------------------------------------

# National Pokedex ID → species name.  In Gold/Silver party/battle data,
# species IDs are stored directly as National Dex numbers (1-251).
# 0xFD (253) is the Egg placeholder.
_GEN2_SPECIES: dict[int, str] = {
    1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur", 4: "Charmander",
    5: "Charmeleon", 6: "Charizard", 7: "Squirtle", 8: "Wartortle",
    9: "Blastoise", 10: "Caterpie", 11: "Metapod", 12: "Butterfree",
    13: "Weedle", 14: "Kakuna", 15: "Beedrill", 16: "Pidgey",
    17: "Pidgeotto", 18: "Pidgeot", 19: "Rattata", 20: "Raticate",
    21: "Spearow", 22: "Fearow", 23: "Ekans", 24: "Arbok",
    25: "Pikachu", 26: "Raichu", 27: "Sandshrew", 28: "Sandslash",
    29: "Nidoran-F", 30: "Nidorina", 31: "Nidoqueen", 32: "Nidoran-M",
    33: "Nidorino", 34: "Nidoking", 35: "Clefairy", 36: "Clefable",
    37: "Vulpix", 38: "Ninetales", 39: "Jigglypuff", 40: "Wigglytuff",
    41: "Zubat", 42: "Golbat", 43: "Oddish", 44: "Gloom",
    45: "Vileplume", 46: "Paras", 47: "Parasect", 48: "Venonat",
    49: "Venomoth", 50: "Diglett", 51: "Dugtrio", 52: "Meowth",
    53: "Persian", 54: "Psyduck", 55: "Golduck", 56: "Mankey",
    57: "Primeape", 58: "Growlithe", 59: "Arcanine", 60: "Poliwag",
    61: "Poliwhirl", 62: "Poliwrath", 63: "Abra", 64: "Kadabra",
    65: "Alakazam", 66: "Machop", 67: "Machoke", 68: "Machamp",
    69: "Bellsprout", 70: "Weepinbell", 71: "Victreebel", 72: "Tentacool",
    73: "Tentacruel", 74: "Geodude", 75: "Graveler", 76: "Golem",
    77: "Ponyta", 78: "Rapidash", 79: "Slowpoke", 80: "Slowbro",
    81: "Magnemite", 82: "Magneton", 83: "Farfetchd", 84: "Doduo",
    85: "Dodrio", 86: "Seel", 87: "Dewgong", 88: "Grimer",
    89: "Muk", 90: "Shellder", 91: "Cloyster", 92: "Gastly",
    93: "Haunter", 94: "Gengar", 95: "Onix", 96: "Drowzee",
    97: "Hypno", 98: "Krabby", 99: "Kingler", 100: "Voltorb",
    101: "Electrode", 102: "Exeggcute", 103: "Exeggutor", 104: "Cubone",
    105: "Marowak", 106: "Hitmonlee", 107: "Hitmonchan", 108: "Lickitung",
    109: "Koffing", 110: "Weezing", 111: "Rhyhorn", 112: "Rhydon",
    113: "Chansey", 114: "Tangela", 115: "Kangaskhan", 116: "Horsea",
    117: "Seadra", 118: "Goldeen", 119: "Seaking", 120: "Staryu",
    121: "Starmie", 122: "Mr. Mime", 123: "Scyther", 124: "Jynx",
    125: "Electabuzz", 126: "Magmar", 127: "Pinsir", 128: "Tauros",
    129: "Magikarp", 130: "Gyarados", 131: "Lapras", 132: "Ditto",
    133: "Eevee", 134: "Vaporeon", 135: "Jolteon", 136: "Flareon",
    137: "Porygon", 138: "Omanyte", 139: "Omastar", 140: "Kabuto",
    141: "Kabutops", 142: "Aerodactyl", 143: "Snorlax", 144: "Articuno",
    145: "Zapdos", 146: "Moltres", 147: "Dratini", 148: "Dragonair",
    149: "Dragonite", 150: "Mewtwo", 151: "Mew",
    152: "Chikorita", 153: "Bayleef", 154: "Meganium",
    155: "Cyndaquil", 156: "Quilava", 157: "Typhlosion",
    158: "Totodile", 159: "Croconaw", 160: "Feraligatr",
    161: "Sentret", 162: "Furret", 163: "Hoothoot", 164: "Noctowl",
    165: "Ledyba", 166: "Ledian", 167: "Spinarak", 168: "Ariados",
    169: "Crobat", 170: "Chinchou", 171: "Lanturn", 172: "Pichu",
    173: "Cleffa", 174: "Igglybuff", 175: "Togepi", 176: "Togetic",
    177: "Natu", 178: "Xatu", 179: "Mareep", 180: "Flaaffy",
    181: "Ampharos", 182: "Bellossom", 183: "Marill", 184: "Azumarill",
    185: "Sudowoodo", 186: "Politoed", 187: "Hoppip", 188: "Skiploom",
    189: "Jumpluff", 190: "Aipom", 191: "Sunkern", 192: "Sunflora",
    193: "Yanma", 194: "Wooper", 195: "Quagsire", 196: "Espeon",
    197: "Umbreon", 198: "Murkrow", 199: "Slowking", 200: "Misdreavus",
    201: "Unown", 202: "Wobbuffet", 203: "Girafarig", 204: "Pineco",
    205: "Forretress", 206: "Dunsparce", 207: "Gligar", 208: "Steelix",
    209: "Snubbull", 210: "Granbull", 211: "Qwilfish", 212: "Scizor",
    213: "Shuckle", 214: "Heracross", 215: "Sneasel", 216: "Teddiursa",
    217: "Ursaring", 218: "Slugma", 219: "Magcargo", 220: "Swinub",
    221: "Piloswine", 222: "Corsola", 223: "Remoraid", 224: "Octillery",
    225: "Delibird", 226: "Mantine", 227: "Skarmory", 228: "Houndour",
    229: "Houndoom", 230: "Kingdra", 231: "Phanpy", 232: "Donphan",
    233: "Porygon2", 234: "Stantler", 235: "Smeargle", 236: "Tyrogue",
    237: "Hitmontop", 238: "Smoochum", 239: "Elekid", 240: "Magby",
    241: "Miltank", 242: "Blissey", 243: "Raikou", 244: "Entei",
    245: "Suicune", 246: "Larvitar", 247: "Pupitar", 248: "Tyranitar",
    249: "Lugia", 250: "Ho-Oh", 251: "Celebi",
    253: "Egg",  # 0xFD placeholder used for eggs in party data
}


# ---------------------------------------------------------------------------
# Badge bitmask tables
# ---------------------------------------------------------------------------

# Each tuple is (bit_mask, display_name). Read from Johto/Kanto badge bytes.
_JOHTO_BADGES: list[tuple[int, str]] = [
    (0x01, "Zephyr (Falkner)"),
    (0x02, "Hive (Bugsy)"),
    (0x04, "Plain (Whitney)"),
    (0x08, "Fog (Morty)"),
    (0x10, "Mineral (Jasmine)"),
    (0x20, "Storm (Chuck)"),
    (0x40, "Glacier (Pryce)"),
    (0x80, "Rising (Clair)"),
]

_KANTO_BADGES: list[tuple[int, str]] = [
    (0x01, "Boulder (Brock)"),
    (0x02, "Cascade (Misty)"),
    (0x04, "Thunder (Lt. Surge)"),
    (0x08, "Rainbow (Erika)"),
    (0x10, "Soul (Janine)"),
    (0x20, "Marsh (Sabrina)"),
    (0x40, "Volcano (Blaine)"),
    (0x80, "Earth (Blue)"),
]


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def decode_gen2_name(pyboy: "PyBoy", start_addr: int, length: int) -> str:
    """Decode a Gen 2 encoded string from emulator RAM.

    Reads up to ``length`` bytes starting at ``start_addr``.  Stops early at
    the 0x50 string terminator.  Unknown bytes are rendered as ``"?"``.

    Args:
        pyboy: Running PyBoy instance.
        start_addr: WRAM address of the first character byte.
        length: Maximum number of characters to read (typically 10 for names).

    Returns:
        Decoded ASCII string, empty string for untitled or null entries.
    """
    chars: list[str] = []
    for i in range(length):
        b = pyboy.memory[start_addr + i]
        if b == 0x50:
            break
        chars.append(_GEN2_CHAR.get(b, "?"))
    return "".join(chars)


def read_bcd(pyboy: "PyBoy", start_addr: int, length: int) -> int:
    """Read a packed BCD (Binary Coded Decimal) integer from RAM.

    Gen 2 stores money as 3-byte BCD: each nibble encodes one decimal digit.
    For example, ¥12,345 is stored as 0x01 0x23 0x45.

    Args:
        pyboy: Running PyBoy instance.
        start_addr: WRAM address of the most-significant byte.
        length: Number of BCD bytes to read (typically 3 for money).

    Returns:
        Decoded integer value.
    """
    val = 0
    for i in range(length):
        b = pyboy.memory[start_addr + i]
        val = val * 100 + (((b >> 4) & 0xF) * 10) + (b & 0xF)
    return val
