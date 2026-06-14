"""
Pokemon Gold/Silver In-Game Sprite Downloader
Source: pret/pokegold (https://github.com/pret/pokegold)
Downloads tilesets, overworld elements, NPC sprites, and animated tiles.
"""

import urllib.request
import json
import os
import time

BASE_URL = "https://raw.githubusercontent.com/pret/pokegold/master/gfx"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game-sprites")

# ─────────────────────────────────────────────────────────────────────────────
# DATA: MAIN TILESETS
# Each tuple: (filename, display_name, description)
# These are sprite sheets containing all tiles for a given environment.
# ─────────────────────────────────────────────────────────────────────────────
TILESETS = [
    ("cave.png", "Cave tileset",
     "Tileset for cave interiors: Mt. Mortar, Slowpoke Well, Tohjo Falls, Union Cave. "
     "Contains stalactites, ledges, rock walls, boulders, and cave floors."),
    ("champions_room.png", "Champion's Room tileset",
     "Tileset for the final Champion's room in the Pokemon League. Lance's arena. "
     "Ornate floor tiles and pillars mark the destination of the main story."),
    ("dark_cave.png", "Dark Cave tileset",
     "Tileset for Dark Cave near Violet City. Identical to cave but pitch-black until "
     "HM05 Flash is used, revealing hidden items and Pokemon like Dunsparce."),
    ("elite_four_room.png", "Elite Four room tileset",
     "Tileset for the four Elite Four chambers at the Pokemon League. Each room has "
     "a distinct style matching its trainer's type specialty."),
    ("facility.png", "Facility tileset",
     "Tileset used in miscellaneous facilities such as the Safari Zone gate and "
     "special buildings not covered by other tilesets."),
    ("forest.png", "Forest (Ilex Forest) tileset",
     "Tileset for Ilex Forest — the first major forest in Johto. Contains dense trees, "
     "cut stumps, grass, and the shrine to the legendary Pokemon Celebi."),
    ("game_corner.png", "Game Corner tileset",
     "Tileset for the Goldenrod Game Corner casino. Slot machine floors, neon counters, "
     "and the prize exchange desk. Team Rocket hides a base beneath it in Kanto."),
    ("gate.png", "Route Gate tileset",
     "Tileset for route gates and checkpoints between routes. Also used at the "
     "Mahogany Town gate leading to the Lake of Rage."),
    ("house.png", "House interior tileset",
     "Standard house interior tileset used in homes throughout Johto and Kanto. "
     "Contains furniture, bookshelves, doors, and floor patterns."),
    ("ice_path.png", "Ice Path tileset",
     "Tileset for Ice Path — a sliding-ice-puzzle cave east of Mahogany Town. "
     "Contains icy floors, boulders, and rock walls. Required to reach Blackthorn City."),
    ("johto.png", "Johto outdoor tileset",
     "The primary outdoor tileset for Johto's routes and towns. Contains tall grass, "
     "ledges (one-way drops), cut trees, signs, building doors, paths, water, and "
     "flowers. This sheet contains the ledge and tree tiles the player interacts with most."),
    ("johto_modern.png", "Johto modern outdoor tileset",
     "An alternate version of the Johto outdoor tileset used in certain towns and "
     "later-game areas with a slightly different visual style."),
    ("kanto.png", "Kanto outdoor tileset",
     "The outdoor tileset for the post-game Kanto region — routes and towns visited "
     "after defeating the Elite Four. Similar to the Johto tileset but with Kanto-specific elements."),
    ("lab.png", "Laboratory tileset",
     "Tileset for Professor Elm's lab in New Bark Town and similar research buildings. "
     "Contains lab benches, computers, bookshelves, and equipment."),
    ("lighthouse.png", "Lighthouse tileset",
     "Tileset for the Olivine Lighthouse (Glitter Lighthouse). Spiral staircases, "
     "beacon equipment, and railings. Jasmine nurses a sick Ampharos at the top."),
    ("mansion.png", "Mansion interior tileset",
     "Tileset for large mansion-style interiors, including the Mahogany Town rocket "
     "base and other elaborate buildings."),
    ("mart.png", "Poke Mart tileset",
     "Tileset for all Poke Mart interiors. Counter desks, shelves, and the cash "
     "register area where items like Potions and Poke Balls are purchased."),
    ("park.png", "National Park tileset",
     "Tileset for the National Park near Goldenrod City — site of the Bug Catching "
     "Contest. Contains manicured paths, park benches, and enclosed grass areas."),
    ("players_house.png", "Player's house tileset",
     "Tileset for the player character's house in New Bark Town. Ground floor with "
     "TV, mom's kitchen, and staircase to the player's room."),
    ("players_room.png", "Player's bedroom tileset",
     "Tileset for the player's bedroom where the game begins. Contains a bed, "
     "console, computer (for PC storage), and a map of the Johto region on the wall."),
    ("pokecenter.png", "Pokemon Center tileset",
     "Tileset for all Pokemon Center interiors. Contains the healing counter, "
     "Nurse Joy's desk, the PC terminal, and waiting area. Fully heals the party for free."),
    ("port.png", "Port tileset",
     "Tileset for port and harbor areas including Olivine City's dock and the "
     "S.S. Aqua boarding area. Contains gangplanks, bollards, and water docks."),
    ("radio_tower.png", "Radio Tower tileset",
     "Tileset for the Goldenrod Radio Tower — a major story location. Team Rocket "
     "takes it over broadcasting a signal to call Giovanni back. Five floors of staff and rocket grunts."),
    ("ruins_of_alph.png", "Ruins of Alph tileset",
     "Tileset for the Ruins of Alph — ancient ruins with Unown puzzles. Completing "
     "each puzzle unlocks areas swarming with the 26 different Unown letter Pokemon."),
    ("tower.png", "Tower tileset",
     "Tileset for Sprout Tower in Violet City and the Burned Tower in Ecruteak City. "
     "The Burned Tower is where the three legendary beasts (Raikou, Entei, Suicune) were resurrected."),
    ("traditional_house.png", "Traditional house tileset",
     "Tileset for Japanese-style traditional house interiors, used in Ecruteak City "
     "and similar culturally-themed locations."),
    ("train_station.png", "Train Station tileset",
     "Tileset for Goldenrod City's train station. The Magnet Train connects "
     "Goldenrod (Johto) to Saffron City (Kanto) — requires the Pass key item to board."),
    ("underground.png", "Underground tileset",
     "Tileset for the Goldenrod Underground shopping area beneath the city. "
     "Contains shop stalls, a hair salon, and the Goldenrod Game Corner entrance."),
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA: ANIMATED TILE SETS (subdirectories)
# ─────────────────────────────────────────────────────────────────────────────
TILESET_ANIM = [
    # flower/
    ("flower/cgb_1.png", "Flower animation frame 1 (GBC color)",
     "First frame of the animated flower tiles in GBC color mode. Flowers sway on routes."),
    ("flower/cgb_2.png", "Flower animation frame 2 (GBC color)",
     "Second frame of the animated flower tiles in GBC color mode."),
    ("flower/dmg_1.png", "Flower animation frame 1 (DMG grayscale)",
     "First frame of the flower animation for original Game Boy (grayscale) mode."),
    ("flower/dmg_2.png", "Flower animation frame 2 (DMG grayscale)",
     "Second frame of the flower animation for original Game Boy (grayscale) mode."),
    # lava/
    ("lava/1.png", "Lava animation frame 1",
     "First frame of the animated lava tiles. Lava appears in the post-game Kanto "
     "region near Mt. Silver and the Cinnabar Island area."),
    ("lava/2.png", "Lava animation frame 2", "Second frame of the lava animation loop."),
    ("lava/3.png", "Lava animation frame 3", "Third frame of the lava animation loop."),
    ("lava/4.png", "Lava animation frame 4", "Fourth frame of the lava animation loop."),
    # roofs/
    ("roofs/azalea.png", "Azalea Town roof tiles",
     "Roof color tiles for buildings in Azalea Town, home of Kurt the Poke Ball craftsman."),
    ("roofs/goldenrod.png", "Goldenrod City roof tiles",
     "Roof color tiles for buildings in Goldenrod City — the largest city in Johto."),
    ("roofs/new_bark.png", "New Bark Town roof tiles",
     "Roof color tiles for buildings in New Bark Town — the player's starting hometown."),
    ("roofs/olivine.png", "Olivine City roof tiles",
     "Roof color tiles for buildings in Olivine City, the port town with the Glitter Lighthouse."),
    ("roofs/violet.png", "Violet City roof tiles",
     "Roof color tiles for buildings in Violet City, the first Gym town. Home of Sprout Tower."),
    # tower-pillar/ (10 frames of pillar animation)
    ("tower-pillar/1.png", "Tower pillar animation frame 1",
     "Animated pillar tile frame 1, used in Sprout Tower and Burned Tower."),
    ("tower-pillar/2.png", "Tower pillar animation frame 2", "Animated pillar tile frame 2."),
    ("tower-pillar/3.png", "Tower pillar animation frame 3", "Animated pillar tile frame 3."),
    ("tower-pillar/4.png", "Tower pillar animation frame 4", "Animated pillar tile frame 4."),
    ("tower-pillar/5.png", "Tower pillar animation frame 5", "Animated pillar tile frame 5."),
    ("tower-pillar/6.png", "Tower pillar animation frame 6", "Animated pillar tile frame 6."),
    ("tower-pillar/7.png", "Tower pillar animation frame 7", "Animated pillar tile frame 7."),
    ("tower-pillar/8.png", "Tower pillar animation frame 8", "Animated pillar tile frame 8."),
    ("tower-pillar/9.png", "Tower pillar animation frame 9", "Animated pillar tile frame 9."),
    ("tower-pillar/10.png", "Tower pillar animation frame 10", "Animated pillar tile frame 10."),
    # water/
    ("water/water.png", "Water animation tiles",
     "Animated water surface tiles — the rippling blue water used on routes, lakes, "
     "and the ocean. Required for HM03 Surf travel."),
    # whirlpool/
    ("whirlpool/1.png", "Whirlpool animation frame 1",
     "First frame of whirlpool animation. Whirlpools block certain water routes "
     "and are cleared using HM06 Whirlpool, learned from Pryce after the 7th badge."),
    ("whirlpool/2.png", "Whirlpool animation frame 2", "Second frame of whirlpool animation."),
    ("whirlpool/3.png", "Whirlpool animation frame 3", "Third frame of whirlpool animation."),
    ("whirlpool/4.png", "Whirlpool animation frame 4", "Fourth frame of whirlpool animation."),
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA: OVERWORLD INTERACTIVE ELEMENTS
# ─────────────────────────────────────────────────────────────────────────────
OVERWORLD = [
    ("boulder_dust.png", "Boulder dust animation",
     "Dust cloud animation that appears when the player pushes a boulder using "
     "HM04 Strength. Boulders must be pushed into holes to solve cave puzzles."),
    ("chris_fish.png", "Fishing animation (Kris / female player)",
     "Overworld fishing animation frames for the female player character (Kris) "
     "casting a fishing rod into water to encounter wild Pokemon."),
    ("cut_grass.png", "Cut grass animation",
     "Animation showing grass being sliced when HM01 Cut is used on tall grass. "
     "Cut reveals items and clears the tile temporarily."),
    ("cut_tree.png", "Cut tree sprite",
     "The overworld sprite for a small tree that can be cut using HM01 Cut. "
     "These small saplings block many route entrances and shortcuts throughout Johto and Kanto."),
    ("fishing_rod.png", "Fishing rod sprite",
     "The overworld sprite of the fishing rod being cast. Three versions exist: "
     "Old Rod (Magikarp only), Good Rod (Poliwag, Goldeen etc.), Super Rod (rare Pokemon)."),
    ("grass_rustle.png", "Tall grass rustle animation",
     "The animation of tall grass rustling when a wild Pokemon encounter is triggered "
     "while walking through grass on any route."),
    ("headbutt_tree.png", "Headbutt tree animation",
     "Animation frames for trees when the move Headbutt is used on them. "
     "Some trees shake loose overworld Pokemon like Heracross, Aipom, and Pineco."),
    ("heal_machine.png", "Pokemon Center healing machine",
     "The large Nurse Joy healing machine inside every Pokemon Center. "
     "Placing the party on it fully restores all HP and PP for free."),
    ("shadow.png", "Character shadow sprite",
     "The small oval shadow that appears under the player character and NPCs "
     "while walking on the overworld."),
    ("trainer_battle_pokeball_tiles.png", "Trainer battle Poke Ball animation",
     "The Poke Ball throw animation tiles used at the start of all trainer battles. "
     "The trainer throws their ball which expands into the battle screen."),
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA: NPC AND CHARACTER SPRITES
# ─────────────────────────────────────────────────────────────────────────────
SPRITES = [
    ("beauty.png", "Beauty trainer",
     "Overworld sprite for the Beauty trainer class — fashionable women found in "
     "cities who use Clefairy, Flaaffy, and similar Pokemon."),
    ("big_lapras.png", "Large Lapras decoration",
     "Oversized Lapras sprite used as a room decoration item in the player's bedroom."),
    ("big_onix.png", "Large Onix decoration",
     "Oversized Onix sprite used as a room decoration item."),
    ("big_snorlax.png", "Large Snorlax decoration",
     "Oversized Snorlax sprite used as a room decoration item."),
    ("biker.png", "Biker trainer",
     "Overworld sprite for the Biker trainer class — found on cycling roads, "
     "using Poison-type and Dark-type Pokemon."),
    ("bill.png", "Bill",
     "Bill, the PC Box inventor from Goldenrod City. Gives the player an Eevee "
     "after collecting all Johto Pokedex data. His system connects all Pokemon Centers."),
    ("bird.png", "Bird NPC sprite",
     "Generic bird sprite that appears perched on rooftops and in towns as decoration."),
    ("black_belt.png", "Black Belt trainer",
     "Overworld sprite for the Black Belt trainer class — Fighting-type specialists "
     "found in gyms and on routes."),
    ("blaine.png", "Blaine",
     "Blaine, the Cinnabar Island Gym Leader who now operates from the Seafoam Islands "
     "after Cinnabar's volcano erupted. Specializes in Fire-type Pokemon."),
    ("blue.png", "Blue (Gary Oak)",
     "Blue (Gary Oak), who becomes the Viridian City Gym Leader in the post-game. "
     "Grandson of Professor Oak and rival from the original Red/Blue games."),
    ("boulder.png", "Boulder / Strength rock",
     "The pushable boulder overworld sprite used in Strength puzzles throughout "
     "caves. Must be pushed into holes to create paths or solve puzzles."),
    ("brock.png", "Brock",
     "Brock, the Pewter City Gym Leader specializing in Rock-type Pokemon. "
     "One of the original Kanto Gym Leaders, reappears in the post-game."),
    ("bruno.png", "Bruno",
     "Bruno, the second Elite Four member specializing in Fighting-type Pokemon. "
     "Uses Hitmonchan, Hitmonlee, Hitmontop, and Machamp."),
    ("bug_catcher.png", "Bug Catcher trainer",
     "Overworld sprite for the Bug Catcher trainer class — young trainers who "
     "specialize in Bug-type Pokemon like Caterpie and Beedrill."),
    ("bugsy.png", "Bugsy",
     "Bugsy, the Azalea Town Gym Leader specializing in Bug-type Pokemon. "
     "Second Gym Leader in Johto; his ace is Scyther. Gives the Hive Badge."),
    ("cal.png", "Cal",
     "Cal, a trainer encountered in the Pokemon League's training areas. "
     "Uses a mixed team and serves as a gatekeeper before the Elite Four."),
    ("captain.png", "Captain",
     "The ship captain sprite, used for the captain of the S.S. Aqua ferry "
     "that travels between Olivine City (Johto) and Vermilion City (Kanto)."),
    ("chris.png", "Kris (female player character)",
     "The female player character Kris's overworld sprite — the first selectable "
     "female protagonist in a main series Pokemon game. Exclusive to Pokemon Crystal."),
    ("chris_bike.png", "Kris on bicycle",
     "Kris's overworld sprite while riding the Bicycle, obtained from the Bike Shop "
     "in Goldenrod City. Doubles movement speed on routes."),
    ("chuck.png", "Chuck",
     "Chuck, the Cianwood City Gym Leader specializing in Fighting-type Pokemon. "
     "Uses Primeape and Poliwrath. Gives the Storm Badge."),
    ("clair.png", "Clair",
     "Clair, the Blackthorn City Gym Leader who specializes in Dragon-type Pokemon. "
     "The 8th and final Johto Gym Leader. Cousin of Lance. Gives the Rising Badge."),
    ("clerk.png", "Store clerk",
     "Store clerk NPC sprite, found at Poke Marts and shops throughout Johto and Kanto."),
    ("cooltrainer_f.png", "Cool Trainer (female)",
     "Female Cool Trainer sprite — higher-level trainers found on later routes "
     "and Victory Road. Use a variety of evolved Pokemon."),
    ("cooltrainer_m.png", "Cool Trainer (male)",
     "Male Cool Trainer sprite — higher-level trainers found on later routes."),
    ("daisy.png", "Daisy Oak",
     "Daisy Oak, Professor Oak's granddaughter and Blue's sister. Gives the player "
     "the Expn Card in Pallet Town, unlocking Kanto Radio stations on the Pokegear."),
    ("dragon.png", "Dragon tamer trainer",
     "Overworld sprite for Dragon-type trainer class specialists found in Blackthorn "
     "City's Dragon Den and Victory Road."),
    ("elder.png", "Elder NPC",
     "Elder NPC sprite used for wise old men like the Tin Tower elders in Ecruteak "
     "City, who guard the legendary Ho-Oh."),
    ("elm.png", "Professor Elm",
     "Professor Elm, the Pokemon professor of New Bark Town who gives the player "
     "their starter Pokemon (Chikorita, Cyndaquil, or Totodile)."),
    ("erika.png", "Erika",
     "Erika, the Celadon City Gym Leader specializing in Grass-type Pokemon. "
     "Uses Tangela, Jumpluff, and Victreebel. Gives the Rainbow Badge."),
    ("fairy.png", "Lass trainer",
     "Lass-type trainer sprite used for young girl trainers on early routes."),
    ("falkner.png", "Falkner",
     "Falkner, the Violet City Gym Leader specializing in Flying-type Pokemon. "
     "The first Johto Gym Leader. Uses Pidgey and Pidgeot. Gives the Zephyr Badge."),
    ("famicom.png", "Famicom (NES) console",
     "The Famicom (NES) console sprite used as a decorative item that can be placed "
     "in the player's room."),
    ("fisher.png", "Fisher trainer",
     "Fisher trainer class sprite — fishermen found near water routes who use "
     "Water-type Pokemon caught with fishing rods."),
    ("fishing_guru.png", "Fishing Guru",
     "The Fishing Guru NPC who distributes fishing rods to the player: Old Rod "
     "(Route 32), Good Rod (Olivine City), Super Rod (Route 12 in Kanto)."),
    ("fruit_tree.png", "Berry tree / Fruit tree",
     "Overworld sprite for Berry trees that grow Berries. Berries can be harvested "
     "and given to Pokemon to hold, or used for various effects."),
    ("gameboy_kid.png", "Game Boy kid",
     "Child NPC sprite holding a Game Boy — used as a room decoration and generic "
     "child NPC in towns."),
    ("gentleman.png", "Gentleman trainer",
     "Gentleman trainer class sprite — older wealthy trainers found in cities who "
     "use well-bred Pokemon."),
    ("gold_trophy.png", "Gold trophy",
     "The Gold Trophy awarded for winning the Goldenrod City Bug Catching Contest "
     "with the highest-scoring Bug-type Pokemon."),
    ("gramps.png", "Grandpa / Old man NPC",
     "Elderly male NPC sprite used for old men in towns who give advice or items."),
    ("granny.png", "Grandma / Old woman NPC",
     "Elderly female NPC sprite used for old women in towns."),
    ("gym_guide.png", "Gym Guide",
     "The Gym Guide NPC posted at the entrance of each Pokemon Gym who explains "
     "the Gym Leader's specialty type and how many badges the player has."),
    ("janine.png", "Janine",
     "Janine, the Fuchsia City Gym Leader specializing in Poison-type Pokemon. "
     "Koga's daughter who took over after he joined the Elite Four."),
    ("jasmine.png", "Jasmine",
     "Jasmine, the Olivine City Gym Leader specializing in Steel-type Pokemon. "
     "The first Steel-type Gym Leader in the series. Her Steelix is her ace."),
    ("karen.png", "Karen",
     "Karen, the fourth Elite Four member specializing in Dark-type Pokemon. "
     "Famous quote: 'Strong Pokemon. Weak Pokemon. That is only the selfish perception.'"),
    ("kimono_girl.png", "Kimono Girl",
     "One of the five Kimono Girls in Ecruteak City who perform dances and challenge "
     "the player. Each uses a different Eeveelution: Vaporeon, Jolteon, Flareon, Espeon, Umbreon."),
    ("koga.png", "Koga",
     "Koga, the third Elite Four member. Former Fuchsia City Gym Leader specializing "
     "in Poison-type Pokemon. Uses Ariados, Forretress, Muk, Venomoth, and Crobat."),
    ("kurt.png", "Kurt",
     "Kurt, the Poke Ball craftsman in Azalea Town who makes special Poke Balls "
     "from Apricorns: Fast Ball, Friend Ball, Heavy Ball, Level Ball, Love Ball, Lure Ball, Moon Ball."),
    ("lance.png", "Lance",
     "Lance, the Pokemon Champion who rules from the League. Dragon-type master and "
     "co-protagonist during the Team Rocket Radio Tower arc. His ace is Dragonite."),
    ("lass.png", "Lass trainer",
     "Lass trainer class sprite — young girl trainers found on early routes using "
     "Clefairy, Snubbull, and similar Pokemon."),
    ("link_receptionist.png", "Link Receptionist",
     "Receptionist NPC for Game Link cable trading and battling. Found at Pokemon "
     "Centers and the Colosseum building in Goldenrod City."),
    ("misty.png", "Misty",
     "Misty, the Cerulean City Gym Leader specializing in Water-type Pokemon. "
     "Found at Cerulean Cape with her boyfriend until the player activates the Radio Tower event."),
    ("mom.png", "Player's Mom",
     "The player's mother in New Bark Town. Saves money on the player's behalf when "
     "enabled — purchases healing items and occasionally rare goods."),
    ("monster.png", "Monster / Creature sprite",
     "Generic monster or creature NPC sprite used in certain story events."),
    ("morty.png", "Morty",
     "Morty, the Ecruteak City Gym Leader specializing in Ghost-type Pokemon. "
     "Uses Gastly, Haunter, and Gengar. Gives the Fog Badge."),
    ("n64.png", "Nintendo 64 console",
     "The Nintendo 64 console sprite used as a decorative room item. "
     "One of several real consoles that can be displayed in the player's room."),
    ("nurse.png", "Nurse Joy",
     "Nurse Joy, the Pokemon Center nurse who fully heals the player's party when "
     "spoken to. Present in every Pokemon Center in Johto and Kanto."),
    ("oak.png", "Professor Oak",
     "Professor Oak, the original Pokemon professor from Pallet Town. In Gold/Silver "
     "he evaluates the player's Pokedex and provides the Pokerus explanation."),
    ("officer.png", "Officer Jenny",
     "Officer Jenny, the police officer NPC found in towns. Reports the rival's "
     "theft of a Pokemon from Elm's lab at the game's start."),
    ("paper.png", "Sign / Notice board",
     "Overworld sprite for paper signs and notice boards. Signs on routes describe "
     "what Pokemon can be found, warn of danger, or mark city borders."),
    ("pharmacist.png", "Pharmacist",
     "The pharmacist NPC in Cianwood City who creates a special medicine for Jasmine's "
     "sick Ampharos (Amphy) at the Olivine Lighthouse — a key story errand."),
    ("poke_ball.png", "Poke Ball item (overworld)",
     "The Poke Ball as it appears on the overworld map — a shiny ball indicating "
     "a hidden or visible item that can be picked up by pressing A."),
    ("pokedex.png", "Pokedex",
     "The Pokedex device sprite. Records data on every Pokemon seen and caught. "
     "Professor Elm gives it at the start. Completing it unlocks special rewards from Oak."),
    ("pokefan_f.png", "PokeFan (female)",
     "Female PokeFan trainer class sprite — obsessive Pokemon fans who carry photos "
     "of their Pokemon and call the player on the Pokegear repeatedly."),
    ("pokefan_m.png", "PokeFan (male)",
     "Male PokeFan trainer class sprite. PokeFans can be registered for rematches "
     "via the Pokegear phone system."),
    ("pryce.png", "Pryce",
     "Pryce, the Mahogany Town Gym Leader specializing in Ice-type Pokemon. "
     "The 7th Johto Gym Leader. Uses Seel, Dewgong, and Piloswine. Gives the Glacier Badge."),
    ("receptionist.png", "Receptionist",
     "Generic receptionist NPC sprite found at Pokemon Centers, the Radio Tower, "
     "and other official buildings."),
    ("red.png", "Red",
     "Red, the silent protagonist of Pokemon Red/Blue, who retreated to Mt. Silver "
     "after becoming Champion. The true final boss with the highest-level trainer team in the game."),
    ("reds_mom.png", "Red's Mom",
     "Red's mother NPC in Pallet Town. Mentions that Red left for Mt. Silver "
     "after becoming the Champion of the Indigo League."),
    ("rival.png", "Rival (Silver)",
     "The rival character Silver — son of Team Rocket boss Giovanni. Stole his "
     "starter Pokemon from Elm's lab. His arc involves learning to care for Pokemon."),
    ("rock.png", "Rock obstacle sprite",
     "Overworld rock sprite used as obstacles and scenery in caves and rocky routes. "
     "Distinct from the pushable boulder — cannot be moved."),
    ("rocker.png", "Rocker trainer",
     "Rocker trainer class sprite — electric guitar-themed trainers who use "
     "Electric-type or sound-themed Pokemon."),
    ("rocket.png", "Team Rocket Grunt (male)",
     "Male Team Rocket Grunt overworld sprite. Members of the villainous Team Rocket "
     "organization who operate in Mahogany Town and the Goldenrod Radio Tower."),
    ("rocket_girl.png", "Team Rocket Grunt (female)",
     "Female Team Rocket Grunt overworld sprite. Rocket Grunts use Rattata, Zubat, "
     "Koffing, and Houndour."),
    ("sabrina.png", "Sabrina",
     "Sabrina, the Saffron City Gym Leader specializing in Psychic-type Pokemon. "
     "Uses Espeon as her ace in the Gold/Silver rematch."),
    ("sage.png", "Sage trainer",
     "Sage trainer class sprite — monks found in Sprout Tower and other spiritual "
     "locations who use Bellsprout."),
    ("sailor.png", "Sailor trainer",
     "Sailor trainer class sprite — found on ships (S.S. Aqua) and port areas, "
     "using Water-type Pokemon."),
    ("scientist.png", "Scientist trainer",
     "Scientist trainer class sprite — found in labs and underground facilities, "
     "using Pokemon with special moves or evolutions."),
    ("silver_trophy.png", "Silver trophy",
     "The Silver Trophy awarded for second place in the National Park Bug Catching Contest."),
    ("slowpoke.png", "Slowpoke (overworld)",
     "Overworld Slowpoke sprite blocking Slowpoke Well in Azalea Town. Team Rocket "
     "is cutting off their tails to sell as delicacies — the player must stop them."),
    ("snes.png", "Super Nintendo (SNES) console",
     "The Super Nintendo console sprite used as a decorative room item."),
    ("sudowoodo.png", "Sudowoodo (overworld)",
     "Sudowoodo's overworld sprite blocking the road north of Goldenrod City, "
     "disguised as a tree. Water it with the SquirtBottle to trigger a battle with this unique Rock-type."),
    ("super_nerd.png", "Super Nerd trainer",
     "Super Nerd trainer class sprite — found in Radio Tower and research facilities, "
     "using Electric-type or gimmick Pokemon."),
    ("surf.png", "Surfing (HM03 Surf) sprite",
     "The overworld sprite of the player surfing on a Pokemon using HM03 Surf. "
     "Required to cross water routes, reach islands, and find hidden areas."),
    ("surfing_pikachu.png", "Surfing Pikachu",
     "Special overworld sprite for a Pikachu that knows Surf — a rare event Pokemon "
     "from the Pokemon Stadium 2 minigame, unobtainable through normal gameplay."),
    ("surge.png", "Lt. Surge",
     "Lt. Surge, the Vermilion City Gym Leader specializing in Electric-type Pokemon. "
     "Former military soldier. Uses Raichu in the post-game. Gives the Thunder Badge."),
    ("swimmer_girl.png", "Swimmer (female)",
     "Female Swimmer trainer class sprite — found in water routes and swimming pools, "
     "using Water-type Pokemon."),
    ("swimmer_guy.png", "Swimmer (male)",
     "Male Swimmer trainer class sprite — found surfing water routes between towns."),
    ("teacher.png", "Teacher NPC",
     "Teacher NPC sprite found in schools and Pokemon Academy-style buildings, "
     "giving tips about battle mechanics."),
    ("twin.png", "Twins trainer",
     "Twins trainer class sprite — paired girl trainers who challenge the player "
     "simultaneously in double-battle-style encounters."),
    ("whitney.png", "Whitney",
     "Whitney, the Goldenrod City Gym Leader specializing in Normal-type Pokemon. "
     "Famous for her Miltank — considered one of the hardest Gym Leaders by new players."),
    ("will.png", "Will",
     "Will, the first Elite Four member specializing in Psychic-type Pokemon. "
     "Uses Xatu, Exeggutor, Jynx, and Slowbro."),
    ("youngster.png", "Youngster trainer",
     "Youngster trainer class sprite — young boy trainers found on early routes "
     "using basic Pokemon like Rattata and Sandshrew."),
]

# Player character sprites
PLAYER = [
    ("chris_back.png", "Kris back sprite (battle)",
     "The back view of the female player character Kris during Pokemon battles. "
     "The first female protagonist in the main Pokemon series (Pokemon Crystal exclusive)."),
]


def download_file(url, dest_path, retries=2):
    """Download binary file to dest_path. Returns True on success."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                if resp.status == 200:
                    with open(dest_path, "wb") as f:
                        f.write(resp.read())
                    return True
        except Exception as e:
            if attempt < retries:
                print(f"    Retry {attempt}: {e}")
                time.sleep(1.0)
            else:
                print(f"    FAILED: {e}")
    return False


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_batch(batch, gfx_subpath, out_subdir, category, index_entries):
    """Download a list of (rel_path, name, description) tuples."""
    out_dir = os.path.join(OUT_DIR, out_subdir)
    ok = 0
    skip = 0
    fail = 0
    for rel_path, name, description in batch:
        filename = rel_path.replace("/", "_")  # flatten subdirs into filename
        png_dest = os.path.join(out_dir, filename)
        json_dest = png_dest.replace(".png", ".json")
        url = f"{BASE_URL}/{gfx_subpath}/{rel_path}"

        # Skip if already downloaded
        if os.path.exists(png_dest):
            skip += 1
        else:
            success = download_file(url, png_dest)
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  [FAIL] {name}")
                continue
            time.sleep(0.2)

        # Always write/overwrite metadata JSON
        meta = {
            "name": name,
            "category": category,
            "filename": filename,
            "source_path": f"gfx/{gfx_subpath}/{rel_path}",
            "source_repo": "pret/pokegold",
            "source_url": url,
            "description": description,
            "game": "Pokemon Gold / Silver",
            "platform": "Game Boy Color",
            "release_year": 2000,
        }
        write_json(json_dest, meta)

        index_entries.append({
            "name": name,
            "category": category,
            "filename": filename,
            "folder": out_subdir,
            "description": description[:80] + ("..." if len(description) > 80 else ""),
        })

        label = f"  [{category[:8]:8}] {name[:40]:40} {'OK' if ok or skip else 'FAIL'}"
        print(label)

    return ok, skip, fail


def main():
    print("Pokemon Gold/Silver In-Game Sprite Downloader")
    print("Source: github.com/pret/pokegold")
    print(f"Output: {OUT_DIR}")
    print("-" * 60)

    index_entries = []
    total_ok = total_skip = total_fail = 0

    sections = [
        (TILESETS,      "tilesets",         "tilesets",    "Tileset"),
        (TILESET_ANIM,  "tilesets",         "tilesets",    "Tileset Anim"),
        (OVERWORLD,     "overworld",        "overworld",   "Overworld"),
        (SPRITES,       "sprites",          "npc-sprites", "NPC Sprite"),
        (PLAYER,        "player",           "player",      "Player"),
    ]

    for batch, gfx_sub, out_sub, category in sections:
        print(f"\n=== {category} ({len(batch)} files) ===")
        ok, skip, fail = process_batch(batch, gfx_sub, out_sub, category, index_entries)
        total_ok += ok
        total_skip += skip
        total_fail += fail
        print(f"  -> {ok} downloaded, {skip} skipped (already exist), {fail} failed")

    # Master index
    index_path = os.path.join(OUT_DIR, "game-sprites-index.json")
    write_json(index_path, {
        "total": len(index_entries),
        "source": "pret/pokegold (https://github.com/pret/pokegold)",
        "game": "Pokemon Gold / Silver",
        "entries": index_entries,
    })

    print("\n" + "-" * 60)
    print(f"Done! {total_ok} downloaded, {total_skip} skipped, {total_fail} failed.")
    print(f"Master index: game-sprites-index.json ({len(index_entries)} entries)")


if __name__ == "__main__":
    main()
