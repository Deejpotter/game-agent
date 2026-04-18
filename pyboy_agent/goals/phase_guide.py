"""
pyboy_agent.goals.phase_guide
==============================
Story phase guide for Pokemon Gold/Silver.

``BADGE_PHASE_MAP`` maps the current Johto badge count (0-7) to a concise
next-step hint injected into the reasoning model's prompt each turn.

Each entry is intentionally verbose — the reasoning model uses these hints as
its primary compass when the player is between objectives.  The hints list:
  - Which routes/towns to traverse
  - Which key items or NPCs to interact with
  - Which HMs are needed and where to get them
  - Which gym leader to challenge and their weakness

HM routing notes (Gold/Silver US)
-----------------------------------
- HM01 Cut     : given by the Charcoal Man in Ilex Forest (need to catch Farfetch'd)
- HM02 Fly     : given by Chuck's wife outside Cianwood Gym
- HM03 Surf    : received from the Kimono Girls in Ecruteak Dance Theater (after Burned Tower)
- HM04 Strength: given by a sailor in Olivine Café
- HM05 Flash   : top floor of Sprout Tower, Violet City
- HM06 Whirlpool: received from Lance after clearing Team Rocket's Mahogany Hideout
- HM07 Waterfall: picked up from the floor of Ice Path
"""

from __future__ import annotations

# Keyed by badge count (0 = no badges, 7 = seven badges, needs 8th).
BADGE_PHASE_MAP: dict[int, str] = {
    0: (
        "STORY[0/8 badges]: Get starter from Prof. Elm → "
        "visit Mr. Pokemon (Route 30, cottage north of Cherrygrove) → "
        "return to Elm → beat Rival Silver → go north via Routes 36/31 to "
        "Violet City → climb Sprout Tower for Flash(HM05) → "
        "beat Falkner(Flying gym, use Rock/Electric moves)."
    ),
    1: (
        "STORY[1/8 badges]: Head south: Route 32 → Union Cave → Route 33 → "
        "Azalea Town. Help Kurt at Slowpoke Well (go south of Azalea). "
        "In Ilex Forest: return Charcoal Man's Farfetch'd to get HM01 Cut. "
        "Beat Bugsy(Bug gym, use Fire/Flying/Rock moves)."
    ),
    2: (
        "STORY[2/8 badges]: Use Cut on tree in Ilex Forest, head north. "
        "Route 34 → Goldenrod City. Beat Whitney(Normal gym). "
        "Beware Miltank with Rollout — use Fighting-type moves or paralyse/poison it. "
        "Also explore National Park on Route 36."
    ),
    3: (
        "STORY[3/8 badges]: Head east to Ecruteak City (Routes 35-37). "
        "Visit Burned Tower NE of Ecruteak — legendary dogs escape (good). "
        "Beat all 5 Kimono Girls in the Dance Theater to receive HM03 Surf. "
        "Beat Morty(Ghost gym, use Normal/Dark moves — Ghost moves miss Ghost)."
    ),
    4: (
        "STORY[4/8 badges]: Go west via Routes 38-39 to Olivine City. "
        "Glitter Lighthouse: Jasmine's Ampharos is sick — need SecretPotion. "
        "Surf west to Cianwood City. Get SecretPotion from pharmacist. "
        "Beat Chuck(Fighting gym). Get HM02 Fly from Chuck's wife (outside gym). "
        "Return to Olivine, give SecretPotion to Jasmine at top of lighthouse."
    ),
    5: (
        "STORY[5/8 badges]: Beat Jasmine(Steel gym, use Fire/Ground/Fighting). "
        "Travel east to Mahogany Town (Route 42). "
        "North to Lake of Rage: catch/beat Red Gyarados with Lance. "
        "Help Lance clear Team Rocket Hideout in Mahogany (basement). "
        "Receive HM06 Whirlpool from Lance. Beat Pryce(Ice gym, use Fire/Rock/Steel)."
    ),
    6: (
        "STORY[6/8 badges]: Fly to Goldenrod — Team Rocket has seized the Radio Tower. "
        "Go to Goldenrod Underground, fight Rockets. Rescue Director from Warehouse "
        "(get Basement Key → Card Key). Beat Executive Ariana. Save Director at Radio Tower. "
        "Team Rocket disbands. Then: Route 44 → Ice Path(use Whirlpool) → Blackthorn City."
    ),
    7: (
        "STORY[7/8 badges]: Navigate Ice Path — use Strength to push boulders, "
        "pick up HM07 Waterfall from the floor. "
        "Beat Clair(Dragon gym, use Ice moves — Kingdra has no 4× weakness). "
        "Complete Dragon's Den quiz for Rising Badge. "
        "Fly to New Bark Town: receive Master Ball from Prof. Elm. "
        "Surf east → Route 27 → Mt. Silver gate → Victory Road → Indigo Plateau."
    ),
}
