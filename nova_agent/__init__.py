"""
nova_agent — single-model tool-calling GBC game agent.

Key architectural differences from pyboy_agent:

1. **One LLM call per turn** (not two).  The vision-capable model receives
   the screenshot directly and decides via structured tool calls.

2. **Screenshot overlay** — RAM facts (position, HP, badges) are drawn onto
   the screenshot image, not sent as a separate text block.

3. **Tool-based interaction** — the model calls press_buttons, navigate_to,
   update_knowledge, add_event, and set_goal.  No rigid JSON return schema.

4. **Tile-graph pathfinder** — navigate_to(x, y) runs BFS over a learned
   walkability graph and executes the path automatically.

5. **Structured knowledge base** — sectioned JSON memory (current_status,
   game_progress, objectives, party_status, notes) instead of a one-liner.

6. **Screen-type state machine** — OVERWORLD/BATTLE/DIALOGUE/MENU with
   focused hints per state.

7. **Rolling summarization** — history is condensed when it grows long, then
   the KB is reviewed for stale data.

Run::

    python -m nova_agent --rom "H:/Games/GBC/Pokemon Silver.gbc"
    python -m nova_agent --help
"""

__version__ = "0.1.0"
