# Game Agent — TODOs

## Backlog

| Status | Item |
|--------|------|
| Todo | **Test `--script` flag on mGBA 0.11.0** — when 0.11.0 releases, verify `mGBA.exe --script mgba_launcher.lua` works end-to-end on Windows Qt. If confirmed, the manual scripting window step can be removed from the startup flow entirely. |
| Todo | **Session auto-resume for agent.py** — on startup, scan `~/.mgba-live-mcp/runtime/` for the newest session dir and offer to re-attach if `heartbeat.json` is recent (e.g. < 5 min old). Currently requires manually passing `--session <id>`. |
| Todo | **Add `games/pokemon-firered.json`** — gym order, story path, and RAM offsets differ from Sapphire. |
| Todo | **HP turns valid after state load** — RAM HP addresses return 0/0 for a few frames after `pyboy.load_state()`. Detect when HP becomes valid (e.g. `hp_max > 0` for 2+ consecutive turns) before trusting it in the prompt. |
| Todo | **operator override UI** — instead of writing to `agent_message.txt` manually, add a simple `--message "..."` CLI subcommand or a small stdin reader thread so interventions can be typed directly into the agent terminal. |
| In Progress | **Verify pyboy_agent.py wall detection fix** — RAM position delta now used as primary check; needs a full run in a real indoor room to confirm no false walls. |
| Completed | Refactor agent.py from Pokemon-specific to generic `--game` flag |
| Completed | Create `games/pokemon-sapphire.json` reference profile |
| Completed | Confirm LM Studio + `google/gemma-4-e4b` working at port 1234 |
| Completed | Confirm `--script` CLI flag non-functional on Windows mGBA Qt 0.10.5 |
| Completed | Generate `mgba_launcher.lua` with hardcoded session dir + `os.getenv` patch |
| Completed | Confirm bridge works end-to-end — heartbeat.json written at frame 37470 in session 20260417-102408 |
| Completed | Add clear 4-step startup instructions to agent output (ROM first → Scripting → Load script → Run) |
| Completed | Document that the 1-second freeze on script load is expected (`os.execute mkdir`) |
| Completed | Explain `--session <id>` for resuming after agent crash without reloading mGBA |
| Completed | Create `.github/copilot-instructions.md` and path-specific instructions files |
| Completed | Update all docs and agent.py comments to reflect confirmed working state |
| Completed | Fix `has_ram` always-False bug (`"note"` key exclusion) |
| Completed | Fix `perceive()` JSON truncation — raise `max_tokens` 1024 → 2048 |
| Completed | Fix false LOW HP warning when `hp_max == 0` after state restore |
| Completed | Normalise `passable_directions` and `player_facing` to title case |
| Completed | Fix wall detection — use RAM position delta as primary; hash fallback only when no RAM |
| Completed | Scope wall keys to `map_{bank}_{number}` instead of fuzzy VLM location names |
| Completed | Move RAM read to top of turn loop (single read per turn, shared across nav/decide/wall) |
| Completed | Add operator override drop file (`agent_message.txt`) for live interventions |
| Completed | Add `--headless` flag for windowless max-speed testing |
| Completed | Make `--rom` optional via `ROM_PATH` in `.env` |
| Completed | Update `.github/copilot-instructions.md` to cover both agents |
| Completed | Update `agent-loop.instructions.md` for `pyboy_agent.py` architecture |
| Completed | Update `game-profiles.instructions.md` for GBC profiles and RAM offset format |
