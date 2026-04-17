# Game Agent — TODOs

## Backlog

| Status | Item |
|--------|------|
| Todo | **Test `--script` flag on mGBA 0.11.0** — when 0.11.0 releases, verify `mGBA.exe --script mgba_launcher.lua` works end-to-end on Windows Qt. If confirmed, the manual scripting window step can be removed from the startup flow entirely. |
| Todo | **Add a `--headless` or SDL-only flag** — investigate CLI-only mGBA builds where `--script` might work, enabling fully unattended startup. |
| Todo | **Session auto-resume** — on startup, scan `~/.mgba-live-mcp/runtime/` for the newest session dir and offer to re-attach if `heartbeat.json` is recent (e.g. < 5 min old). Currently requires manually passing `--session <id>`. |
| Todo | **RAM state logging** — read `ram_offsets` from the game profile each turn and log HP, map ID, badges etc. to a per-session log file for debugging and progress tracking. |
| Todo | **Add `games/pokemon-firered.json`** — gym order, story path, and RAM offsets differ from Sapphire. |
| Todo | **Screenshot diff stall detection** — compare consecutive screenshots pixel-by-pixel; if identical for N turns, force a B press then a directional to break the loop. |
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
