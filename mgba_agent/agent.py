"""
mgba_agent/agent.py — Backward-compatible entry point.

The agent has been refactored into a package. This file is kept so that
`python mgba_agent/agent.py` continues to work exactly as before.

For new invocations prefer:
    python -m mgba_agent --rom <path> [options]

See main.py for the full CLI and loop.py for the turn loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so that 'mgba_agent' is importable
# whether this file is run as `python mgba_agent/agent.py` or directly.
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from mgba_agent.main import main  # noqa: E402

if __name__ == "__main__":
    main()
