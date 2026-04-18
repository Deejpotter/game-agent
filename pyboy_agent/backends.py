"""
pyboy_agent.backends
====================
OpenAI-compatible client factory functions.

Handles the GitHub Copilot (OpenClaw) special case: the token lives in a
JSON file that OpenClaw refreshes automatically, so we re-read it on every
401 Unauthorized response instead of requiring a manual restart.
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

# Path where OpenClaw writes the refreshed Copilot session token.
COPILOT_TOKEN_PATH = Path.home() / ".openclaw" / "credentials" / "github-copilot.token.json"

# Required headers for the GitHub Copilot API.
# These identify the client as a VS Code extension to avoid rate-limit buckets
# reserved for browser sessions.
COPILOT_HEADERS: dict[str, str] = {
    "Editor-Version": "vscode/1.95.0",
    "Editor-Plugin-Version": "copilot-chat/0.22.4",
    "Copilot-Integration-Id": "vscode-chat",
    "Openai-Intent": "conversation-panel",
    "User-Agent": "GitHubCopilotGame/1.0",
}


def load_copilot_token() -> str:
    """Read the current Copilot session token from the OpenClaw credentials file.

    OpenClaw keeps this file up to date, so re-reading on a 401 response gives
    a fresh token without any user interaction.

    Raises:
        FileNotFoundError: if OpenClaw is not running or the user is not signed in.
    """
    if not COPILOT_TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"GitHub Copilot token not found at {COPILOT_TOKEN_PATH}. "
            "Ensure OpenClaw is running and you are signed in to GitHub Copilot."
        )
    import json
    data = json.loads(COPILOT_TOKEN_PATH.read_text(encoding="utf-8"))
    return data["token"]


def make_copilot_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at the GitHub Copilot API."""
    return OpenAI(
        base_url="https://api.githubcopilot.com",
        api_key=load_copilot_token(),
        default_headers=COPILOT_HEADERS,
    )


def make_client(backend_cfg: dict) -> OpenAI:
    """Create an OpenAI client from a BACKENDS entry.

    Automatically handles the Copilot sentinel api_key value.
    For Copilot backends, call load_copilot_token() first and set api_key
    in the config dict before calling this.
    """
    return OpenAI(
        base_url=backend_cfg["base_url"],
        api_key=backend_cfg["api_key"],
    )


def is_copilot_backend(cfg: dict) -> bool:
    """Return True if this backend config points at the Copilot API."""
    return cfg.get("base_url", "") == "https://api.githubcopilot.com"


def is_local_backend(cfg: dict) -> bool:
    """Return True if this backend is running on localhost (LM Studio / Ollama).

    Local backends support enable_thinking=True in extra_body.
    """
    return cfg.get("base_url", "").startswith("http://localhost")
