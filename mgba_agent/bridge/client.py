"""
mgba_agent/bridge/client.py — Direct file-IPC client for mgba_live_bridge.lua.

BridgeClient talks to the Lua bridge running inside mGBA by writing command.lua
and reading response.json, bypassing the mgba-live-mcp CLI entirely.

_to_lua_value() serialises Python values into Lua literals for the command file.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import uuid
from pathlib import Path
from typing import Any

from ..config import SETTLE_FRAMES


def _to_lua_value(value: Any) -> str:
    """Serialize a Python value to a Lua literal (mirrors live_cli.py's to_lua_value)."""
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(_to_lua_value(v) for v in value) + "}"
    if isinstance(value, dict):
        parts: list[str] = []
        for k in sorted(value.keys(), key=str):
            ks = str(k)
            if ks.isidentifier():
                parts.append(f"{ks} = {_to_lua_value(value[k])}")
            else:
                parts.append(f'["{ks}"] = {_to_lua_value(value[k])}')
        return "{ " + ", ".join(parts) + " }"
    raise TypeError(f"Unsupported value type: {type(value)}")


class BridgeClient:
    """Talks directly to mgba_live_bridge.lua via file IPC.

    Bypasses the mgba-live-mcp CLI entirely so there is no session.json,
    no PID tracking, and no prune_dead_sessions() interference.
    """

    def __init__(self, ipc_dir: Path) -> None:
        self.ipc_dir = ipc_dir
        self.command_path = ipc_dir / "command.lua"
        self.response_path = ipc_dir / "response.json"
        self.heartbeat_path = ipc_dir / "heartbeat.json"
        shots = ipc_dir / "screenshots"
        shots.mkdir(exist_ok=True)
        self._shots_dir = shots

    async def send(
        self, kind: str, payload: dict | None = None, timeout: float = 15.0
    ) -> dict:
        """Send one command to the bridge and return its response dict."""
        payload = payload or {}
        req_id = uuid.uuid4().hex
        command = {"id": req_id, "kind": kind, **payload}

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout

        # Wait for bridge to consume any previous command
        while self.command_path.exists():
            if loop.time() > deadline:
                raise TimeoutError("Bridge busy — command.lua not consumed in time")
            await asyncio.sleep(0.02)

        # Clear stale response — retry on Windows file-locking errors
        for _ in range(10):
            try:
                self.response_path.unlink(missing_ok=True)
                break
            except PermissionError:
                await asyncio.sleep(0.05)

        # Write command atomically
        tmp = self.command_path.with_suffix(".tmp")
        tmp.write_text("return " + _to_lua_value(command) + "\n", encoding="utf-8")
        tmp.replace(self.command_path)

        # Wait for matching response
        while loop.time() < deadline:
            if self.response_path.exists():
                try:
                    resp = json.loads(self.response_path.read_text(encoding="utf-8"))
                    if resp.get("id") == req_id:
                        return resp
                except json.JSONDecodeError:
                    pass
            await asyncio.sleep(0.02)

        # Clean up command if bridge never consumed it
        if self.command_path.exists():
            self.command_path.unlink(missing_ok=True)
        raise TimeoutError(f"Bridge timeout waiting for response to '{kind}'")

    async def screenshot(self) -> str:
        """Capture a screenshot and return it as base64-encoded PNG."""
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shot_path = self._shots_dir / f"shot-{ts}.png"
        resp = await self.send("screenshot", {"path": shot_path.as_posix()})
        if not resp.get("ok"):
            raise RuntimeError(f"Bridge screenshot error: {resp.get('error', resp)}")
        raw = shot_path.read_bytes()
        return base64.b64encode(raw).decode()

    async def tap_and_screenshot(
        self, key: str, duration: int = 2, wait_frames: int = SETTLE_FRAMES
    ) -> str | None:
        """Press a button and return a screenshot taken after wait_frames settle."""
        tap_resp = await self.send("tap_key", {"key": key, "duration": duration})
        if not tap_resp.get("ok"):
            raise RuntimeError(f"Bridge tap error: {tap_resp.get('error', tap_resp)}")

        # Wait for the tap + settle frames to pass, using the heartbeat frame counter.
        tap_frame = tap_resp.get("frame", 0)
        target_frame = tap_frame + duration + wait_frames
        for _ in range(300):  # up to 6 s at 20 ms polls
            try:
                hb = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
                if hb.get("frame", 0) >= target_frame:
                    break
            except (json.JSONDecodeError, OSError):
                pass
            await asyncio.sleep(0.02)

        try:
            return await self.screenshot()
        except Exception:
            return None  # screenshot is best-effort; caller can retry

    async def read_range(self, start: int, length: int) -> list[int]:
        """Read `length` bytes starting at `start` address. Returns list of byte values."""
        if length <= 0:
            return []
        resp = await self.send("read_range", {"start": start, "length": length})
        if resp.get("error"):
            raise RuntimeError(f"Bridge read_range error: {resp['error']}")
        inner = resp.get("data", {})
        # Bridge wraps the Lua read_range result {start,length,data} as the "data" field
        if isinstance(inner, dict):
            data = inner.get("data", [])
        else:
            data = inner  # fallback: bare list (defensive)
        if not isinstance(data, list):
            raise TypeError(f"read_range expected list, got {type(data).__name__}: {inner!r:.120}")
        return data

    async def read_u8(self, addr: int) -> int:
        """Read a single unsigned byte from `addr`."""
        data = await self.read_range(addr, 1)
        return data[0] if data else 0

    async def read_u16(self, addr: int) -> int:
        """Read a little-endian unsigned 16-bit value from `addr`."""
        data = await self.read_range(addr, 2)
        return (data[1] << 8 | data[0]) if len(data) >= 2 else 0

    async def read_u32(self, addr: int) -> int:
        """Read a little-endian unsigned 32-bit value from `addr`."""
        data = await self.read_range(addr, 4)
        if len(data) >= 4:
            return data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)
        return 0
