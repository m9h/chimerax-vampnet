#!/usr/bin/env python3
"""MCP stdio proxy for chimerax-vampnet.

Bridges Claude Desktop / Cursor / Continue (which speak MCP over stdio)
to the bundle's HTTP/JSON server (running inside ChimeraX). The proxy
is a thin JSON-RPC 2.0 -> HTTP translator. No deps beyond stdlib.

Usage from Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):

  "mcpServers": {
    "chimerax-vampnet": {
      "command": "python3",
      "args": ["/path/to/chimerax-vampnet/scripts/mcp_stdio_proxy.py"],
      "env": {"VAMPNET_BRIDGE_URL": "http://127.0.0.1:7345"}
    }
  }

The user opens ChimeraX, runs `vampnet mcp serve port 7345`, then
launches Claude Desktop. The MCP client spawns this proxy; the proxy
forwards tool calls into the running ChimeraX session.

Protocol implemented:
  initialize           -> ack with server capabilities
  tools/list           -> proxy GET /tools, reformat as MCP Tool list
  tools/call           -> proxy POST /tools/<name>, reformat as MCP content
  notifications/*      -> ignored (we don't push)
  any other method     -> JSON-RPC method-not-found error
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

BRIDGE_URL = os.environ.get("VAMPNET_BRIDGE_URL", "http://127.0.0.1:7345")
PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "chimerax-vampnet"
SERVER_VERSION = "0.1.0"


def _read_message() -> dict | None:
    """Read one JSON-RPC message from stdin.

    Supports both line-delimited and Content-Length-prefixed framing
    (Claude Desktop uses Content-Length).
    """
    first = sys.stdin.buffer.readline()
    if not first:
        return None

    if first.startswith(b"Content-Length:"):
        length = int(first.split(b":", 1)[1].strip())
        # Consume blank line.
        while True:
            line = sys.stdin.buffer.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        body = sys.stdin.buffer.read(length)
        return json.loads(body.decode("utf-8"))
    # Line-delimited fallback.
    try:
        return json.loads(first.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _write_message(obj: dict) -> None:
    body = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _http_get(path: str) -> dict:
    req = urllib.request.Request(f"{BRIDGE_URL}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BRIDGE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ok_response(req_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err_response(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _handle_initialize(req_id: Any, params: dict) -> dict:
    return _ok_response(req_id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    })


def _handle_tools_list(req_id: Any, params: dict) -> dict:
    try:
        manifest = _http_get("/tools")
    except urllib.error.URLError as e:
        return _err_response(req_id, -32000,
                             f"ChimeraX-VAMPnet bridge not reachable at {BRIDGE_URL}",
                             {"detail": str(e), "hint": "Run `vampnet mcp serve` inside ChimeraX."})
    return _ok_response(req_id, {
        "tools": [{
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t.get("input_schema", {"type": "object", "properties": {}}),
        } for t in manifest.get("tools", [])],
    })


def _handle_tools_call(req_id: Any, params: dict) -> dict:
    name = params.get("name")
    args = params.get("arguments") or {}
    try:
        body = _http_post(f"/tools/{name}", args)
    except urllib.error.HTTPError as e:
        return _err_response(req_id, -32602, f"tool call failed: {e.code}", {"body": e.read().decode("utf-8", errors="replace")})
    except urllib.error.URLError as e:
        return _err_response(req_id, -32000, f"bridge unreachable: {e}", None)

    if body.get("ok") is False or "error" in body:
        return _err_response(req_id, -32603, body.get("error", "tool failed"), body)

    payload = json.dumps(body.get("result", body), default=str, indent=2)
    return _ok_response(req_id, {
        "content": [{"type": "text", "text": payload}],
        "isError": False,
    })


_HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
}


def main():
    while True:
        try:
            msg = _read_message()
        except KeyboardInterrupt:
            return
        if msg is None:
            return

        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method is None:
            continue
        if method.startswith("notifications/"):
            # No id -> notification, no response.
            continue

        handler = _HANDLERS.get(method)
        if handler is None:
            if req_id is not None:
                _write_message(_err_response(req_id, -32601, f"method not found: {method}"))
            continue

        try:
            response = handler(req_id, params)
        except Exception as e:
            response = _err_response(req_id, -32603, f"handler error: {e}")
        _write_message(response)


if __name__ == "__main__":
    main()
