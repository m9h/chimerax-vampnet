"""HTTP/JSON bridge so MCP-capable LLM agents can drive the bundle.

Started from inside ChimeraX:
  vampnet mcp serve [port 7345]

The bridge runs a stdlib `http.server` in a daemon thread and exposes
each bundle command as a POST endpoint that returns the same
JSON-serializable dict the command returns to the ChimeraX log. The
companion script `scripts/mcp_stdio_proxy.py` (shipped separately)
adapts this to the MCP stdio transport for Claude Desktop / Cursor.

Endpoints:
  GET  /tools                       -> {tools: [...]}
  GET  /health                      -> {status: "ok", model_loaded: bool}
  POST /tools/<name>  (JSON body)   -> {ok: True, result: <dict>}
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict


# Tool manifest. Each entry is a ChimeraX command-line template (a
# Callable that takes the request body and returns a CLI string).
TOOL_MANIFEST: list[dict] = [
    {
        "name": "vampnet_load_ensemble",
        "description": "Load a conformational ensemble from disk (MD/.dcd, AlphaFlow/.npz, BioEmu/.npz or dir of .pdbs) into the active ChimeraX session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "ChimeraX model spec, e.g. '#1'"},
                "path": {"type": "string"},
                "source": {"type": "string", "enum": ["md", "alphaflow", "bioemu", "auto"]},
            },
            "required": ["structure_id", "path"],
        },
    },
    {
        "name": "vampnet_fit",
        "description": "Train a VAMPnet on the loaded ensembles and report VAMP-2 score + implied timescales + state populations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_states": {"type": "integer", "default": 4},
                "lag": {"type": "integer", "default": 10},
                "features": {"type": "string", "enum": ["ca_distances", "torsions", "contacts"], "default": "ca_distances"},
                "epochs": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "vampnet_timescales",
        "description": "Implied timescales test across a range of lag times.",
        "input_schema": {"type": "object", "properties": {"taus": {"type": "array", "items": {"type": "integer"}}}, "required": ["taus"]},
    },
    {"name": "vampnet_states", "description": "Color current frame by VAMPnet state and register a coordset-change handler so the structure recolors as the user scrubs.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "vampnet_means", "description": "Build per-state mean-structure models in the session.", "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "vampnet_animate",
        "description": "Synthetic trajectory interpolating along a slow VAMPnet mode.",
        "input_schema": {
            "type": "object",
            "properties": {"mode": {"type": "integer", "default": 1}, "n_frames": {"type": "integer", "default": 100}},
        },
    },
    {"name": "vampnet_network", "description": "MSM transition matrix as a structured graph.", "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "vampnet_save",
        "description": "Save the fitted model to disk (pickle).",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "vampnet_load",
        "description": "Load a previously saved model.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
]


_COMMAND_TEMPLATES: Dict[str, Callable[[dict], str]] = {
    "vampnet_load_ensemble": lambda a: f"vampnet load_ensemble {a['structure_id']} {a['path']} source={a.get('source', 'auto')}",
    "vampnet_fit": lambda a: f"vampnet fit nStates={a.get('n_states', 4)} lag={a.get('lag', 10)} features={a.get('features', 'ca_distances')} epochs={a.get('epochs', 200)}",
    "vampnet_timescales": lambda a: f"vampnet timescales taus={','.join(str(t) for t in a['taus'])}",
    "vampnet_states": lambda a: "vampnet states",
    "vampnet_means": lambda a: "vampnet means",
    "vampnet_animate": lambda a: f"vampnet animate mode={a.get('mode', 1)} nFrames={a.get('n_frames', 100)}",
    "vampnet_network": lambda a: "vampnet network",
    "vampnet_save": lambda a: f"vampnet save {a['path']}",
    "vampnet_load": lambda a: f"vampnet load {a['path']}",
}


_BRIDGE_STATE = {"server": None, "thread": None, "session": None, "port": None}


def _dispatch(session, tool_name: str, args: dict) -> Any:
    """Translate an MCP tool call into a ChimeraX command and run it."""
    if tool_name not in _COMMAND_TEMPLATES:
        return {"error": f"unknown tool: {tool_name}"}
    cmd_str = _COMMAND_TEMPLATES[tool_name](args)
    try:
        from chimerax.core.commands import run
        result = run(session, cmd_str)
        # `run` returns a list of return values from the command; collapse
        # if singleton.
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        return {"result": result, "command": cmd_str}
    except Exception as e:
        return {"error": str(e), "command": cmd_str}


def _make_handler(session):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Silence the default access log; ChimeraX has its own logger.
            return

        def _json(self, status: int, body: dict):
            data = json.dumps(body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                model = getattr(session, "_vampnet_model", None)
                return self._json(200, {"status": "ok", "model_loaded": model is not None})
            if self.path == "/tools":
                return self._json(200, {"tools": TOOL_MANIFEST})
            return self._json(404, {"error": "not found", "path": self.path})

        def do_POST(self):
            if not self.path.startswith("/tools/"):
                return self._json(404, {"error": "use /tools/<name>"})
            tool_name = self.path[len("/tools/"):]
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                args = json.loads(body) if body else {}
            except Exception as e:
                return self._json(400, {"error": f"bad json: {e}"})
            result = _dispatch(session, tool_name, args)
            return self._json(200, {"ok": "error" not in result, **result})

    return _Handler


def start(session, port: int = 7345) -> dict:
    """Spin up the HTTP bridge in a daemon thread. Idempotent."""
    if _BRIDGE_STATE["server"] is not None:
        return {"status": "already_running", "port": _BRIDGE_STATE["port"]}

    handler_cls = _make_handler(session)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, name="vampnet-mcp-bridge", daemon=True)
    thread.start()

    _BRIDGE_STATE["server"] = httpd
    _BRIDGE_STATE["thread"] = thread
    _BRIDGE_STATE["session"] = session
    _BRIDGE_STATE["port"] = port

    msg = f"vampnet HTTP bridge listening on http://127.0.0.1:{port}"
    try:
        session.logger.info(msg)
    except Exception:
        print(msg)

    return {
        "status": "started",
        "port": port,
        "endpoint": f"http://127.0.0.1:{port}",
        "tools": [t["name"] for t in TOOL_MANIFEST],
    }


def stop() -> dict:
    """Stop the bridge if running."""
    httpd = _BRIDGE_STATE["server"]
    if httpd is None:
        return {"status": "not_running"}
    httpd.shutdown()
    httpd.server_close()
    _BRIDGE_STATE["server"] = None
    _BRIDGE_STATE["thread"] = None
    _BRIDGE_STATE["port"] = None
    return {"status": "stopped"}
