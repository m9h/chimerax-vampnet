"""End-to-end test of the MCP stdio proxy.

Spins up a stdlib HTTP mock that emulates the bundle's HTTP bridge
(http://127.0.0.1:<port>), launches the proxy as a subprocess speaking
MCP stdio, and verifies:
  - initialize  -> capability ack
  - tools/list  -> proxied tool manifest, name + description preserved
  - tools/call  -> POST proxied through, result returned as text content
  - unknown method -> JSON-RPC method-not-found
"""

import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROXY = Path(__file__).resolve().parent.parent / "scripts" / "mcp_stdio_proxy.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _MockBridge:
    def __init__(self):
        self.port = _free_port()
        self.received = []
        self.tool_responses: dict = {}

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a, **kw): pass

            def _write(self, status, body):
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/tools":
                    self._write(200, {"tools": [
                        {"name": "vampnet_fit", "description": "fit a VAMPnet",
                         "input_schema": {"type": "object", "properties": {"lag": {"type": "integer"}}}},
                        {"name": "vampnet_states", "description": "color by state",
                         "input_schema": {"type": "object", "properties": {}}},
                    ]})
                else:
                    self._write(404, {"error": "not found"})

            def do_POST(self):
                tool = self.path.split("/")[-1]
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                outer.received.append((tool, body))
                resp = outer.tool_responses.get(tool, {"result": {"echo": body}})
                self._write(200, {"ok": True, **resp})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class _ProxyClient:
    """Speaks MCP over stdio to the proxy subprocess."""

    def __init__(self, env_extra: dict):
        env = os.environ.copy()
        env.update(env_extra)
        self.proc = subprocess.Popen(
            [sys.executable, str(PROXY)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, bufsize=0,
        )
        self.next_id = 0

    def send(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        body = json.dumps(msg).encode("utf-8")
        framed = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body
        self.proc.stdin.write(framed)
        self.proc.stdin.flush()
        return self._read()

    def _read(self) -> dict:
        header = self.proc.stdout.readline()
        if not header.startswith(b"Content-Length:"):
            raise RuntimeError(f"unexpected response prefix: {header!r}")
        length = int(header.split(b":", 1)[1].strip())
        # Eat blank separator line(s).
        while True:
            line = self.proc.stdout.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        body = self.proc.stdout.read(length)
        return json.loads(body.decode("utf-8"))

    def stop(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=5)


def test_initialize_returns_capabilities():
    bridge = _MockBridge()
    client = _ProxyClient({"VAMPNET_BRIDGE_URL": bridge.url()})
    try:
        resp = client.send("initialize", {})
        assert resp["id"] == 1
        assert "result" in resp
        info = resp["result"]
        assert info["serverInfo"]["name"] == "chimerax-vampnet"
        assert "tools" in info["capabilities"]
    finally:
        client.stop()
        bridge.stop()


def test_tools_list_proxies_manifest():
    bridge = _MockBridge()
    client = _ProxyClient({"VAMPNET_BRIDGE_URL": bridge.url()})
    try:
        resp = client.send("tools/list", {})
        tools = resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert "vampnet_fit" in names
        assert "vampnet_states" in names
        assert tools[0]["inputSchema"]["type"] == "object"
    finally:
        client.stop()
        bridge.stop()


def test_tools_call_round_trips_arguments_and_result():
    bridge = _MockBridge()
    bridge.tool_responses["vampnet_fit"] = {
        "result": {"vamp2_score": 1.84, "implied_timescales": [166.4], "state_populations": [0.756, 0.211, 0.021, 0.012]}
    }
    client = _ProxyClient({"VAMPNET_BRIDGE_URL": bridge.url()})
    try:
        resp = client.send("tools/call", {"name": "vampnet_fit", "arguments": {"n_states": 4, "lag": 100}})
        assert resp["result"]["isError"] is False
        text = resp["result"]["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed["vamp2_score"] == 1.84

        # Bridge must have actually received the call with the right args.
        assert bridge.received[0][0] == "vampnet_fit"
        assert bridge.received[0][1]["n_states"] == 4
        assert bridge.received[0][1]["lag"] == 100
    finally:
        client.stop()
        bridge.stop()


def test_unknown_method_returns_method_not_found():
    bridge = _MockBridge()
    client = _ProxyClient({"VAMPNET_BRIDGE_URL": bridge.url()})
    try:
        resp = client.send("prompts/list", {})
        assert "error" in resp
        assert resp["error"]["code"] == -32601
    finally:
        client.stop()
        bridge.stop()


def test_bridge_unreachable_returns_clear_error():
    """When ChimeraX isn't running (no bridge), tools/list should return
    a JSON-RPC error with a clear hint, not crash."""
    # No bridge started — proxy is given a dead URL.
    client = _ProxyClient({"VAMPNET_BRIDGE_URL": "http://127.0.0.1:1"})
    try:
        resp = client.send("tools/list", {})
        assert "error" in resp
        assert "bridge not reachable" in resp["error"]["message"].lower() or \
               "unreachable" in resp["error"]["message"].lower()
    finally:
        client.stop()


if __name__ == "__main__":
    test_initialize_returns_capabilities()
    test_tools_list_proxies_manifest()
    test_tools_call_round_trips_arguments_and_result()
    test_unknown_method_returns_method_not_found()
    test_bridge_unreachable_returns_clear_error()
    print("OK")
