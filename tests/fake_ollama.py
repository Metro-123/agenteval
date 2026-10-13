"""A real HTTP server that speaks enough of Ollama's API to test the client.

Deliberately not a mock of ``OllamaClient`` -- it binds a socket and speaks HTTP,
so the tests exercise URL construction, JSON encoding, response parsing, token
accounting, error mapping and retry behaviour for real. Only the inference is
faked, because real inference needs weights the user has no data budget for.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Models this fake claims to have installed.
INSTALLED = ["qwen2.5:0.5b", "llama3.2:1b", "judge:1b"]


class _Handler(BaseHTTPRequestHandler):
    # set by serve()
    state: dict = {}

    def log_message(self, *args) -> None:  # silence request logging
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send(200, {"models": [
                {"name": n, "size": 1_000_000, "digest": "sha256:fake"}
                for n in INSTALLED]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send(404, {"error": "not found"})
            return

        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(raw.decode("utf-8"))
        self.state["requests"].append(req)
        self.state["headers"].append(dict(self.headers))

        # scripted failure injection, for the retry tests
        if self.state.get("fail_next"):
            self.state["fail_next"] -= 1
            self._send(503, {"error": "loading model"})
            return

        model = req.get("model")
        if model not in INSTALLED:
            self._send(404, {"error": f"model '{model}' not found"})
            return

        user = req["messages"][-1]["content"]
        system = req["messages"][0]["content"]
        options = req.get("options", {})

        if "PASS" in system:  # a judge
            good = "the answer is 42" in user
            reply = "PASS" if good else "FAIL"
        else:
            reply = self.state["reply"] or "the answer is 42"

        # honour seed/stream/keep_alive so tests can assert the client sent them
        self._send(200, {
            "model": model,
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": reply},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": len(user.split()),
            "eval_count": len(reply.split()),
        })


def serve(reply: str = "the answer is 42"):
    """Start the fake on a free port. Returns ``(base_url, state, shutdown)``."""
    state = {"requests": [], "headers": [], "fail_next": 0, "reply": reply}
    handler = type("H", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    def shutdown() -> None:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)

    return f"http://127.0.0.1:{httpd.server_address[1]}", state, shutdown
