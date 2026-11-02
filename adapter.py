"""Thin transport adapter around engine.solve() - the ONLY place with I/O and networking.

Transport choice: Python's standard-library http.server (plus a CLI).
    * The website already talks JSON over HTTP (Vite proxies /api to a local server), so HTTP is what
      it can call; a CLI is handy for lab scripts and CI.
    * The lab rules allow only NumPy + the standard library, so no Flask/FastAPI is needed or used.

CLI
    python adapter.py solve samples/lab_instance.json -o out.json [--progress]
    python adapter.py solve - < instance.json                 (stdin -> stdout)

HTTP  (python adapter.py serve --port 8001)
    GET  /engine/health                -> {"ok": true}
    GET  /engine/algorithms            -> names, catalogue numbers and default parameters
    POST /engine/solve                 -> solve(body), synchronous
    POST /engine/runs                  -> start a background run, returns {"run_id"}
    GET  /engine/runs/<id>             -> {"status", "progress", "result"?}
    POST /engine/runs/<id>/cancel      -> early cancellation; the best front so far is returned
Errors are JSON: {"error": "..."} with HTTP 400 / 404.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from engine import solve
from engine.algorithms import DEFAULT_ALGORITHM, REGISTRY

MAX_BODY = 5 * 1024 * 1024
CATALOGUE = {"weighted_sum": 1, "eps_constraint": 2, "goal_programming": 6, "lexicographic": 7, "nsga2": 14, "nsga3": 15,
             "spea2": 16, "moead": 18, "constrained_nsga2": 21, "mosa": 26, "moma": 30}


# ---------------------------------------------------------------------------
# background runs
# ---------------------------------------------------------------------------
class RunStore:
    def __init__(self, keep: int = 50):
        self.runs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.keep = keep

    def start(self, instance: dict) -> str:
        run_id = uuid.uuid4().hex[:12]
        cancel = threading.Event()
        run = {"status": "running", "progress": None, "result": None, "cancel": cancel}
        with self.lock:
            if len(self.runs) >= self.keep:  # forget the oldest finished run
                for k in [k for k, r in self.runs.items() if r["status"] != "running"][:1]:
                    del self.runs[k]
            self.runs[run_id] = run

        def progress(entry):
            run["progress"] = entry

        def work():
            result = solve(instance, progress=progress, should_stop=cancel.is_set)
            run["result"] = result
            run["status"] = "error" if "error" in result else ("cancelled" if cancel.is_set() else "done")

        threading.Thread(target=work, daemon=True).start()
        return run_id

    def get(self, run_id: str) -> dict | None:
        r = self.runs.get(run_id)
        if r is None:
            return None
        out = {"status": r["status"], "progress": r["progress"]}
        if r["status"] != "running":
            out["result"] = r["result"]
        return out

    def cancel(self, run_id: str) -> bool:
        r = self.runs.get(run_id)
        if r is None:
            return False
        r["cancel"].set()
        return True


STORE = RunStore()


def algorithms_info() -> list[dict]:
    return [
        {"name": n, "catalogue_no": CATALOGUE.get(n), "default": n == DEFAULT_ALGORITHM, "defaults": m.DEFAULTS,
         "summary": (m.__doc__ or "").strip().splitlines()[0]}
        for n, m in REGISTRY.items()
    ]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "TimetableEngine/1.0"

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin", "")
        if origin.startswith(("http://localhost", "http://127.0.0.1")):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("Request body too large.")
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"null")
        except json.JSONDecodeError as e:
            raise ValueError(f"Body is not valid JSON: {e}") from None

    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("[adapter] " + fmt % args + "\n")

    def do_OPTIONS(self):
        self.send_response(204)
        origin = self.headers.get("Origin", "")
        if origin.startswith(("http://localhost", "http://127.0.0.1")):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parts = self.path.rstrip("/").split("/")
        if self.path.rstrip("/") == "/engine/health":
            return self._send(200, {"ok": True})
        if self.path.rstrip("/") == "/engine/algorithms":
            return self._send(200, algorithms_info())
        if len(parts) == 4 and parts[1:3] == ["engine", "runs"]:
            r = STORE.get(parts[3])
            return self._send(200, r) if r else self._send(404, {"error": "Run not found."})
        self._send(404, {"error": f"Unknown path {self.path}"})

    def do_POST(self):
        parts = self.path.rstrip("/").split("/")
        try:
            if self.path.rstrip("/") == "/engine/solve":
                result = solve(self._body())
                return self._send(400 if "error" in result else 200, result)
            if self.path.rstrip("/") == "/engine/runs":
                body = self._body()
                if not isinstance(body, dict):
                    return self._send(400, {"error": "The instance must be a JSON object."})
                return self._send(200, {"run_id": STORE.start(body)})
            if len(parts) == 5 and parts[1:3] == ["engine", "runs"] and parts[4] == "cancel":
                return self._send(200, {"cancelled": True}) if STORE.cancel(parts[3]) else self._send(404, {"error": "Run not found."})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": f"Unknown path {self.path}"})


def make_server(host: str = "127.0.0.1", port: int = 8001) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Timetable engine adapter (CLI + HTTP)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve one JSON instance")
    s.add_argument("input", help="path to the instance JSON, or - for stdin")
    s.add_argument("-o", "--output", help="write the result here (default: stdout)")
    s.add_argument("--progress", action="store_true", help="print progress to stderr")
    v = sub.add_parser("serve", help="run the HTTP server")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8001)
    a = ap.parse_args(argv)

    if a.cmd == "solve":
        try:
            text = sys.stdin.read() if a.input == "-" else Path(a.input).read_text(encoding="utf-8")
            instance = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            result = {"error": f"Could not read input: {e}"}
        else:
            def progress(e):
                print(f"gen {e['gen']:4d}  evals {e['evaluations']:6d}  HV {e['hv']:.4f}  best V {e['best_violation']}", file=sys.stderr)

            result = solve(instance, progress=progress if a.progress else None)
        text = json.dumps(result, indent=2)
        if a.output:
            Path(a.output).write_text(text, encoding="utf-8")
            print(f"wrote {a.output}" + (f"  (error: {result['error']})" if "error" in result else ""), file=sys.stderr)
        else:
            print(text)
        return 1 if "error" in result else 0

    srv = make_server(a.host, a.port)
    print(f"Timetable engine listening on http://{a.host}:{a.port}/engine/  (Ctrl+C to stop)", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
