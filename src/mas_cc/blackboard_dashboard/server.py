"""Local read-only HTTP server and portable export for the dashboard."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import BlackboardRunReader


def _asset(name: str) -> bytes:
    return files("mas_cc.blackboard_dashboard.assets").joinpath(name).read_bytes()


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def make_handler(reader: BlackboardRunReader):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "MASCCBlackboard/1"

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"/", "/index.html"}:
                    self._send(200, "text/html; charset=utf-8", _asset("index.html"))
                    return
                if parsed.path == "/app.js":
                    self._send(200, "text/javascript; charset=utf-8", _asset("app.js"))
                    return
                if parsed.path == "/style.css":
                    self._send(200, "text/css; charset=utf-8", _asset("style.css"))
                    return
                if parsed.path == "/api/status":
                    self._send(200, "application/json", _json(reader.status()))
                    return
                if parsed.path == "/api/timeline":
                    self._send(200, "application/json", _json(reader.timeline()))
                    return
                if parsed.path == "/api/snapshot":
                    query = parse_qs(parsed.query)
                    round_index = int(query["round"][0]) if "round" in query else None
                    step = int(query["step"][0]) if "step" in query else None
                    agent = query.get("agent", [None])[0]
                    self._send(
                        200,
                        "application/json",
                        _json(reader.snapshot(round_index, step, agent)),
                    )
                    return
                if parsed.path.startswith("/api/prompt/"):
                    token = parsed.path.removeprefix("/api/prompt/")
                    if not token.isdigit():
                        raise ValueError(
                            "prompt identifier must be a non-negative integer"
                        )
                    self._send(
                        200, "application/json", _json(reader.prompt(int(token)))
                    )
                    return
                self._send(404, "application/json", _json({"error": "not found"}))
            except (OSError, ValueError) as exc:
                self._send(400, "application/json", _json({"error": str(exc)}))

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def serve_dashboard(
    run_dir: str | Path,
    *,
    episode_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    reader = BlackboardRunReader(run_dir, episode_id)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "dashboard must bind to localhost; use an SSH tunnel for remote viewing"
        )
    server = ThreadingHTTPServer((host, port), make_handler(reader))
    print(f"Blackboard dashboard: http://{host}:{server.server_port}")
    print(f"Episode: {reader.episode_dir}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def export_dashboard(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    episode_id: str | None = None,
) -> Path:
    reader = BlackboardRunReader(run_dir, episode_id)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timeline = reader.timeline()
    snapshots = {}
    agents = {}
    for cursor in timeline["available_cursors"]:
        key = f"{cursor['phase']}:{cursor['round_index']}:{cursor['step']}"
        snapshot = reader.snapshot(cursor["round_index"], cursor["step"])
        snapshot.pop("available_cursors", None)
        snapshot.pop("agent", None)
        snapshots[key] = snapshot
        agents[key] = {
            agent_id: reader.snapshot(cursor["round_index"], cursor["step"], agent_id)[
                "agent"
            ]
            for agent_id in timeline["agents"]
        }
    bundle = {
        "schema_version": 1,
        "static_bundle": True,
        "timeline": timeline,
        "snapshots": snapshots,
        "agents": agents,
    }
    embedded = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = (
        _asset("index.html")
        .decode("utf-8")
        .replace(
            '<script id="dashboard-data" type="application/json"></script>',
            f'<script id="dashboard-data" type="application/json">{embedded}</script>',
        )
    )
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / "app.js").write_bytes(_asset("app.js"))
    (destination / "style.css").write_bytes(_asset("style.css"))
    return destination / "index.html"
