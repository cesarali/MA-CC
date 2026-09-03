"""Local read-only HTTP server and portable export for the dashboard."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import TypeAlias
from urllib.parse import parse_qs, unquote, urlparse

from .data import BlackboardRunReader
from .study_data import BlackboardStudyReader, is_study_root


DashboardReader: TypeAlias = BlackboardRunReader | BlackboardStudyReader


def _asset(name: str) -> bytes:
    return files("mas_cc.blackboard_dashboard.assets").joinpath(name).read_bytes()


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def make_handler(reader: DashboardReader):
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
                if isinstance(reader, BlackboardStudyReader):
                    if parsed.path == "/api/study":
                        self._send(200, "application/json", _json(reader.study()))
                        return
                    if parsed.path == "/api/study/cells":
                        self._send(200, "application/json", _json(reader.cells()))
                        return
                    prefix = "/api/study/cell/"
                    if parsed.path.startswith(prefix):
                        token = unquote(parsed.path.removeprefix(prefix))
                        votes = token.endswith("/votes")
                        if votes:
                            token = token.removesuffix("/votes")
                        payload = reader.votes(token) if votes else reader.cell(token)
                        self._send(200, "application/json", _json(payload))
                        return
                    episode_prefix = "/api/study/episode/"
                    if parsed.path.startswith(episode_prefix) and parsed.path.endswith(
                        "/status"
                    ):
                        token = unquote(
                            parsed.path.removeprefix(episode_prefix).removesuffix(
                                "/status"
                            )
                        )
                        self._send(
                            200, "application/json", _json(reader.episode_status(token))
                        )
                        return
                    if parsed.path.startswith(episode_prefix):
                        remainder = parsed.path.removeprefix(episode_prefix)
                        token, separator, action = remainder.rpartition("/")
                        if separator and action in {"timeline", "snapshot"}:
                            episode_reader = reader.episode_reader(unquote(token))
                            if action == "timeline":
                                payload = episode_reader.timeline()
                            else:
                                query = parse_qs(parsed.query)
                                round_index = (
                                    int(query["round"][0]) if "round" in query else None
                                )
                                step = (
                                    int(query["step"][0]) if "step" in query else None
                                )
                                agent = query.get("agent", [None])[0]
                                payload = episode_reader.snapshot(
                                    round_index, step, agent
                                )
                            self._send(200, "application/json", _json(payload))
                            return
                    if parsed.path.startswith("/api/"):
                        self._send(
                            404, "application/json", _json({"error": "not found"})
                        )
                        return
                if parsed.path == "/api/status" and isinstance(
                    reader, BlackboardRunReader
                ):
                    self._send(200, "application/json", _json(reader.status()))
                    return
                if parsed.path == "/api/timeline" and isinstance(
                    reader, BlackboardRunReader
                ):
                    self._send(200, "application/json", _json(reader.timeline()))
                    return
                if parsed.path == "/api/snapshot" and isinstance(
                    reader, BlackboardRunReader
                ):
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
                if parsed.path.startswith("/api/prompt/") and isinstance(
                    reader, BlackboardRunReader
                ):
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
    source = Path(run_dir).expanduser().resolve()
    reader: DashboardReader = (
        BlackboardStudyReader(source)
        if is_study_root(source)
        else BlackboardRunReader(source, episode_id)
    )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "dashboard must bind to localhost; use an SSH tunnel for remote viewing"
        )
    server = ThreadingHTTPServer((host, port), make_handler(reader))
    print(f"Blackboard dashboard: http://{host}:{server.server_port}")
    if isinstance(reader, BlackboardStudyReader):
        print(f"Study: {reader.study_dir}")
    else:
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
