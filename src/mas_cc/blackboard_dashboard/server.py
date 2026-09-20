"""Local read-only HTTP server and portable export for the dashboard."""

from __future__ import annotations

import json
import math
import mimetypes
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Iterable, TypeAlias
from urllib.parse import parse_qs, unquote, urlparse

from .data import BlackboardRunReader
from .study_data import BlackboardStudyReader, is_direct_grid_root, is_study_root


DashboardReader: TypeAlias = BlackboardRunReader | BlackboardStudyReader


def resolve_dashboard_collection(path: str | Path) -> Path:
    """Resolve one exact or uniquely nested study/direct-grid collection."""

    root = Path(path).expanduser().resolve()
    if is_study_root(root) or is_direct_grid_root(root):
        return root
    candidates = {
        candidate.resolve()
        for candidate in root.glob("*/*/*")
        if candidate.is_dir()
        and (is_study_root(candidate) or is_direct_grid_root(candidate))
    }
    if not candidates:
        return root
    if len(candidates) > 1:
        raise ValueError(
            f"multiple dashboard collections found beneath {root}; pass the exact run root"
        )
    return next(iter(candidates))


def _asset(name: str) -> bytes:
    return files("mas_cc.blackboard_dashboard.assets").joinpath(name).read_bytes()


def _encode_default(value: object) -> object:
    """Encode the non-JSON values a reader can hand back (numpy scalars, paths, sets)."""

    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _finite(value: object) -> object:
    """Replace NaN and infinities with null; browsers reject the literals json emits."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_finite(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item") or hasattr(value, "tolist"):
        return _finite(_encode_default(value))
    return value


def _json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=_encode_default,
        )
    except ValueError:
        # A non-finite float is somewhere in the payload: pay for the walk only then.
        text = json.dumps(
            _finite(value), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    return text.encode("utf-8")


_ABSOLUTE_PATH = re.compile(r"(?<![\w.~-])/(?:[^\s/:'\"]+/)+(?=[^\s/:'\"])")


def _public_error(exc: BaseException) -> str:
    """The message for a 4xx body, with server directory names removed."""

    return _ABSOLUTE_PATH.sub("", str(exc))


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
HEALTH_PATHS = frozenset({"/healthz", "/readyz"})


def normalise_base_path(base_path: str | None) -> str:
    """'' for the root, otherwise '/prefix' with no trailing slash."""

    cleaned = "/" + (base_path or "").strip().strip("/")
    return "" if cleaned == "/" else cleaned


def _host_name(header: str) -> str:
    """The host part of a Host header, lower-cased, without port or IPv6 brackets."""

    value = header.strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def make_handler(
    reader: DashboardReader,
    *,
    base_path: str | None = None,
    allowed_hosts: Iterable[str] | None = None,
    access_log: bool = False,
):
    """Build the request handler.

    ``allowed_hosts`` is the Host-header allowlist (DNS-rebinding defence). ``None`` keeps the
    historical behaviour of accepting any Host, which is only safe on a loopback bind;
    ``serve_dashboard`` always passes a list. Health routes are exempt: a kubelet probe sends
    the pod IP as Host and the routes disclose nothing.
    """

    prefix = normalise_base_path(base_path)
    hosts = None if allowed_hosts is None else {h.strip().lower() for h in allowed_hosts} | LOOPBACK_HOSTS

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "MASCCBlackboard/1"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            started = time.monotonic()
            requested = urlparse(self.path).path  # before the base path is stripped
            self._status = 0
            try:
                self._route()
            finally:
                if access_log:
                    print(
                        json.dumps(
                            {
                                "event": "request",
                                "path": requested,
                                "status": self._status,
                                "seconds": round(time.monotonic() - started, 4),
                                "host": _host_name(self.headers.get("Host", "")),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

        def _route(self) -> None:
            path = urlparse(self.path).path
            if path in HEALTH_PATHS:
                self._send(200, "application/json", _json({"status": "ok"}))
                return
            if hosts is not None and _host_name(self.headers.get("Host", "")) not in hosts:
                self._send(421, "application/json", _json({"error": "host not allowed"}))
                return
            if prefix:
                if path == prefix:
                    self.send_response(308)
                    self.send_header("Location", prefix + "/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    self._status = 308
                    return
                if not path.startswith(prefix + "/"):
                    self._send(404, "application/json", _json({"error": "not found"}))
                    return
                self.path = self.path[len(prefix):]
            self._serve()

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self._status = status
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

        def _send_file(self, path: Path) -> None:
            body = path.read_bytes()
            self._status = 200
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _serve(self) -> None:
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
                    if parsed.path == "/api/study/analysis":
                        self._send(
                            200, "application/json", _json(reader.analysis_catalog())
                        )
                        return
                    if parsed.path == "/api/study/analysis/download":
                        identifier = parse_qs(parsed.query).get("id", [""])[0]
                        self._send_file(reader.analysis_file(identifier))
                        return
                    prefix = "/api/study/cell/"
                    if parsed.path.startswith(prefix):
                        token = unquote(parsed.path.removeprefix(prefix))
                        prompts = token.endswith("/prompts")
                        votes = token.endswith("/votes")
                        if prompts:
                            token = token.removesuffix("/prompts")
                        elif votes:
                            token = token.removesuffix("/votes")
                        payload = (
                            reader.prompt_examples(token)
                            if prompts
                            else reader.votes(token)
                            if votes
                            else reader.cell(token)
                        )
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
                        if separator and (
                            action in {"detail", "timeline", "snapshot"}
                            or action.startswith("prompt-")
                        ):
                            episode_reader = reader.episode_reader(unquote(token))
                            if action == "detail":
                                timeline = episode_reader.timeline()
                                edge = (
                                    timeline["available_cursors"][-1]
                                    if timeline["available_cursors"]
                                    else None
                                )
                                payload = {
                                    "schema_version": 1,
                                    "timeline": timeline,
                                    "snapshot": episode_reader.snapshot(
                                        edge["round_index"] if edge else None,
                                        edge["step"] if edge else None,
                                    ),
                                    "statistics": episode_reader.statistics(),
                                }
                            elif action == "timeline":
                                payload = episode_reader.timeline()
                            elif action.startswith("prompt-"):
                                prompt_id = action.removeprefix("prompt-")
                                if not prompt_id.isdigit():
                                    raise ValueError(
                                        "prompt identifier must be a non-negative integer"
                                    )
                                payload = episode_reader.prompt(int(prompt_id))
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
            except (BrokenPipeError, ConnectionResetError):
                return  # the client went away; there is no socket left to answer on
            except (OSError, ValueError) as exc:
                self._send(
                    400, "application/json", _json({"error": _public_error(exc)})
                )
            except Exception:  # noqa: BLE001 - a reader defect must not drop the connection
                request_id = uuid.uuid4().hex[:12]
                print(
                    f"dashboard request {request_id} {parsed.path} failed\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    self._send(
                        500,
                        "application/json",
                        _json({"error": "internal error", "request_id": request_id}),
                    )
                except OSError:
                    return

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def serve_dashboard(
    run_dir: str | Path,
    *,
    episode_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    allowed_hosts: Iterable[str] = (),
    base_path: str | None = None,
    access_log: bool = False,
) -> None:
    source = resolve_dashboard_collection(run_dir)
    reader: DashboardReader = (
        BlackboardStudyReader(source)
        if is_study_root(source) or is_direct_grid_root(source)
        else BlackboardRunReader(source, episode_id)
    )
    allowed = [name for name in allowed_hosts if name.strip()]
    if host not in LOOPBACK_HOSTS and not allowed:
        raise ValueError(
            "the dashboard has no authentication: binding to a non-loopback address requires "
            "at least one --allowed-host (the name clients will use), and the network path to "
            "it must be access-controlled; otherwise bind to localhost and use an SSH tunnel"
        )
    handler = make_handler(
        reader, base_path=base_path, allowed_hosts=allowed, access_log=access_log
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    if threading.current_thread() is threading.main_thread():
        signal.signal(
            signal.SIGTERM,
            lambda *_: threading.Thread(target=server.shutdown, daemon=True).start(),
        )
    print(
        f"Blackboard dashboard: http://{host}:{server.server_port}{normalise_base_path(base_path)}/"
    )
    if isinstance(reader, BlackboardStudyReader):
        print(f"Collection: {reader.study_dir}")
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
