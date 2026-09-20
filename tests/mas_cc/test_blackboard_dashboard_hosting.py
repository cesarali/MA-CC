"""Behaviour needed to run the dashboard behind a name other than localhost."""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Thread

import pytest

from mas_cc.blackboard_dashboard.server import make_handler, normalise_base_path, serve_dashboard
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader

from test_blackboard_study_dashboard import _study


@contextmanager
def _serve(reader, **options):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader, **options))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _get(port, path, host=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=10)
    connection.putrequest("GET", path, skip_host=True)
    connection.putheader("Host", host if host is not None else f"127.0.0.1:{port}")
    connection.endheaders()
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, dict(response.getheaders()), body


def test_health_routes_answer_for_any_host(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader, allowed_hosts=["observatory.example"]) as port:
        for path in ("/healthz", "/readyz"):
            status, _, body = _get(port, path, host="10.244.3.7:8765")
            assert status == 200 and json.loads(body) == {"status": "ok"}


def test_foreign_host_header_is_refused_and_allowed_names_pass(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader, allowed_hosts=["Observatory.Example"]) as port:
        assert _get(port, "/api/study", host="attacker.example")[0] == 421
        assert _get(port, "/", host="attacker.example:80")[0] == 421
        assert _get(port, "/api/study", host="observatory.example")[0] == 200
        assert _get(port, "/api/study", host="observatory.example:8080")[0] == 200
        assert _get(port, "/api/study", host="localhost:1234")[0] == 200
        assert _get(port, "/api/study", host="[::1]:8765")[0] == 200


def test_without_an_allowlist_any_host_is_accepted(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader) as port:
        assert _get(port, "/api/study", host="anything.example")[0] == 200


def test_base_path_prefixes_every_route(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader, base_path="observatory/") as port:
        status, headers, _ = _get(port, "/observatory")
        assert status == 308 and headers["Location"] == "/observatory/"
        assert _get(port, "/observatory/")[0] == 200
        assert _get(port, "/observatory/app.js")[0] == 200
        status, _, body = _get(port, "/observatory/api/study")
        assert status == 200 and len(json.loads(body)["cells"]) == 2
        assert _get(port, "/api/study")[0] == 404
        assert _get(port, "/observatoryX/api/study")[0] == 404
        assert _get(port, "/healthz")[0] == 200


def test_normalise_base_path():
    assert [normalise_base_path(v) for v in (None, "", "/", "a", "/a/", "a/b/")] == ["", "", "", "/a", "/a", "/a/b"]


def test_front_end_uses_only_relative_urls():
    assets = files("mas_cc.blackboard_dashboard.assets")
    script = assets.joinpath("app.js").read_text(encoding="utf-8")
    page = assets.joinpath("index.html").read_text(encoding="utf-8")
    assert not re.search(r"[`'\"]/api", script)
    assert not re.search(r"(?:src|href)=\"/", page)


def test_non_loopback_bind_requires_an_allowed_host(tmp_path: Path):
    root = _study(tmp_path)
    with pytest.raises(ValueError, match="allowed-host"):
        serve_dashboard(root, host="0.0.0.0", port=0)


def test_access_log_is_one_json_line_per_request(tmp_path: Path, capsys):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader, access_log=True) as port:
        _get(port, "/api/study")
        _get(port, "/nope")
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert [(line["path"], line["status"]) for line in lines] == [("/api/study", 200), ("/nope", 404)]
    assert all(set(line) == {"event", "path", "status", "seconds", "host"} for line in lines)


def test_access_log_records_the_requested_path_under_a_base_path(tmp_path: Path, capsys):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader, access_log=True, base_path="/obs") as port:
        _get(port, "/obs/api/study")
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert [(line["path"], line["status"]) for line in lines] == [("/obs/api/study", 200)]
