"""HTTP-level safety behaviour of the dashboard handler, exercised with stub readers."""
from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pytest

from mas_cc.blackboard_dashboard.data import BlackboardRunReader
from mas_cc.blackboard_dashboard.server import make_handler
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader


class _StudyStub(BlackboardStudyReader):
    """A study reader whose payloads are supplied by the test; no filesystem is touched."""

    def __init__(self, **behaviour):  # noqa: D107 - deliberately skips the real constructor
        self._behaviour = behaviour

    def study(self):
        return self._call("study")

    def cell(self, qualified_id):
        return self._call("cell")

    def analysis_catalog(self):
        return self._call("analysis_catalog")

    def _call(self, name):
        value = self._behaviour[name]
        if isinstance(value, BaseException):
            raise value
        return value


class _RunStub(BlackboardRunReader):
    def __init__(self, status):  # noqa: D107
        self._status = status

    def status(self):
        return self._status


@contextmanager
def _serve(reader):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", build_opener(ProxyHandler({}))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _error(opener, url):
    with pytest.raises(HTTPError) as caught:
        opener.open(url)
    return caught.value.code, json.loads(caught.value.read())


def test_unexpected_exception_is_a_json_500_not_a_dropped_connection(capsys):
    with _serve(_StudyStub(study=KeyError("config-0000~cell-0000"))) as (base, opener):
        code, body = _error(opener, base + "/api/study")
    assert code == 500
    assert body["error"] == "internal error"
    assert re.fullmatch(r"[0-9a-f]{12}", body["request_id"])
    assert "cell-0000" not in json.dumps(body)
    assert body["request_id"] in capsys.readouterr().err


def test_non_finite_and_numpy_values_serialise_as_valid_json():
    payload = {"mean": float("nan"), "upper": math.inf, "count": np.int64(3), "share": np.float64(0.25),
               "where": Path("cells/cell-0000"), "tags": {"b", "a"}, "nested": [{"value": np.float32("nan")}]}
    with _serve(_StudyStub(study=payload)) as (base, opener):
        with opener.open(base + "/api/study") as response:
            text = response.read().decode("utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"mean": None, "upper": None, "count": 3, "share": 0.25,
                                "where": "cells/cell-0000", "tags": ["a", "b"], "nested": [{"value": None}]}


def test_client_error_bodies_do_not_disclose_server_paths():
    message = "required dashboard artifact is missing: /shared/home/someone/results/studies/s1/study_manifest.json"
    with _serve(_StudyStub(cell=ValueError(message))) as (base, opener):
        code, body = _error(opener, base + "/api/study/cell/config-0000~cell-0000")
    assert code == 400
    assert body["error"] == "required dashboard artifact is missing: study_manifest.json"


def test_run_reader_payloads_use_the_same_encoder():
    with _serve(_RunStub({"value": float("nan")})) as (base, opener):
        with opener.open(base + "/api/status") as response:
            assert json.load(response) == {"value": None}


def test_missing_analysis_command_names_the_study_not_its_absolute_path(tmp_path):
    reader = BlackboardStudyReader.__new__(BlackboardStudyReader)
    reader.source_kind = "study"
    reader.study_dir = tmp_path / "deep" / "my_study"
    reader.study_dir.mkdir(parents=True)
    catalog = reader.analysis_catalog()
    assert catalog["available"] is False
    assert str(tmp_path) not in catalog["command"]
    assert catalog["command"].endswith("--study-dir my_study")


def test_controller_event_class_is_built_from_a_sanitised_token():
    script = files("mas_cc.blackboard_dashboard.assets").joinpath("app.js").read_text(encoding="utf-8")
    assert "${(row.message_types[0] || 'NO_OP')}" not in script
    assert "classToken(row.message_types[0] || 'NO_OP')" in script
    assert re.search(r"const classToken = value => String\(value\)\.replace\(/\[\^A-Za-z0-9_-\]/g, ''\)", script)
