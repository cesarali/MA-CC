"""Several studies behind one server: catalog listing, study-scoped routes, picker wiring."""
from __future__ import annotations

import json
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Thread

import pytest

from mas_cc.blackboard_dashboard import write_dashboard_bundle
from mas_cc.blackboard_dashboard.catalog import StudyCatalog
from mas_cc.blackboard_dashboard.server import make_handler, serve_dashboard
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader

from test_blackboard_study_dashboard import _direct_grid, _study


@pytest.fixture()
def sources(tmp_path: Path):
    """One published prefix (catalog.json + a bundle) and one directory of live study roots."""
    (tmp_path / "src").mkdir()
    study = _study(tmp_path / "src")
    published = tmp_path / "published"
    index = write_dashboard_bundle(BlackboardStudyReader(study, scheduler=False), published / "alpha" / "dashboard")
    (published / "catalog.json").write_text(json.dumps({"schema_version": 1, "studies": [
        {"study": "alpha", "study_id": index["study_id"], "dashboard": "alpha/dashboard", "cells": 2,
         "episodes_with_detail": index["episodes_with_detail"], "published_at": index["generated_at"]}]}))
    live = tmp_path / "live"
    (live / "holder").mkdir(parents=True)
    grid = _direct_grid(live / "holder")  # .../holder/game/experiment/direct-grid-run
    grid.rename(live / "grid-run")
    (live / "not-a-study").mkdir()
    return published, live


@contextmanager
def _serve(reader, **options):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader, **options))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)

        def get(path):
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, json.loads(response.read() or b"null")

        yield get
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_catalog_lists_published_and_live_studies_without_locations(sources):
    published, live = sources
    catalog = StudyCatalog([str(published), str(live)], scheduler=False)
    entries = catalog.entries()
    assert [(e["key"], e["source"]) for e in entries] == [("alpha", "published"), ("grid-run", "live")]
    assert all(not any(str(published) in str(v) or str(live) in str(v) for v in e.values()) for e in entries)
    assert all(not key.startswith("_") for e in entries for key in e)


def test_study_scoped_routes_reach_the_right_reader(sources):
    published, live = sources
    catalog = StudyCatalog([str(published), str(live)], scheduler=False)
    with _serve(None, catalog=catalog) as get:
        status, listing = get("/api/catalog")
        assert status == 200 and [s["key"] for s in listing["studies"]] == ["alpha", "grid-run"]
        status, alpha = get("/s/alpha/api/study")
        assert status == 200 and len(alpha["cells"]) == 2
        status, grid = get("/s/grid-run/api/study")
        assert status == 200 and len(grid["cells"]) == 4
        status, cell = get("/s/alpha/api/study/cell/config-0000~cell-0000")
        assert status == 200 and cell["cell_id"] == "cell-0000"
        assert get("/s/alpha/api/study/episode/config-0000~cell-0000~episode-0000/detail")[0] == 200
        assert get("/s/nope/api/study")[0] == 400
        assert get("/api/study")[0] == 404  # catalog-only server: no default study
    assert catalog.reader("alpha") is catalog.reader("alpha")


def test_without_a_catalog_the_route_is_absent_and_the_single_study_still_works(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader) as get:
        assert get("/api/catalog")[0] == 404
        assert get("/api/study")[0] == 200
        assert get("/s/anything/api/study")[0] == 404


def test_reader_cache_is_bounded(sources):
    published, live = sources
    catalog = StudyCatalog([str(published), str(live)], scheduler=False, reader_limit=1)
    first = catalog.reader("alpha")
    catalog.reader("grid-run")
    assert catalog.reader("alpha") is not first


def test_serving_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to serve"):
        serve_dashboard(None, port=0)


def test_front_end_boots_through_the_catalog_and_scopes_every_api_call():
    assets = files("mas_cc.blackboard_dashboard.assets")
    script = assets.joinpath("app.js").read_text(encoding="utf-8")
    page = assets.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="study-picker"' in page and 'id="study-picker-wrap"' in page and "hidden" in page.split('id="study-picker-wrap"')[1][:40]
    assert "await fetch(api(path)" in script and "get('api/catalog')" in script
    assert "path !== 'api/catalog'" in script  # the catalog itself is never study-scoped
    assert "api(`api/study/analysis/download" in script  # direct URLs (plot images, downloads) are scoped too
    assert "params.set('study', state.studyKey)" in script


def test_scheduler_can_be_switched_off_for_hosts_without_slurm(sources, monkeypatch):
    """The hosted pod has no squeue/sacct; a live catalog must not shell out per request."""
    import subprocess

    published, live = sources
    calls = []
    real = subprocess.run

    def watched(command, *args, **kwargs):
        calls.append(command[0] if isinstance(command, (list, tuple)) else command)
        return real(["true"], *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", watched)
    catalog = StudyCatalog([str(published), str(live)], scheduler=False)
    reader = catalog.reader("grid-run")
    reader.study()
    assert not [c for c in calls if "squeue" in str(c) or "sacct" in str(c)]


def test_serve_dashboard_passes_the_scheduler_choice_through():
    import inspect

    from mas_cc.blackboard_dashboard import server as server_module

    assert "scheduler" in inspect.signature(server_module.serve_dashboard).parameters
    source = inspect.getsource(server_module.serve_dashboard)
    assert "StudyCatalog(catalogs, scheduler=scheduler)" in source
