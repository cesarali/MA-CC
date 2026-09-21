"""A published bundle must answer exactly like the live reader, from a directory or an object store."""
from __future__ import annotations

import gzip
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from mas_cc.blackboard_dashboard import PublishedStudyReader, write_dashboard_bundle
from mas_cc.blackboard_dashboard.server import make_handler
from mas_cc.blackboard_dashboard.store import StoreError, is_bundle_dir, is_store_url, open_store, safe_relative
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader

from test_blackboard_study_dashboard import _study

CELL = "config-0000~cell-0000"
EPISODE = "config-0000~cell-0000~episode-0000"


def _stable(payload):
    """Live payloads stamp their own refresh time and carry host paths in none of the fields compared."""
    text = json.dumps(payload, sort_keys=True, default=str)
    data = json.loads(text)
    if isinstance(data, dict):
        data.pop("refreshed_at", None)
    return data


class _FakeS3:
    """The one boto3 call the store makes, served from a directory."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root, self.prefix, self.keys = root, prefix, []

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3's signature
        self.keys.append((Bucket, Key))
        assert Key.startswith(self.prefix + "/")
        path = self.root / Key[len(self.prefix) + 1:]
        if not path.is_file():
            error = Exception("missing")
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error
        return {"Body": path.open("rb")}


@pytest.fixture()
def published(tmp_path: Path):
    (tmp_path / "source").mkdir()
    live = BlackboardStudyReader(_study(tmp_path / "source"), scheduler=False)
    bundle = tmp_path / "bundle"
    index = write_dashboard_bundle(live, bundle)
    return live, bundle, index


def test_published_payloads_equal_the_live_reader(published, tmp_path: Path):
    live, bundle, index = published
    reader = PublishedStudyReader(open_store(bundle, cache_dir=tmp_path / "cache"))
    assert index["kind"] == "blackboard_dashboard_bundle" and index["cells"] == 2
    assert _stable(reader.study()) == _stable(live.study())
    for cell in live.study()["cells"]:
        qualified = cell["qualified_id"]
        assert _stable(reader.cell(qualified)) == _stable(live.cell(qualified))
        assert _stable(reader.votes(qualified)) == _stable(live.votes(qualified))
        assert _stable(reader.prompt_examples(qualified)) == _stable(live.prompt_examples(qualified))
    assert _stable(reader.analysis_catalog()) == _stable(live.analysis_catalog())
    # The live status is computed without votes (cheap polling); the published one comes from the
    # full cell payload and is a superset: every field the live status fills in must agree.
    live_status, status = _stable(live.episode_status(EPISODE)), _stable(reader.episode_status(EPISODE))
    assert set(status) == set(live_status)
    assert {k: v for k, v in live_status.items() if v not in (None, {}, [])}.items() <= status.items()
    theirs, ours = live.episode_reader(EPISODE), reader.episode_reader(EPISODE)
    assert ours is reader.episode_reader(EPISODE)
    assert _stable(ours.timeline()) == _stable(theirs.timeline())
    assert _stable(ours.snapshot(0, 1, "agent_001")) == _stable(theirs.snapshot(0, 1, "agent_001"))
    assert _stable(ours.statistics()) == _stable(theirs.statistics())


def test_object_store_reads_are_few_validated_and_cached(published, tmp_path: Path):
    _, bundle, _ = published
    client = _FakeS3(bundle, "ctodie/study/dashboard")
    store = open_store("r2://research-bucket/ctodie/study/dashboard", client=client, cache_dir=tmp_path / "cache")
    reader = PublishedStudyReader(store)
    reader.study()
    reader.study()
    reader.cell(CELL)
    assert store.requests == 3  # index, study, cell - the second study() came from the cache
    assert {bucket for bucket, _ in client.keys} == {"research-bucket"}
    before = store.requests
    reader.episode_reader(EPISODE).timeline()
    fetched = store.requests - before
    assert fetched == sum(name.startswith(f"episodes/{EPISODE}/") for name in store.objects)
    reader.episode_reader(EPISODE).timeline()
    assert store.requests == before + fetched


def test_a_tampered_object_is_refused(published, tmp_path: Path):
    _, bundle, _ = published
    target = bundle / "cells" / f"{CELL}.json"
    target.write_text(target.read_text().replace("cell-0000", "cell-9999", 1))
    reader = PublishedStudyReader(open_store(bundle, cache_dir=tmp_path / "cache"))
    with pytest.raises(StoreError, match="failed validation"):
        reader.cell(CELL)


def test_unknown_ids_and_unsafe_paths_are_refused(published, tmp_path: Path):
    _, bundle, _ = published
    reader = PublishedStudyReader(open_store(bundle, cache_dir=tmp_path / "cache"))
    for bad in ("config-0000~cell-0042", "../../etc/passwd", "/abs", ""):
        with pytest.raises(ValueError):
            reader.cell(bad)
    with pytest.raises(ValueError):
        reader.episode_reader("config-0000~cell-0000~episode-0099")
    with pytest.raises(ValueError):
        reader.analysis_file("../index.json")
    for unsafe in ("../x", "/x", "a/../b", "a//b", "."):
        with pytest.raises(StoreError):
            safe_relative(unsafe)


def test_episode_files_are_stored_compressed_and_the_cache_is_bounded(published, tmp_path: Path):
    _, bundle, _ = published
    stored = sorted(p for p in (bundle / "episodes" / EPISODE).rglob("*") if p.is_file())
    assert stored and all(p.suffix == ".gz" for p in stored)
    assert gzip.decompress(stored[0].read_bytes())
    store = open_store(bundle, cache_dir=tmp_path / "cache", cache_limit_bytes=1)
    reader = PublishedStudyReader(store)
    reader.study()
    assert reader.episode_reader(EPISODE).timeline()["available_cursors"]  # siblings survive eviction while opening
    assert not (store.cache_dir / "study.json").exists()  # evicted: over the limit and not what is being opened
    assert reader.study()["cells"]  # and transparently fetched again


def test_location_detection(published, tmp_path: Path):
    _, bundle, _ = published
    assert is_bundle_dir(bundle) and not is_bundle_dir(tmp_path)
    assert is_store_url("s3://b/p") and is_store_url("r2://b/p") and not is_store_url(str(bundle))
    with pytest.raises(StoreError):
        open_store("r2:///no-bucket")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "index.json").write_text(json.dumps({"schema_version": 1, "objects": {}, "kind": "x"}))
    with pytest.raises(StoreError, match="not a blackboard dashboard bundle"):
        PublishedStudyReader(open_store(tmp_path / "other", cache_dir=tmp_path / "c2"))


def test_bundle_destination_must_be_empty(published):
    live, bundle, _ = published
    with pytest.raises(ValueError, match="not empty"):
        write_dashboard_bundle(live, bundle)


def test_http_server_serves_a_published_bundle(published, tmp_path: Path):
    live, bundle, _ = published
    reader = PublishedStudyReader(open_store(bundle, cache_dir=tmp_path / "cache"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)

        def get(path):
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read()

        status, body = get("/api/study")
        assert status == 200 and len(json.loads(body)["cells"]) == 2
        status, body = get(f"/api/study/cell/{CELL}")
        assert status == 200 and _stable(json.loads(body)) == _stable(live.cell(CELL))
        status, body = get(f"/api/study/episode/{EPISODE}/detail")
        assert status == 200 and json.loads(body)["timeline"]["available_cursors"]
        assert get("/api/study/cell/nope")[0] == 400
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_a_missing_boto3_is_reported_as_itself(monkeypatch, tmp_path: Path):
    import builtins

    real_import = builtins.__import__

    def no_boto3(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_boto3)
    with pytest.raises(StoreError, match=r"needs boto3 \(pip install 'mas-cc\[r2\]'\)"):
        open_store("r2://bucket/prefix", cache_dir=tmp_path / "cache")
