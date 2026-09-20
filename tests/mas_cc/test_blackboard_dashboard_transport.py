"""Transport behaviour: compression, persistent connections, asset revalidation, streamed files."""
from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from mas_cc.blackboard_dashboard.server import GZIP_MIN_BYTES, make_handler
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader

from test_blackboard_study_dashboard import _study


@contextmanager
def _serve(reader):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _request(connection, path, **headers):
    connection.request("GET", path, headers=headers)
    response = connection.getresponse()
    return response, response.read()


def test_json_is_gzipped_only_when_the_client_accepts_it(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        plain, plain_body = _request(connection, "/api/study")
        packed, packed_body = _request(connection, "/api/study", **{"Accept-Encoding": "br, gzip"})
        refused, refused_body = _request(connection, "/api/study", **{"Accept-Encoding": "gzip;q=0"})
        connection.close()
    assert len(plain_body) >= GZIP_MIN_BYTES and plain.getheader("Content-Encoding") is None
    assert packed.getheader("Content-Encoding") == "gzip" and packed.getheader("Vary") == "Accept-Encoding"
    assert int(packed.getheader("Content-Length")) == len(packed_body) < len(plain_body)
    def stable(raw):  # each response stamps its own refresh time
        payload = json.loads(raw)
        payload.pop("refreshed_at", None)
        return payload

    assert stable(gzip.decompress(packed_body)) == stable(plain_body)
    assert refused.getheader("Content-Encoding") is None and stable(refused_body) == stable(plain_body)


def test_small_bodies_are_not_compressed(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        response, body = _request(connection, "/healthz", **{"Accept-Encoding": "gzip"})
        connection.close()
    assert response.getheader("Content-Encoding") is None and json.loads(body) == {"status": "ok"}


def test_one_connection_serves_several_requests(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        first, _ = _request(connection, "/api/study")
        socket_before = connection.sock
        second, _ = _request(connection, "/nope")
        third, _ = _request(connection, "/app.js")
        assert connection.sock is socket_before is not None
        connection.close()
    assert (first.version, first.status, second.status, third.status) == (11, 200, 404, 200)


def test_static_assets_revalidate_with_an_etag(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        first, body = _request(connection, "/app.js")
        etag = first.getheader("ETag")
        again, again_body = _request(connection, "/app.js", **{"If-None-Match": etag})
        stale, stale_body = _request(connection, "/app.js", **{"If-None-Match": '"something-else"'})
        api, _ = _request(connection, "/api/study")
        connection.close()
    assert etag and first.getheader("Cache-Control") == "no-cache" and body
    assert (again.status, again_body) == (304, b"")
    assert stale.status == 200 and stale_body == body
    assert api.getheader("Cache-Control") == "no-store" and api.getheader("ETag") is None


def test_downloads_are_streamed_with_the_right_disposition(tmp_path: Path, monkeypatch):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    package = tmp_path / "study_analysis.zip"
    package.write_bytes(b"PK" + bytes(3 * 1024 * 1024))
    plot = tmp_path / "figure.png"
    plot.write_bytes(b"\x89PNG" + bytes(64))
    targets = {"package": package, "plot": plot}
    monkeypatch.setattr(reader, "analysis_file", lambda identifier: targets[identifier])
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("whole-file read")))
    with _serve(reader) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        archive, archive_body = _request(connection, "/api/study/analysis/download?id=package")
        image, image_body = _request(connection, "/api/study/analysis/download?id=plot")
        connection.close()
    assert len(archive_body) == package.stat().st_size == int(archive.getheader("Content-Length"))
    assert archive.getheader("Content-Disposition") == 'attachment; filename="study_analysis.zip"'
    assert image.getheader("Content-Disposition") == 'inline; filename="figure.png"'
    assert image.getheader("Content-Type") == "image/png" and len(image_body) == 68


def test_persistent_connection_is_not_slower_than_fresh_connections(tmp_path: Path):
    """Regression: unbuffered header/body writes stalled every kept-alive request ~40 ms."""
    import time

    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    with _serve(reader) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        _request(connection, "/healthz")
        started = time.perf_counter()
        for _ in range(30):
            _request(connection, "/healthz")
        kept = time.perf_counter() - started
        connection.close()
    assert kept < 0.6, f"30 kept-alive requests took {kept:.3f}s"
