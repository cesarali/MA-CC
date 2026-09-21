"""The episode view's decision-audit drill-down: a count in the payload, lazy per-audit fetches."""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Thread

from mas_cc.blackboard_dashboard.server import make_handler
from mas_cc.blackboard_dashboard.study_data import BlackboardStudyReader

from test_blackboard_study_dashboard import _study

EPISODE = "config-0000~cell-0000~episode-0000"
ASSETS = files("mas_cc.blackboard_dashboard.assets")


@contextmanager
def _serve(reader):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reader))
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


def test_detail_carries_the_audit_range_so_the_ui_need_not_probe(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    expected = reader.episode_reader(EPISODE).prompt_count()
    with _serve(reader) as get:
        status, detail = get(f"/api/study/episode/{EPISODE}/detail")
        assert status == 200
        assert detail["prompt_attempts"] == expected
        # every index the count advertises must actually resolve, and one past it must not
        for index in range(expected):
            assert get(f"/api/study/episode/{EPISODE}/prompt-{index}")[0] == 200
        assert get(f"/api/study/episode/{EPISODE}/prompt-{expected}")[0] == 400


def test_prompt_payload_is_the_audit_record(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    if not reader.episode_reader(EPISODE).prompt_count():
        return
    with _serve(reader) as get:
        status, payload = get(f"/api/study/episode/{EPISODE}/prompt-0")
    assert status == 200 and payload["audit_index"] == 0 and isinstance(payload["audit"], dict)


def test_prompt_count_matches_the_loaded_audits(tmp_path: Path):
    reader = BlackboardStudyReader(_study(tmp_path), scheduler=False)
    episode = reader.episode_reader(EPISODE)
    assert episode.prompt_count() == len(episode._load()["audits"])


def test_front_end_lists_audits_lazily_and_explains_a_lean_profile():
    script = ASSETS.joinpath("app.js").read_text(encoding="utf-8")
    page = ASSETS.joinpath("index.html").read_text(encoding="utf-8")
    assert 'data-view="prompts"' in page and '<section id="prompts" class="view">' in page
    assert 'id="prompt-list"' in page
    assert "renderPromptList(id, Number(detail.prompt_attempts) || 0)" in script
    # fetched on open, not upfront, and cached per episode
    body = script.split("function renderPromptList(")[1].split("\n  }\n")[0]
    assert "addEventListener('toggle'" in body and "if (!entry.open || entry.dataset.loaded) return;" in body
    assert "state.promptCache.set(index, payload)" in body
    assert "if (state.episodeId !== episodeId) return;" in body  # a late reply must not land on another episode
    # a lean retention profile keeps the audit but not the prompt text; the UI says which it is
    assert re.search(r"audit\.semantic_only \?", script)
    assert "No compiled prompt on this audit record." in script


def test_every_tab_has_a_top_level_section():
    """A view nested inside another view inherits display:none and renders blank.

    The Prompts section was first inserted inside #controller: the markup assertions all passed and
    the tab showed nothing at all in a browser.
    """
    page = ASSETS.joinpath("index.html").read_text(encoding="utf-8")
    depth, nested, seen = 0, [], []
    for match in re.finditer(r'<section(?: id="([a-z-]+)")? class="view[^"]*"|</section>', page):
        if match.group(0).startswith("</"):
            depth = max(0, depth - 1)
        else:
            if match.group(1):
                seen.append(match.group(1))
                if depth:
                    nested.append(match.group(1))
            depth += 1
    assert seen, "no view sections found - the markup shape changed"
    assert nested == [], f"view sections nested inside another view: {nested}"
    tabs = set(re.findall(r'data-view="([a-z-]+)"', page))
    assert tabs - {"overview"} <= set(seen), f"a tab points at no section: {tabs - {'overview'} - set(seen)}"
    assert re.findall(r'<button class="active" data-view="([a-z-]+)"', page) == ["overview"]
