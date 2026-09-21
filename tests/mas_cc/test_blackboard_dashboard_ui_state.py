"""Front-end wiring that has no server side: title, URL state, history, opt-in auto-refresh."""
from __future__ import annotations

import re
from importlib.resources import files

ASSETS = files("mas_cc.blackboard_dashboard.assets")
SCRIPT = ASSETS.joinpath("app.js").read_text(encoding="utf-8")
PAGE = ASSETS.joinpath("index.html").read_text(encoding="utf-8")
STYLE = ASSETS.joinpath("style.css").read_text(encoding="utf-8")


def test_long_study_ids_can_wrap_in_the_title():
    assert "esc(study.study_id).replace(/_/g, '_<wbr>')" in SCRIPT  # escaped first, then break opportunities
    assert "classList.add('study-title')" in SCRIPT
    assert re.search(r"h1\.study-title\{[^}]*overflow-wrap:anywhere", STYLE)


def test_every_study_filter_and_the_sort_order_live_in_the_url():
    declared = re.search(r"const FILTER_PARAMS = (\[\[.*?\]\]);", SCRIPT).group(1)
    assert sorted(re.findall(r"'(filter-[a-z]+|cell-sort)'", declared)) == [
        "cell-sort", "filter-block", "filter-controller", "filter-rho", "filter-status"]
    assert "for (const [name, id] of FILTER_PARAMS) { if ($(id).value) params.set(name, $(id).value); }" in SCRIPT
    # a restored value is applied only if the study actually offers it
    assert "[...$(id).options].some(option => option.value === wanted)" in SCRIPT


def test_navigation_pushes_history_and_everything_else_replaces_it():
    assert "history[moved ? 'pushState' : 'replaceState']" in SCRIPT
    assert "[state.studyKey, state.cellId, state.episodeId]" in SCRIPT
    assert "window.addEventListener('popstate'" in SCRIPT
    assert "!state.restoring" in SCRIPT  # restoring from the URL must not create new entries


def test_auto_refresh_is_opt_in_bounded_and_never_a_free_running_interval():
    box = re.search(r'<input id="auto-refresh"[^>]*>', PAGE).group(0)
    assert "checked" not in box
    assert re.findall(r'<option value="(\d+)"', PAGE.split('id="auto-refresh-seconds"')[1].split("</select>")[0]) == ["10", "30", "60"]
    assert "setInterval(" not in SCRIPT  # the upstream rule stays: no background polling unless asked
    body = SCRIPT.split("function scheduleAutoRefresh()")[1].split("\n  }\n")[0]
    assert "if (!$('auto-refresh').checked || state.staticMode) return;" in body
    assert "document.hidden" in body and "finally { scheduleAutoRefresh(); }" in body
    assert "localStorage" not in body  # never persisted: a new tab starts with it off
