from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.experiments.configured_analysis import run_configured_analysis
from mas_cc.games.hidden_bench.imitation.analysis import INFORMATION_STATISTICS


def _config():
    return load_run_config(
        "configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml",
        environment={},
    )


def test_configured_hidden_bench_analysis_forwards_every_setting(tmp_path, monkeypatch):
    captured = {}

    def fake_analyze(run_dir, output_dir, **kwargs):
        captured.update(run_dir=run_dir, output_dir=output_dir, **kwargs)
        return {"n_events": 20}

    monkeypatch.setattr(
        "mas_cc.games.hidden_bench.imitation.analysis.analyze_hidden_bench_imitation",
        fake_analyze,
    )

    summary = run_configured_analysis(_config(), tmp_path)

    assert summary == {"n_events": 20}
    assert captured["run_dir"] == tmp_path
    assert captured["output_dir"] == tmp_path / "hidden_bench_imitation_analysis"
    assert captured["statistics"] == INFORMATION_STATISTICS
    assert captured["bootstrap_resamples"] == 1000
    assert captured["null_permutations"] == 1000
    assert captured["confidence"] == 0.95
    assert captured["seed"] == 20260810
    # This run config has both analysis.comet_export and the logging.comet
    # master switch on, so the combined flag forwarded to the analysis
    # function is true too.
    assert captured["comet_export"] is True
    assert captured["comet_project"] == "mas-cc"
    assert captured["comet_run_name"] == "hidden-bench-imitation-reasoning-control-10-20260810/analysis"


def test_comet_export_is_off_when_the_logging_master_switch_is_off(tmp_path, monkeypatch):
    captured = {}

    def fake_analyze(run_dir, output_dir, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "mas_cc.games.hidden_bench.imitation.analysis.analyze_hidden_bench_imitation",
        fake_analyze,
    )
    config = _config()
    config = replace(config, logging=replace(config.logging, comet=False))

    run_configured_analysis(config, tmp_path)

    assert captured["comet_export"] is False


def test_disabled_configured_analysis_is_a_noop(tmp_path):
    config = _config()
    config = replace(config, analysis=replace(config.analysis, enabled=False))
    assert run_configured_analysis(config, tmp_path) is None


def test_configured_analysis_rejects_unknown_statistics_before_reading_results(tmp_path):
    config = _config()
    config = replace(config, analysis=replace(config.analysis, estimators=("typo_mi",)))
    with pytest.raises(ValueError, match="unsupported.*typo_mi"):
        run_configured_analysis(config, tmp_path)
