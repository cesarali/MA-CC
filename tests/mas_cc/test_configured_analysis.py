from dataclasses import replace
from pathlib import Path

import pytest

from mas_cc.config import load_run_config
from mas_cc.experiments.configured_analysis import (
    per_cell_reports_enabled,
    run_configured_analysis,
    run_configured_cell_analysis,
)
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
    expected = _config()
    assert captured["seed"] == expected.analysis.options["seed"]
    # This run config has both analysis.comet_export and the logging.comet
    # master switch on, so the combined flag forwarded to the analysis
    # function is true too.
    assert captured["comet_export"] is True
    assert captured["comet_project"] == "mas-cc"
    assert captured["comet_run_name"] == (
        f"{expected.experiment.name}-{expected.execution.seed}/analysis"
    )


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


def test_cell_analysis_keeps_only_parameter_named_markdown_reports(tmp_path, monkeypatch):
    captured = {}

    def fake_analyze(run_dir, output_dir, **kwargs):
        captured.update(run_dir=Path(run_dir), output_dir=Path(output_dir), **kwargs)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "information_estimates.md").write_text("# MI\n", encoding="utf-8")
        (Path(output_dir) / "truth_current_estimates.md").write_text(
            "# Truth current\n", encoding="utf-8"
        )
        (Path(output_dir) / "large_intermediate.csv").write_text("discarded\n", encoding="utf-8")
        return {"n_events": 20}

    monkeypatch.setattr(
        "mas_cc.games.hidden_bench.imitation.analysis.analyze_hidden_bench_imitation",
        fake_analyze,
    )
    config = _config()
    config = replace(
        config,
        game=replace(
            config.game,
            population_size=32,
            options={**dict(config.game.options), "social_group_size": 4},
        ),
        control=replace(
            config.control,
            options={**dict(config.control.options), "sensor_sample_size": 8},
        ),
        analysis=replace(
            config.analysis,
            options={**dict(config.analysis.options), "per_cell_reports": True},
        ),
    )

    summary = run_configured_cell_analysis(config, tmp_path, "cell-0007")

    suffix = "cell-0007__N-32__q-4__qc-8"
    reports = tmp_path / "reports"
    assert summary is not None
    assert summary["cell_report_slug"] == suffix
    assert (reports / f"information_estimates__{suffix}.md").read_text() == "# MI\n"
    assert (reports / f"truth_current_estimates__{suffix}.md").read_text() == "# Truth current\n"
    assert set(path.name for path in reports.iterdir()) == {
        f"information_estimates__{suffix}.md",
        f"truth_current_estimates__{suffix}.md",
    }
    assert not list(tmp_path.glob(".cell-analysis-*"))
    assert captured["comet_export"] is False


def test_per_cell_reports_option_is_boolean():
    config = _config()
    enabled = replace(
        config,
        analysis=replace(
            config.analysis,
            options={**dict(config.analysis.options), "per_cell_reports": True},
        ),
    )
    assert per_cell_reports_enabled(enabled) is True
    invalid = replace(
        config,
        analysis=replace(
            config.analysis,
            options={**dict(config.analysis.options), "per_cell_reports": "yes"},
        ),
    )
    with pytest.raises(ValueError, match="per_cell_reports must be a boolean"):
        run_configured_analysis(invalid, ".")
