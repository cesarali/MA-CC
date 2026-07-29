import json
from argparse import Namespace

from naming_game import cli
from naming_game.analysis import empowerment as empowerment_analysis


def _write_config(path, *, auto_analyze):
    path.write_text(
        "\n".join(
            [
                "population_size: 2",
                "max_population_rounds: 1",
                "committee_sizes: [0]",
                "regimes: [neutral]",
                "window_interactions: 2",
                "replications:",
                "  unit: per_stratum",
                "  count: 1",
                f"auto_analyze: {'true' if auto_analyze else 'false'}",
                "quick_bootstrap_resamples: 1",
                "quick_null_permutations: 1",
                "provider: openai",
                "model: mock/model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_automatic_analysis_failure_does_not_fail_completed_experiment(
    tmp_path, monkeypatch, caplog, capsys
):
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "run"
    _write_config(config, auto_analyze=True)

    def fail_analysis(*args, **kwargs):
        raise RuntimeError("plot backend unavailable")

    monkeypatch.setattr(empowerment_analysis, "make_experiment_summary", fail_analysis)
    caplog.set_level("ERROR")
    exit_code = cli.main(
        [
            "experiment",
            "--config",
            str(config),
            "--mock",
            "--output-dir",
            str(output),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["analysis"]["status"] == "failed"
    assert (output / "interactions.parquet").exists()
    assert (output / "episodes.parquet").exists()
    assert (output / "analysis" / "empowerment_estimates.parquet").exists()
    assert (output / "analysis" / "summary.md").exists()
    assert "rerun naming-game analyze-empowerment later" in caplog.text


def test_experiment_analyze_flag_overrides_disabled_yaml(tmp_path, monkeypatch, capsys):
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "run"
    _write_config(config, auto_analyze=False)
    calls = []

    def fake_analysis(history_dir, output_dir, settings):
        calls.append((history_dir, output_dir, settings))
        return {"output_dir": str(output_dir), "warnings": 0}

    monkeypatch.setattr(cli, "analyze_histories", fake_analysis)
    exit_code = cli.main(
        [
            "experiment",
            "--config",
            str(config),
            "--mock",
            "--analyze",
            "--output-dir",
            str(output),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["analysis"]["status"] == "completed"
    assert calls[0][0] == output
    assert calls[0][1] == output / "analysis"


def test_standalone_analysis_defaults_inside_history_directory(tmp_path, monkeypatch):
    calls = []

    def fake_analysis(history_dir, output_dir, settings):
        calls.append((history_dir, output_dir))
        return {"output_dir": str(output_dir)}

    monkeypatch.setattr(cli, "analyze_histories", fake_analysis)
    args = Namespace(
        history_dir=tmp_path / "history",
        output_dir=None,
        horizons=[1],
        bootstrap_resamples=1,
        null_permutations=1,
        seed=1,
    )
    assert cli._analyze_empowerment(args) == 0
    assert calls == [(tmp_path / "history", tmp_path / "history" / "analysis")]
