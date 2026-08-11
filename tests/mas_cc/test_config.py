from pathlib import Path

import pytest
import yaml

from mas_cc.config import (
    config_schema,
    load_component_config,
    load_run_config,
    parse_run_config,
    resolved_config_yaml,
    validate_run_config,
)
from mas_cc.llm_runtime.exceptions import ConfigurationError


def _resolved_mapping():
    return {
        "schema_version": 1,
        "llm_provider": {
            "schema_version": 1,
            "type": "mock",
            "model": "deterministic-v1",
        },
        "prompt": {
            "schema_version": 1,
            "prompt_family": "binary",
            "prompt_version": 1,
            "blocks": ["task", "output_contract"],
            "response_contract": {"type": "choice", "allowed_values": ["A", "B"]},
        },
        "game": {
            "schema_version": 1,
            "type": "toy_coordination",
            "population_size": 2,
            "horizon": 2,
        },
    }


def test_parse_applies_all_section_defaults_and_is_immutable():
    config = parse_run_config(_resolved_mapping())
    assert config.execution.seed == 0
    assert config.logging.level == "INFO"
    assert config.storage.output_dir == "results"
    assert config.analysis.estimators == ()
    assert config.analysis.comet_export is False
    assert config.experiment.name == "unnamed-experiment"
    with pytest.raises(TypeError):
        config.llm_provider.options["x"] = 1


def test_component_resolution_override_and_environment_default(tmp_path: Path):
    components = tmp_path / "components"
    runs = tmp_path / "runs"
    components.mkdir()
    runs.mkdir()
    provider = components / "provider.yaml"
    provider.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "type": "university",
                "model": "model-a",
                "credentials_env": "POTSDAM_API_KEY",
                "request_concurrency": 10,
            }
        ),
        encoding="utf-8",
    )
    values = _resolved_mapping()
    values["llm_provider"] = {
        "component": "../components/provider.yaml",
        "overrides": {"request_concurrency": 3},
    }
    values["storage"] = {"output_dir": "${OUTPUT_ROOT:-fallback}"}
    run = runs / "run.yaml"
    run.write_text(yaml.safe_dump(values), encoding="utf-8")

    config = load_run_config(run, environment={"OUTPUT_ROOT": "artifacts"})
    assert config.llm_provider.request_concurrency == 3
    assert config.llm_provider.credentials_env == "POTSDAM_API_KEY"
    assert config.storage.output_dir == "artifacts"


def test_whole_environment_references_retain_scalar_types(tmp_path: Path):
    values = _resolved_mapping()
    values["execution"] = {
        "parallelism": "${WORKERS}",
        "fail_fast": "${FAIL_FAST:-false}",
    }
    path = tmp_path / "typed-env.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    config = load_run_config(path, environment={"WORKERS": "4"})
    assert config.execution.parallelism == 4
    assert config.execution.fail_fast is False


def test_reusable_component_loads_and_validates_independently():
    component = load_component_config(
        "configs/components/llm_providers/university.yaml",
        "llm_provider",
        environment={},
    )
    assert component.type == "university"
    assert component.credentials_env == "POTSDAM_API_KEY"


def test_repo_smoke_config_resolves_deterministically_and_without_secret_values():
    path = Path("configs/runs/old/provider_smoke_test.yaml")
    first = load_run_config(path, environment={})
    second = load_run_config(path, environment={})
    rendered = resolved_config_yaml(first)
    assert first == second
    assert rendered == resolved_config_yaml(second)
    assert "credentials_env: POTSDAM_API_KEY" in rendered
    assert "request_concurrency: 2" in rendered
    assert "replace-with-your-key" not in rendered


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda raw: raw["game"].update(population_size=1), "game.population_size"),
        (lambda raw: raw["execution"].update(parallelism=0), "execution.parallelism"),
        (lambda raw: raw["llm_provider"].update(max_retires=2), "llm_provider.max_retires"),
        (lambda raw: raw.update(schema_version=99), "schema_version"),
    ],
)
def test_validation_errors_name_the_exact_invalid_field(mutate, field):
    raw = _resolved_mapping()
    raw["execution"] = {}
    mutate(raw)
    with pytest.raises(ConfigurationError) as captured:
        parse_run_config(raw)
    assert any(issue.field == field for issue in captured.value.issues)
    result = validate_run_config(raw)
    assert not result.is_valid


def test_inline_and_expanded_secrets_are_rejected(tmp_path: Path):
    inline = _resolved_mapping()
    inline["llm_provider"]["api_key"] = "<redacted>"
    with pytest.raises(ConfigurationError) as captured:
        parse_run_config(inline)
    assert any(issue.field == "llm_provider.api_key" for issue in captured.value.issues)

    expanded = _resolved_mapping()
    expanded["storage"] = {"output_dir": "${OPENAI_API_KEY}"}
    path = tmp_path / "secret.yaml"
    path.write_text(yaml.safe_dump(expanded), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot be expanded"):
        load_run_config(path, environment={"OPENAI_API_KEY": "not-exported"})


def test_credentials_env_must_be_a_variable_name():
    raw = _resolved_mapping()
    raw["llm_provider"]["credentials_env"] = "not a variable name"
    with pytest.raises(ConfigurationError) as captured:
        parse_run_config(raw)
    assert any(issue.field == "llm_provider.credentials_env" for issue in captured.value.issues)


def test_schema_describes_resolved_version_one():
    schema = config_schema()
    assert schema["properties"]["schema_version"] == {"const": 1}
    assert set(schema["required"]) >= {"llm_provider", "prompt", "game"}
    assert schema["properties"]["llm_provider"]["additionalProperties"] is False


def test_aggregation_defaults_to_the_correctness_rules_not_to_off():
    """The section is optional, but its defaults are the unbiased ones.

    A config that says nothing about aggregation must still forward-fill,
    relabel, and band by percentile — the biased variants have to be asked for
    explicitly, never fallen into.
    """

    config = parse_run_config(_resolved_mapping())

    assert config.aggregation.forward_fill == "absorbing"
    assert config.aggregation.relabel_by_winner is True
    assert config.aggregation.percentiles == (10, 50, 90)
    assert config.observability.comet.writer == "master_only"


def test_an_unknown_aggregation_metric_fails_at_load_not_at_the_first_cell():
    """A typo in a day-long Slurm job must not surface six hours in."""

    raw = _resolved_mapping()
    raw["aggregation"] = {"cell_metrics": ["dominant_action_share", "dominat_action_share"]}

    with pytest.raises(ConfigurationError) as captured:
        parse_run_config(raw)

    assert any(issue.field == "aggregation.cell_metrics" for issue in captured.value.issues)


@pytest.mark.parametrize(
    ("section", "value", "field"),
    [
        ({"forward_fill": "carry"}, None, "aggregation.forward_fill"),
        ({"percentiles": [10, 150]}, None, "aggregation.percentiles[1]"),
        ({"rolling_window": 0}, None, "aggregation.rolling_window"),
        ({"sweep_metrics": ["terminal_mutual_information"]}, None, "aggregation.sweep_metrics"),
    ],
)
def test_aggregation_validation_names_the_exact_invalid_field(section, value, field):
    raw = _resolved_mapping()
    raw["aggregation"] = section

    with pytest.raises(ConfigurationError) as captured:
        parse_run_config(raw)

    assert any(issue.field == field for issue in captured.value.issues)


def test_only_the_master_may_be_configured_as_a_comet_writer():
    """One writer per experiment key is what keeps the step counters unraced."""

    raw = _resolved_mapping()
    raw["observability"] = {"comet": {"writer": "workers"}}

    with pytest.raises(ConfigurationError) as captured:
        parse_run_config(raw)

    assert any(issue.field == "observability.comet.writer" for issue in captured.value.issues)


def test_resolved_config_records_the_aggregation_and_observability_sections():
    """Both change what a reported number means, so a run must carry its own."""

    exported = resolved_config_yaml(parse_run_config(_resolved_mapping()))
    reloaded = yaml.safe_load(exported)

    assert reloaded["aggregation"]["forward_fill"] == "absorbing"
    assert reloaded["observability"]["comet"]["heartbeat_seconds"] == 60.0
    assert reloaded["observability"]["comet"]["metric_plots"] is False
