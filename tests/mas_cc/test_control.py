import tempfile
from pathlib import Path

import pytest

from mas_cc.config import ConfigLoader, ControlConfig, GridSpec, load_run_config_or_grid
from mas_cc.core import AgentId
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.control import ForcedActionControl, NoneControl, create_control


def test_none_mechanism_never_overrides():
    control = create_control(ControlConfig())
    assert isinstance(control, NoneControl)
    assert control.override(agent_id=AgentId("agent-000"), interaction_index=1, state=None) is None


def test_forced_action_overrides_only_configured_agents():
    control = create_control(
        ControlConfig(
            mechanism="forced_action",
            options={"agent_ids": ["agent-000", "agent-002"], "forced_value": "Q"},
        )
    )
    assert isinstance(control, ForcedActionControl)
    assert control.override(agent_id=AgentId("agent-000"), interaction_index=1, state=None) == "Q"
    assert control.override(agent_id=AgentId("agent-002"), interaction_index=99, state=None) == "Q"
    assert control.override(agent_id=AgentId("agent-001"), interaction_index=1, state=None) is None


def test_forced_action_respects_pulse_cutoff():
    control = create_control(
        ControlConfig(
            mechanism="forced_action",
            options={"agent_ids": ["agent-000"], "forced_value": "Q", "until_interaction": 3},
        )
    )
    assert control.override(agent_id=AgentId("agent-000"), interaction_index=3, state=None) == "Q"
    assert control.override(agent_id=AgentId("agent-000"), interaction_index=4, state=None) is None


def test_forced_action_requires_agent_ids_and_forced_value():
    with pytest.raises(ConfigurationError):
        create_control(ControlConfig(mechanism="forced_action", options={"forced_value": "Q"}))
    with pytest.raises(ConfigurationError):
        create_control(ControlConfig(mechanism="forced_action", options={"agent_ids": ["agent-000"]}))


def test_unknown_mechanism_raises():
    with pytest.raises(ConfigurationError):
        create_control(ControlConfig(mechanism="does-not-exist"))


def _write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


_BASE_YAML = """
llm_provider:
  type: mock
  model: mock-model
prompt:
  prompt_family: naming_convention_decision
  prompt_version: 1
  schema_version: 2
  response_contract:
    type: json_reason
    allowed_values: [Q, M]
game:
  type: naming_convention
  population_size: 4
  horizon: 6
"""


def test_control_config_round_trips_through_the_loader():
    path = _write(
        _BASE_YAML
        + """
control:
  mechanism: forced_action
  options:
    agent_ids: [agent-000]
    forced_value: Q
"""
    )
    try:
        config = ConfigLoader(environment={}).load(path)
        assert config.control.mechanism == "forced_action"
        assert config.control.options["forced_value"] == "Q"
        assert config.to_dict()["control"]["mechanism"] == "forced_action"
    finally:
        path.unlink()


def test_control_defaults_to_none_mechanism_when_omitted():
    path = _write(_BASE_YAML)
    try:
        config = ConfigLoader(environment={}).load(path)
        assert config.control.mechanism == "none"
        assert config.control.options == {}
    finally:
        path.unlink()


def test_grid_can_sweep_control_options_with_no_grid_changes():
    path = _write(
        _BASE_YAML
        + """
control:
  mechanism: forced_action
  options:
    agent_ids: [agent-000]
    forced_value: Q
grid:
  control.options.forced_value: [Q, M]
  control.mechanism: [forced_action, none]
"""
    )
    try:
        source = load_run_config_or_grid(path, environment={})
        assert isinstance(source, GridSpec)
        cells = source.cells
        assert len(cells) == 4
        # Each cell's overrides.json-equivalent dict already carries the condition label.
        labels = {cell.cell_id: dict(cell.overrides) for cell in cells}
        assert labels["cell-0000"] == {
            "control.options.forced_value": "Q",
            "control.mechanism": "forced_action",
        }
        mechanisms = {cell.config.control.mechanism for cell in cells}
        assert mechanisms == {"forced_action", "none"}
    finally:
        path.unlink()
