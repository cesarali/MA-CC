"""A control option no mechanism reads is a configuration error, not a silent default."""

import pytest

from mas_cc.config import load_run_config_or_grid
from mas_cc.control.forced_action import ForcedActionControl
from mas_cc.games.hidden_bench.imitation.controller import SoftTargetControl, ThresholdTargetControl
from mas_cc.games.relational_reasoning.imitation_round_feedback.controller import (
    RelationalRoundBudgetedControl,
)
from mas_cc.llm_runtime.exceptions import ConfigurationError

# The v0.4 next-experiment template, so the check is pinned to the config the runs use.
RELATIONAL = (
    "analysis/preanalysis_task003_new_setup_and_coarsegraining/50_next_experiment/designs/"
    "config_template.yaml"
)


def _relational_options(**overrides):
    source = load_run_config_or_grid(RELATIONAL)
    cells = getattr(source, "cells", None)
    config = cells[0].config if cells else source
    return {**dict(config.control.options), **overrides}


def test_the_v04_template_has_no_unread_option():
    RelationalRoundBudgetedControl.from_options(_relational_options())


def test_an_unknown_key_fails_control_creation():
    with pytest.raises(ConfigurationError, match=r"control\.options\.zz_not_an_option"):
        RelationalRoundBudgetedControl.from_options(_relational_options(zz_not_an_option=1))


def test_a_misspelled_key_names_the_option_it_resembles():
    options = _relational_options()
    options["intervention_budgt"] = options.pop("intervention_budget")
    with pytest.raises(ConfigurationError, match=r"did you mean 'intervention_budget'"):
        RelationalRoundBudgetedControl.from_options(options)


def test_ignored_options_accepts_a_key_meant_for_another_mechanism():
    # A grid sharing one block across threshold_target and soft_target cells.
    shared = {"target": "correct", "threshold": 0.5, "beta": 4.0}
    with pytest.raises(ConfigurationError, match=r"control\.options\.beta"):
        ThresholdTargetControl.from_options(shared)
    ThresholdTargetControl.from_options({**shared, "ignored_options": ["beta"]})
    assert SoftTargetControl.from_options({**shared, "ignored_options": ["beta"]}).beta == 4.0


def test_ignored_options_must_be_a_list_of_names():
    with pytest.raises(ConfigurationError, match=r"control\.options\.ignored_options"):
        ThresholdTargetControl.from_options({"target": "correct", "ignored_options": "beta"})


def test_forced_action_rejects_an_unknown_key():
    options = {"agent_ids": ["agent_0"], "forced_value": "A"}
    ForcedActionControl.from_options(options)
    with pytest.raises(ConfigurationError, match=r"control\.options\.until_interation"):
        ForcedActionControl.from_options({**options, "until_interation": 3})
