from __future__ import annotations

import json

import pandas as pd
import pytest

from mas_cc.studies.weighted_summaries import weighted_rho_aliases


def _derived(q_values, *, duplicate=False):
    rows = []
    for q in q_values:
        for budget in (6, 18):
            rows.append({"social_group_size": q, "sensor_sample_size": float(q), "intervention_budget": budget,
                         "target_semantics": False, "metric": "round_target_actuation_cmi", "estimate": q + budget,
                         "marginalized_dimensions": json.dumps(["epistemic_persistence"]), "support_status": "supported"})
    if duplicate:
        rows.append(dict(rows[0]))
    return pd.DataFrame(rows)


def _legacy(q_values):
    return pd.DataFrame([{"social_group_size": q, "sensor_sample_size": float(q), "intervention_budget": budget,
                          "target_semantics": "false", "metric": "T_pi", "estimate": -1.0}
                         for q in q_values for budget in (6, 18)])


def test_two_group_sizes_are_distinct_report_rows():
    outputs = {"study_aggregated_metrics": _derived([3, 12]), "rho_aggregated_descriptive_summary": _legacy([3, 12])}
    weighted_rho_aliases(outputs)
    report = outputs["rho_aggregated_descriptive_summary"]
    assert len(report) == 4 and not report.duplicated(["social_group_size", "intervention_budget"]).any()
    assert sorted(report["estimate"]) == [9, 18, 21, 30]  # derived values replaced every legacy row


def test_single_group_size_matches_as_before():
    outputs = {"study_aggregated_metrics": _derived([12]), "rho_aggregated_descriptive_summary": _legacy([12])}
    weighted_rho_aliases(outputs)
    assert sorted(outputs["rho_aggregated_descriptive_summary"]["estimate"]) == [18, 30]


def test_a_genuine_duplicate_view_is_still_refused():
    outputs = {"study_aggregated_metrics": _derived([3, 12], duplicate=True),
               "rho_aggregated_descriptive_summary": _legacy([3, 12])}
    with pytest.raises(ValueError, match="multiple derived rho views"):
        weighted_rho_aliases(outputs)
