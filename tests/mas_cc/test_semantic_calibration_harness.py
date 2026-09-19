"""Label sheet + agreement scoring for the semantic attribution: shapes, kappa, threshold sweep."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mas_cc.analysis.semantic_attribution import PRESSURE_LEVELS

SCRIPTS = Path(__file__).parents[2] / "scripts" / "Cygnus" / "analysis"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sheet_module = _load("semantic_label_sheet")
agreement_module = _load("semantic_agreement")


def _judged(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = {
            "source_path": "x", "cell_id": f"cell-{i % 3}", "episode_id": f"ep-{i // 5}", "round_index": i % 5,
            "within_round_index": 0, "focal_agent_id": f"agent_{i:03d}", "message_id": f"m{i}", "message_type": "REPORT",
            "reply_to": None, "text": f"message {i}", "vote_before": "A", "vote_after": "B", "controller_target": "B",
            "correct_answer": "A", "controller_message_exposed": bool(i % 2), "intervention_budget": 6,
            "controller_action": "ADVOCATE_Z", "possible_answers_json": json.dumps(["A", "B", "C"]),
            "request_id": f"rid{i}", "question_version": "q1", "estimator_version": "v",
        }
        cites = float(rng.uniform(0.6, 1.0)) if i % 2 else float(rng.uniform(0.0, 0.4))
        stance = ["A", "B", "none"][i % 3]
        level = i % 4
        probabilities = {lvl: (0.7 if k == level else 0.1) for k, lvl in enumerate(PRESSURE_LEVELS)}
        rows += [
            {**base, "question": "cites_controller", "answer_type": "noul", "value": cites, "label": None, "confidence": None, "probabilities_json": None},
            {**base, "question": "stance", "answer_type": "choice", "value": None, "label": stance, "confidence": 0.9, "probabilities_json": "{}"},
            {**base, "question": "pressure_toward_target", "answer_type": "score", "value": level / 3, "label": None, "confidence": 0.8, "probabilities_json": json.dumps(probabilities)},
            {**base, "question": "provides_evidence", "answer_type": "noul", "value": 0.9 if i % 3 == 0 else 0.2, "label": None, "confidence": None, "probabilities_json": None},
        ]
    return pd.DataFrame(rows)


def test_sheet_has_one_row_per_sampled_message_and_blank_human_columns():
    judged = _judged()
    sheet = sheet_module.build_sheet(judged, sample=12, seed=1)
    assert len(sheet) == 12 and sheet["message_id"].is_unique
    assert set(sheet_module.HUMAN_COLUMNS) <= set(sheet.columns)
    assert (sheet[list(sheet_module.HUMAN_COLUMNS)] == "").all().all()
    again = sheet_module.build_sheet(judged, sample=12, seed=1)
    assert sheet["message_id"].tolist() == again["message_id"].tolist(), "same seed, same sheet for every labeller"


def test_cohen_kappa_basics():
    assert agreement_module.cohen_kappa([True, False, True], [True, False, True]) == 1.0
    assert math.isnan(agreement_module.cohen_kappa([True, True], [True, True]))  # no variation, undefined
    assert abs(agreement_module.cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3], weights="linear", levels=[0, 1, 2, 3]) - 1.0) < 1e-12
    assert agreement_module.cohen_kappa([0, 3], [3, 0], weights="linear", levels=[0, 1, 2, 3]) < 0


def test_perfect_human_labels_score_perfectly_and_threshold_is_found():
    judged = _judged()
    sheet = sheet_module.build_sheet(judged, sample=None, seed=1)
    # Fill the sheet with the labels the synthetic model "meant".
    truth = {row["message_id"]: row for row in judged.drop_duplicates("message_id").to_dict("records")}
    sheet["human_cites_controller"] = [("yes" if int(m[1:]) % 2 else "no") for m in sheet["message_id"]]
    sheet["human_stance"] = [["A", "B", "none"][int(m[1:]) % 3] for m in sheet["message_id"]]
    sheet["human_pressure"] = [str(int(m[1:]) % 4) for m in sheet["message_id"]]
    sheet["human_provides_evidence"] = [("yes" if int(m[1:]) % 3 == 0 else "no") for m in sheet["message_id"]]
    sheet["labeller"] = "A"
    sheet["labeller_index"] = 0
    report, rows = agreement_module.agreement(judged, sheet.astype(str).assign(labeller_index=0))
    model = report["model_vs_human"]
    assert model["stance"]["accuracy"] == 1.0 and model["stance"]["kappa"] == 1.0
    assert model["pressure_toward_target"]["accuracy"] == 1.0 and model["pressure_toward_target"]["linear_weighted_kappa"] == 1.0
    assert model["cites_controller"]["best"]["kappa"] == 1.0 and 0.4 <= model["cites_controller"]["best_threshold"] <= 0.6
    assert model["provides_evidence"]["best"]["accuracy"] == 1.0
    assert len(rows) == len(sheet) and "human_vs_human" not in report
    assert truth  # the fixture is what the labels were derived from


def test_two_sheets_give_a_human_ceiling():
    judged = _judged()
    a = sheet_module.build_sheet(judged, sample=None, seed=1)
    for column in sheet_module.HUMAN_COLUMNS:
        a[column] = "no" if "stance" not in column and "pressure" not in column else ("none" if "stance" in column else "0")
    b = a.copy()
    b.loc[b.index[:10], "human_cites_controller"] = "yes"
    sheets = pd.concat([a.assign(labeller_index=0), b.assign(labeller_index=1)], ignore_index=True).astype({"labeller_index": int})
    report, _ = agreement_module.agreement(judged, sheets.astype({c: str for c in sheet_module.HUMAN_COLUMNS}))
    assert report["labellers"] == 2 and report["human_vs_human"]["n"] == len(a)
    assert report["human_vs_human"]["cites_controller_kappa"] < 1.0
    assert math.isnan(report["human_vs_human"]["stance_kappa"])  # both said `none` everywhere: undefined, not 1
