import json
import importlib.util
from pathlib import Path

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "relational_study05_state_matching.py"
_SPEC = importlib.util.spec_from_file_location("relational_study05_state_matching", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DEFAULT_CONFIGS = _MODULE.DEFAULT_CONFIGS
analyze = _MODULE.analyze
preflight = _MODULE.preflight


def test_study05_preflight_validates_the_complete_120_episode_design():
    report = preflight(DEFAULT_CONFIGS, None)

    assert report["status"] == "passed"
    assert report["total_cells"] == 12
    assert report["total_episodes"] == 120
    assert report["total_logical_provider_calls"] == 2880


def test_study05_analysis_writes_only_the_requested_effect_table_and_figure(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    episode_dir = run_dir / "data" / "episodes" / "episode"
    episode_dir.mkdir(parents=True)
    rows = []
    for n_z in (6, 9, 12):
        before = ["Z"] * n_z + ["T"] * (24 - n_z)
        for action, n_after in ((ADVOCATE_TARGET, n_z + 2), (NO_OP, n_z - 1)):
            rows.append(
                {
                    "record_type": "relational_imitation_round_feedback",
                    "round_index": 0,
                    "task_id": "task_0001",
                    "seed": 100 + n_z,
                    "controller_target": "Z",
                    "controller_action": action,
                    "population_state_before": before,
                    "population_state_after": ["Z"] * n_after + ["T"] * (24 - n_after),
                    "agent_ids": [f"agent_{index:03d}" for index in range(1, 25)],
                    "initial_knowledge_class_by_agent": [0, 1] * 12,
                }
            )
    (episode_dir / "round_trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output_dir = tmp_path / "analysis"
    table = analyze([run_dir], output_dir, bootstrap_resamples=10, confidence=0.95, seed=7)

    assert [row["n_Z_0"] for row in table] == [6, 9, 12]
    assert all(row["chi"] == 3 / 24 for row in table)
    assert (output_dir / "state_matching_effects.csv").is_file()
    assert (output_dir / "state_matching_chi.png").is_file()
