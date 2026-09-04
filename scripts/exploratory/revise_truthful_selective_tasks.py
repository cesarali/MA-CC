"""Create the equality-fixed, diversity-reranked frozen development task set."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from mas_cc.core import Seed
from mas_cc.musr_team_allocation_generator.ambiguity import (
    TeamAllocationCompletionIndex,
)
from mas_cc.musr_team_allocation_generator.latent_problem import (
    problem_from_latent_values,
)
from mas_cc.musr_team_allocation_generator.selective_design import (
    build_selective_design,
)
from mas_cc.probes.musr_truthful_selective.config import load_config
from mas_cc.probes.musr_truthful_selective.diversity import (
    apply_diversity_ranking,
    build_diversity_audit,
)
from mas_cc.probes.musr_truthful_selective.symbolic import write_design_artifacts

ROOT = Path("results/studies/musr_truthful_selective_task_calibration_01")
CONFIG = Path(
    "configs/runs/relational_reasoning/blackboard_game/task_calibration_truthful_selective_01/calibration.yaml"
)
REPLACEMENT = 237


def main() -> None:
    config = load_config(CONFIG)
    rows = list(
        csv.DictReader(
            (ROOT / "symbolic_scan/candidate_worlds.csv").open(encoding="utf-8")
        )
    )
    row = next(value for value in rows if int(value["candidate_id"]) == REPLACEMENT)
    if row["passed"] != "True" or (
        int(row["gold_index"]),
        int(row["false_target_index"]),
    ) != (1, 0):
        raise RuntimeError(
            "candidate 237 is not a frozen old-gate pass with task_002 balance"
        )
    old_task = ROOT / "tasks/task_002"
    archive = ROOT / "tasks/task_002_candidate_53_archived"
    if archive.exists():
        shutil.rmtree(archive)
    old_task.rename(archive)
    problem = problem_from_latent_values(
        tuple(int(value) for value in row["latent_values"].split("|"))
    )
    design = build_selective_design(
        problem,
        TeamAllocationCompletionIndex(),
        config.symbolic,
        seed=int(Seed(config.seed).derive(f"candidate:{REPLACEMENT}")),
        false_target_index=0,
    )
    write_design_artifacts(
        ROOT / "tasks",
        design,
        task_id="task_002",
        candidate_id=REPLACEMENT,
        seed=int(Seed(config.seed).derive("task:2:replacement:237")),
    )
    task_roots = [ROOT / f"tasks/task_{index:03d}" for index in (1, 2, 3)]
    before = json.loads(
        (ROOT / "analysis/controller_diversity_audit.json").read_text(encoding="utf-8")
    )
    reranked = [apply_diversity_ranking(path) for path in task_roots]
    after = [build_diversity_audit(path) for path in task_roots]
    manifest = {
        "schema_version": 1,
        "selection_basis": "existing symbolic-pass rows only; no new scan or behavioral outcomes",
        "replacement": {
            "task_id": "task_002",
            "old_candidate": 53,
            "new_candidate": 237,
            "reason": (
                "candidate 237 is the earliest prospectively ranked candidate with 12 "
                "positive-marginal nonredundant additions and 9/9 latent coverage; "
                "candidate 53 has 11 additions and 6/9 coverage"
            ),
        },
        "final_tasks": {"task_001": 42, "task_002": 237, "task_003": 130},
        "diagnostic_budget_grid": [3, 6, 9, 12],
        "before_diversity": before,
        "after_diversity": after,
        "reranked_profiles": reranked,
    }
    (ROOT / "analysis/task_revision_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ROOT / "analysis/controller_diversity_audit.json").write_text(
        json.dumps(after, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(ROOT / "analysis/task_revision_manifest.json")


if __name__ == "__main__":
    main()
