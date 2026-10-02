"""Offline pool comparison for task_003; does not alter task or study artifacts.

Run from the repository: python results/preanalysis_task003_new_setup_and_coarsegraining/10_task_and_facts/audit_target_pools.py
Writes JSON to stdout. Counts of four-fact proofs use the supplied proof inventory;
whole-pool posteriors independently test proof availability at any size.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.latent_problem import problem_from_latent_values
from mas_cc.musr_team_allocation_generator.symbolic_facts import true_canonical_facts


def audit(data_dir, existing_audit):
    rows = list(csv.DictReader((data_dir / "task_003_facts.csv").open()))
    proofs = json.loads((data_dir / "task_003_proofs.json").read_text())
    facts = {f.fact_id: f for f in true_canonical_facts(
        problem_from_latent_values((3, 1, 1, 2, 2, 1, 1, 1, 2)))}
    assert set(facts) == {r["fact"] for r in rows}
    index = TeamAllocationCompletionIndex()
    prior = index.metrics_for_facts(()).probabilities
    posterior = {f: index.metrics_for_facts((v,)).probabilities for f, v in facts.items()}
    for row in rows:
        for target in range(3):
            assert abs(float(row[f"dP_A{target}"]) -
                       (posterior[row["fact"]][target] - prior[target])) < 1e-12
    private = {r["fact"] for r in rows if r["allocation"] in ("agent only", "both")}
    frozen = {r["fact"] for r in rows if r["allocation"] in ("pool only", "both")}
    eligible = {r["fact"] for r in rows if r["eligible"].lower() == "true"}
    positive = {t: {f for f, p in posterior.items() if p[t] - prior[t] > 1e-12} for t in (0, 2)}
    pools = {"historical_false_pool": frozen,
             "existing_truth_v1": set(json.loads(existing_audit.read_text())["ordered_pool_ids"]),
             "private_union": private}
    for target in (0, 2):
        pools[f"positive_A{target}"] = positive[target]
        pools[f"weak_positive_A{target}"] = positive[target] & eligible
        pools[f"contrastive_A{target}"] = {
            f for f in positive[target] if posterior[f][target] > posterior[f][2-target] + 1e-12}
    output = {"world": [3, 1, 1, 2, 2, 1, 1, 1, 2], "prior": prior,
              "positive_pool_overlap": sorted(positive[0] & positive[2]), "pools": {}}
    for name, ids in pools.items():
        ordered = sorted(ids)
        metrics = index.metrics_for_facts(tuple(facts[f] for f in ordered))
        output["pools"][name] = {
            "fact_ids": ordered, "count": len(ids),
            "sorted_ids_sha256": hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest(),
            "private_overlap": len(ids & private), "non_private_eligible": len(ids - eligible),
            "posterior": metrics.probabilities,
            "valid_completions": metrics.valid_completion_count,
            "proves_truth": metrics.probabilities[0] == 1.0,
            "contained_min4_proofs": sum(set(p) <= ids for p in proofs["min4"]),
            "mean_individual_lift": [sum(posterior[f][t]-prior[t] for f in ordered)/len(ids) for t in range(3)],
        }
    return output


if __name__ == "__main__":
    data_dir = ROOT / "results/game_analysis"
    existing = ROOT / ("configs/runs/relational_reasoning/blackboard_game/iclr_experiments/"
                       "report_only_authored_q3_q12_chatoss_v5_target_aligned/truth_pool_audit.json")
    print(json.dumps(audit(data_dir, existing), indent=2))
