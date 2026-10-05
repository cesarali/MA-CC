"""Full sweep over both setups, arms, persistences and controller rules.

Writes per-round trajectories to results/trajectories.parquet and the
coarse-grained analysis to results/.
"""
from __future__ import annotations
import json, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "10_task_and_facts"))
from engine import World
from exact_game import ExactGame, GameRules, AgentPolicy, ControllerPolicy

# The setups these results were produced with (2 October single-pool designs). The keys
# are kept because saved results are named after them; the symmetric file is superseded.
SETUP_FILES = {
    "task003_symmetric_v2": HERE.parent / "50_next_experiment" / "archive" / "task003_symmetric_sharedpool_superseded.json",
    "task003_nosolution_v2": HERE.parent / "50_next_experiment" / "designs" / "task003_nosolution.json",
}
OUT = HERE / "results"
WORLD = (3, 1, 1, 2, 2, 1, 1, 1, 2)
SETUPS = ("task003_symmetric_v2", "task003_nosolution_v2")
RHOS = (0.75, 1.0)
ARMS = {"silent": None, "truth": 0, "false": 2}
RULES = ("lru", "greedy_posterior")
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ROUNDS = 30


def main() -> None:
    OUT.mkdir(exist_ok=True)
    w = World(WORLD)
    ap = AgentPolicy(vote_rule="bayes", post_rule="vote_aligned_novel")
    rows = []
    t0 = time.perf_counter()
    for setup in SETUPS:
        d = json.loads(SETUP_FILES[setup].read_text())
        asg = d["agents"]["agent_assignments"]
        pool = d["controller_pool"]["fact_ids"]
        for rho in RHOS:
            rules = GameRules(rounds=ROUNDS, rho=rho)
            for arm, target in ARMS.items():
                # the silent arm has no controller, so the rule is irrelevant:
                # run it once and label it for both.
                rule_set = ("lru",) if target is None else RULES
                for rule in rule_set:
                    cp = ControllerPolicy(target=target, budget=3,
                                          gate="always", rule=rule)
                    g = ExactGame(w, asg, pool, rules, ap, cp)
                    for ep in range(EPISODES):
                        for r in g.run_episode(ep, setup):
                            rows.append({"setup": setup, "rho": rho, "arm": arm,
                                         "rule": rule, "episode": ep, **r})
                    print(f"  {setup:<22} rho={rho:<4} {arm:<7} {rule:<17} "
                          f"done ({time.perf_counter()-t0:5.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    # the silent arm is rule-independent: duplicate it under the oracle label
    sil = df[(df.arm == "silent")].copy()
    sil["rule"] = "greedy_posterior"
    df = pd.concat([df, sil], ignore_index=True)
    df.to_parquet(OUT / "trajectories.parquet")
    print(f"\n{len(df):,} rows in {time.perf_counter()-t0:.1f}s -> {OUT/'trajectories.parquet'}")
    print(f"cells: {df.groupby(['setup','rho','arm','rule']).ngroups}, "
          f"episodes/cell: {EPISODES}, rounds: {ROUNDS}")


if __name__ == "__main__":
    main()
