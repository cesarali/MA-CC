"""Run the exact simulator on the two v2 setups and print the trajectories."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "10_task_and_facts"))
from engine import World
from exact_game import ExactGame, GameRules, AgentPolicy, ControllerPolicy

# The setups these results were produced with (2 October single-pool designs). The keys
# are kept because saved results are named after them; the symmetric file is superseded.
SETUP_FILES = {
    "task003_symmetric_v2": HERE.parent / "50_next_experiment" / "archive" / "task003_symmetric_sharedpool_superseded.json",
    "task003_nosolution_v2": HERE.parent / "50_next_experiment" / "archive" / "task003_nosolution_ineligible_superseded.json",
}
WORLD = (3, 1, 1, 2, 2, 1, 1, 1, 2)


def load(setup: str):
    d = json.loads(SETUP_FILES[setup].read_text())
    return d["agents"]["agent_assignments"], d["controller_pool"]["fact_ids"]


def mean_rows(all_rows):
    n = len(all_rows)
    out = []
    for t in range(len(all_rows[0])):
        agg = {"round_index": t}
        for k in all_rows[0][0]:
            if k == "round_index":
                continue
            agg[k] = sum(rows[t][k] for rows in all_rows) / n
        out.append(agg)
    return out


def main(episodes: int = 20, rounds: int = 30):
    w = World(WORLD)
    rules_by_rho = {rho: GameRules(rounds=rounds, rho=rho) for rho in (0.75, 1.0)}
    ap = AgentPolicy(vote_rule="bayes", post_rule="vote_aligned")
    arms = {
        "silent": ControllerPolicy(target=None),
        "truth": ControllerPolicy(target=0, budget=3, gate="always", rule="lru"),
        "false": ControllerPolicy(target=2, budget=3, gate="always", rule="lru"),
    }
    t0 = time.perf_counter()
    for setup in ("task003_symmetric_v2", "task003_nosolution_v2"):
        assignment, pool = load(setup)
        print("=" * 78)
        print(setup)
        for rho, rules in rules_by_rho.items():
            for arm, cp in arms.items():
                g = ExactGame(w, assignment, pool, rules, ap, cp)
                rows = mean_rows([g.run_episode(s, setup) for s in range(episodes)])
                first, mid, last = rows[0], rows[rounds // 2], rows[-1]
                print(
                    f"  rho={rho:<4} {arm:<7} "
                    f"e: {first['e']:.3f} -> {mid['e']:.3f} -> {last['e']:.3f}   "
                    f"share_A0 {last['share_A0']:.3f}  share_A2 {last['share_A2']:.3f}  "
                    f"|K|={last['mean_nfacts']:.1f}  ctl_posts/rd={last['controller_posts']:.1f}"
                )
    dt = time.perf_counter() - t0
    cells = 2 * 2 * 3
    print("=" * 78)
    print(f"{cells} cells x {episodes} episodes x {rounds} rounds in {dt:.1f} s")
    print(f"  -> {dt/cells/episodes*1000:.1f} ms per episode")
    print(f"  -> 12 cells x 100 episodes would take ~{dt/episodes*100/2:.0f} s")


if __name__ == "__main__":
    main(episodes=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
