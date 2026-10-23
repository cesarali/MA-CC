"""Run the mean-field models over a set of conditions and store their predictions.

  --closure M1..M4       which model (theory/models/mean_field.py)
  --set dev|all|post_stop|check|k2|k4|op|op_post_stop
        dev        development cells: both setups, rho {0.25, 0.75, 1}, q {3, 6}, t <= 40
        all        every reading-run cell: both setups, 8 rho, q {3, 6, 9, 12}, t <= 40
        post_stop  both setups, rho {0.25, 0.75}, q 6, four post-stop modes, t <= 130
        k2, k4     K-projects worlds: both setups, rho {1, 0.75, 0.5, 0.25}, q {3, 6} (W, rate, budget x K)
        op         Object Placements world: both setups, 8 rho, q {3, 6, 9, 12}
        op_post_stop  Object Placements world: both setups, rho {0.25, 0.75}, q 6, four post-stop modes, t <= 130
        check      numerical convergence: symmetric and nosolution, rho 0.5, q 6, replicas 50/100/200, dt 0.05/0.025
  --tag NAME            output name (default: closure)
  --seeds S, --replicas R   realisations per condition (M3) and population replicas
Writes analysis/theory/results/theory_runs/<tag>_<set>.parquet: one row per (setup, rho, q, arm, mode, t) with vote
shares x0..x2 among voters, mean beliefs, controller messages, controller-only share of memory slots and board, and
per-fact m_f, b_f (columns m_<fact>, b_<fact>).

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory/models/run_models.py --closure M2 --set dev
"""
from __future__ import annotations
import argparse, multiprocessing, pathlib, sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mean_field import Model, Spec  # noqa: E402

ANALYSIS = HERE.parents[1]                                        # analysis/
OUT = HERE.parent / "results" / "theory_runs"
RHOS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
MODES = ["normal", "no_comm", "frozen_board", "purge_controller_facts"]


def specs(which, closure, seeds=1, replicas=None):
    out = []
    kw = dict(closure=closure, replicas=replicas)
    if which in ("dev", "all"):
        rhos, qs = ([0.25, 0.75, 1.0], [3, 6]) if which == "dev" else (RHOS, [3, 6, 9, 12])
        for s in ("symmetric", "nosolution"):
            for r in rhos:
                for q in qs:
                    for arm in ("silent", "false"):
                        for seed in range(1, seeds + 1):
                            out.append(Spec(s, arm, r, q, seed=seed, **kw))
    elif which == "post_stop":
        for s in ("symmetric", "nosolution"):
            for r in (0.25, 0.75):
                for mode in MODES:
                    for arm in ("silent", "false"):
                        for seed in range(1, seeds + 1):
                            out.append(Spec(s, arm, r, 6, t_end=130.0, post_stop_mode=mode, seed=seed, **kw))
    elif which in ("k2", "k4"):
        K = int(which[1])
        wf = str(ANALYSIS / "task003_llm_free" / "llmfree_core" / f"world_task003_K{K}.json")
        for base in ("symmetric", "nosolution"):
            for r in (1.0, 0.75, 0.5, 0.25):
                for q in (3, 6):
                    for arm in ("silent", "false"):
                        for seed in range(1, seeds + 1):
                            out.append(Spec(f"task003_{base}_K{K}", arm, r, q, seed=seed, world_file=wf, W=48 * K,
                                            lambda_c=float(K), budget=30.0 * K, **kw))
    elif which == "op":
        op = ANALYSIS / "object_placements_llm_free"
        wf = str(op / "world" / "world_op1.json")
        for base in ("op_symmetric", "op_nosolution"):
            for r in RHOS:
                for q in (3, 6, 9, 12):
                    for arm in ("silent", "false"):
                        for seed in range(1, seeds + 1):
                            out.append(Spec(base, arm, r, q, seed=seed, world_file=wf,
                                            setups_dir=str(op / "experimental_setup"), **kw))
    elif which == "op_post_stop":
        op = ANALYSIS / "object_placements_llm_free"
        wf, sd = str(op / "world" / "world_op1.json"), str(op / "experimental_setup")
        for base in ("op_symmetric", "op_nosolution"):
            for r in (0.25, 0.75):
                for mode in MODES:
                    for arm in ("silent", "false"):
                        for seed in range(1, seeds + 1):
                            out.append(Spec(base, arm, r, 6, t_end=130.0, post_stop_mode=mode, seed=seed,
                                            world_file=wf, setups_dir=sd, **kw))
    elif which == "check":
        for s in ("symmetric", "nosolution"):
            for arm in ("silent", "false"):
                for rep, dt in ((50, 0.05), (100, 0.05), (200, 0.05), (100, 0.025)):
                    out.append(Spec(s, arm, 0.5, 6, closure=closure, replicas=rep, dt=dt))
    return out


def run(spec):
    rows = []
    for o in Model(spec).run():
        m, b = o.pop("m"), o.pop("b")
        row = dict(setup=spec.setup.replace("task003_", "").split("_K")[0], rho=spec.rho, q=spec.q, arm=spec.arm, mode=spec.post_stop_mode, seed=spec.seed,
                   replicas=spec.replicas or 100, dt=spec.dt, **o)
        row.update({f"m_{j}": v for j, v in enumerate(m)})
        row.update({f"b_{j}": v for j, v in enumerate(b)})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closure", default="M2", choices=["M1", "M2", "M3", "M3s", "M4"])
    ap.add_argument("--seeds", type=int, default=1, help="realisations per condition (M3 is stochastic)")
    ap.add_argument("--replicas", type=int, help="population replicas (default 100)")
    ap.add_argument("--set", default="dev", choices=["dev", "all", "post_stop", "check", "k2", "k4", "op", "op_post_stop"])
    ap.add_argument("--tag")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = specs(a.set, a.closure, a.seeds, a.replicas)
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(run, sp, chunksize=1)
    keep = None
    df = pd.DataFrame([r for p in parts for r in p])
    if a.seeds > 1:                     # keep the ensemble small: drop per-fact columns except at t = 30 and 40
        per_fact = [c for c in df.columns if c[:2] in ("m_", "b_")]
        df.loc[~df.t.isin([30.0, 40.0, 130.0]), per_fact] = float("nan")
    path = OUT / f"{a.tag or a.closure}_{a.set}.parquet"
    df.to_parquet(path, index=False)
    print(len(sp), "runs ->", path)


if __name__ == "__main__":
    main()
