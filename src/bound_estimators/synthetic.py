"""Known-law synthetic benchmark with exact KL / MI by path enumeration.

A small population of ``n_agents`` agents votes for three allocations.  Each round every
agent independently redraws its vote from a softmax over ``kappa * log(fraction + eps)``
plus a law-specific ``pull`` vector, so the count vector is a time-inhomogeneous Markov
chain on the (n_agents+2 choose 2) count states.  Three laws share the initial
distribution: silent baseline, target 0, target 2.  Exact path probabilities over
horizon ``H`` are enumerated, giving exact KL(p_z || p_base), I(Z; Gamma), I(Z; Y_h)
and KL(Q || p_base) to compare against the estimators under the same parent / triplet
structure as the real archive.

Scenarios (per the specification):
  identical      controlled == silent                       -> everything zero
  generic        both targets share one perturbation        -> I = 0, C > 0
  opposite       targets pull in opposite directions        -> tight-ish bound
  transient      target pull at round 1 only, then a memoryless reset at the last round
                 -> I(Z; Gamma) > 0 but I(Z; Y_H) = 0
  rare_tail      strong pull toward a state the baseline almost never visits
  reversed       target 0 pulls toward allocation 2 and vice versa (direction check)
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .crossfit import Candidate
from .data import Comparison
from .folds import derive_seed, make_fold_plan
from .models import TrainConfig
from .objectives import endpoint_support, exact_discrete_kl, exact_discrete_mi
from .pipelines import run_problem
from .uncertainty import paired_bootstrap


@dataclass
class Law:
    """pull[t] is the (3,) additive logit pull applied at round t (1-based)."""
    pulls: list[np.ndarray]
    reset_last: bool = False   # last round ignores the current state (memoryless reset)


@dataclass
class Scenario:
    name: str
    base: Law
    targets: dict[int, Law]
    n_agents: int = 8
    horizon: int = 3
    kappa: float = 1.0
    p_init: tuple[float, float, float] = (0.45, 0.30, 0.25)
    eps: float = 0.05


def _states(n: int) -> np.ndarray:
    return np.array(endpoint_support(n), dtype=np.int64)


def _log_multinomial(states: np.ndarray, log_p: np.ndarray) -> np.ndarray:
    """log P(state | per-agent categorical log_p) for every count state (multinomial)."""
    n = states[0].sum()
    return gammaln(n + 1) - gammaln(states + 1).sum(axis=1) + states @ log_p


def _kernel(states: np.ndarray, pull: np.ndarray, kappa: float, eps: float, reset: bool) -> np.ndarray:
    """log K[y, y'] for the multinomial-imitation step."""
    n = states[0].sum()
    frac = states / n
    logits = kappa * np.log(frac + eps) + pull[None, :]
    if reset:
        logits = np.broadcast_to(pull[None, :], logits.shape)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.stack([_log_multinomial(states, lp) for lp in logp])  # (S, S)


def exact_path_logprob(sc: Scenario, law: Law) -> np.ndarray:
    """Tensor of log path probabilities with one axis per round (Y_0 ... Y_H)."""
    states = _states(sc.n_agents)
    log_init = _log_multinomial(states, np.log(np.asarray(sc.p_init)))
    logp = log_init
    for t in range(1, sc.horizon + 1):
        K = _kernel(states, law.pulls[t - 1], sc.kappa, sc.eps, law.reset_last and t == sc.horizon)
        logp = logp[..., None] + K  # broadcast previous axis against new axis
    return logp


def exact_quantities(sc: Scenario) -> dict[str, float]:
    lp_base = exact_path_logprob(sc, sc.base)
    lp = {z: exact_path_logprob(sc, sc.targets[z]) for z in (0, 2)}
    p_base, p = np.exp(lp_base).ravel(), {z: np.exp(lp[z]).ravel() for z in (0, 2)}
    q = 0.5 * p[0] + 0.5 * p[2]
    out = {"kl0": exact_discrete_kl(p[0], p_base), "kl2": exact_discrete_kl(p[2], p_base)}
    out["cost"] = 0.5 * out["kl0"] + 0.5 * out["kl2"]
    out["info_path"] = exact_discrete_mi({0: p[0], 2: p[2]})
    out["kl_mixture"] = exact_discrete_kl(q, p_base)
    for h in range(1, sc.horizon + 1):
        axes = tuple(i for i in range(sc.horizon + 1) if i != h)
        marg = {z: np.exp(lp[z]).sum(axis=axes) for z in (0, 2)}
        out[f"info_end_h{h}"] = exact_discrete_mi(marg)
    return out


def sample_triplets(sc: Scenario, m: int, seed: int) -> Comparison:
    """m parents: shared Y_0, then independent silent / target-0 / target-2 continuations."""
    rng = np.random.default_rng(seed)
    states = _states(sc.n_agents)
    p_init = np.exp(_log_multinomial(states, np.log(np.asarray(sc.p_init))))
    y0 = rng.choice(len(states), size=m, p=p_init / p_init.sum())
    kernels = {}
    for name, law in (("silent", sc.base), (0, sc.targets[0]), (2, sc.targets[2])):
        kernels[name] = [np.exp(_kernel(states, law.pulls[t - 1], sc.kappa, sc.eps,
                                        law.reset_last and t == sc.horizon)) for t in range(1, sc.horizon + 1)]
    paths = {}
    for name, ks in kernels.items():
        arr = np.zeros((m, sc.horizon + 1, 3), dtype=np.int64)
        cur = y0.copy()
        arr[:, 0] = states[cur]
        for t, K in enumerate(ks, start=1):
            nxt = np.array([rng.choice(len(states), p=K[c] / K[c].sum()) for c in cur])
            arr[:, t] = states[nxt]
            cur = nxt
        paths[name] = arr
    return Comparison(q=0, rho=0.0, schedule="synthetic", budget=0, parent_ids=[f"p{i}" for i in range(m)],
                      silent=paths["silent"], controlled={0: paths[0], 2: paths[2]}, horizon=sc.horizon)


def scenarios() -> list[Scenario]:
    z = np.zeros(3)
    e0, e2 = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    H = 3

    def law(pull_seq, reset=False):
        return Law(pulls=[np.asarray(p, dtype=float) for p in pull_seq], reset_last=reset)

    generic = 0.8 * np.array([0.0, 1.0, -1.0])
    return [
        Scenario("identical", law([z] * H), {0: law([z] * H), 2: law([z] * H)}),
        Scenario("generic", law([z] * H), {0: law([generic] * H), 2: law([generic] * H)}),
        Scenario("opposite", law([z] * H), {0: law([0.8 * e0] * H), 2: law([0.8 * e2] * H)}),
        Scenario("transient", law([z] * H, reset=True),
                 {0: law([1.2 * e0, z, z], reset=True), 2: law([1.2 * e2, z, z], reset=True)}),
        Scenario("rare_tail", law([z] * H), {0: law([2.5 * e0] * H), 2: law([2.5 * e2] * H)}),
        Scenario("reversed", law([z] * H), {0: law([0.8 * e2] * H), 2: law([0.8 * e0] * H)}),
    ]


ENDPOINT_CANDS = [Candidate("constant"), Candidate("linear", 0.1), Candidate("linear", 1.0),
                  Candidate("quadratic", 0.1), Candidate("quadratic", 1.0), Candidate("mlp", 0.01, 4), Candidate("mlp", 0.1, 4)]
PATH_CANDS = [Candidate("constant"), Candidate("linear", 0.1), Candidate("linear", 1.0),
              Candidate("quadratic", 0.1), Candidate("quadratic", 1.0),
              Candidate("flat", 0.01, 4), Candidate("flat", 0.1, 4), Candidate("gru", 0.01, 4), Candidate("gru", 0.1, 4)]


def _run_one(args: tuple[str, int, int, int, dict[str, Any]]) -> dict[str, Any]:
    name, m, rep, master_seed, train = args
    sc = next(s for s in scenarios() if s.name == name)
    exact = exact_quantities(sc)
    comp = sample_triplets(sc, m, derive_seed(master_seed, "synthetic", name, m, rep))
    plan = make_fold_plan(comp.parent_ids, master_seed, repeat=rep, tag=f"syn_{name}_{m}")
    tcfg = TrainConfig(**train)
    H = sc.horizon
    out: dict[str, Any] = {"scenario": name, "m": m, "replicate": rep, **{f"exact_{k}": v for k, v in exact.items()}}
    e = run_problem(comp, "endpoint", H, ENDPOINT_CANDS, plan, tcfg)
    out["est_info_end"] = e["selected"]["score"]; out["info_model"] = e["selection"]["selected_by_fold"]
    out["est_info_end_freq_alpha0"] = e["frequency"]["insample_mi_alpha0"]
    out["est_info_end_freq_heldout_alpha1"] = e["frequency"]["heldout_score_alpha1"]
    pp_info = e["selected"]["per_parent_eps1e-06"]
    costs = {}
    for z in (0, 2):
        c = run_problem(comp, f"cost{z}", H, PATH_CANDS, plan, tcfg)
        out[f"est_kl{z}_nwj"] = c["selected"]["nwj_raw"]; out[f"est_kl{z}_nwj_cap5"] = c["selected"]["nwj_cap5"]
        out[f"est_kl{z}_dv"] = c["selected"]["dv_raw"]; out[f"est_kl{z}_plugin"] = c["selected"]["plugin_raw"]
        out[f"kl{z}_model"] = c["selection"]["selected_by_fold"]; out[f"kl{z}_ess"] = c["selected"]["tail_raw"]["ess"]
        costs[z] = c["selected"]["per_parent_nwj_raw"]
    pp_cost = 0.5 * (costs[0] + costs[2])
    out["est_cost_nwj"] = float(pp_cost.mean())
    pi = run_problem(comp, "pathinfo", H, PATH_CANDS, plan, tcfg)
    out["est_info_path"] = pi["selected"]["score"]
    bs = paired_bootstrap({"info": pp_info, "cost": pp_cost}, 1000, derive_seed(master_seed, "synboot", name, m, rep))
    out["info_lo"], out["info_hi"] = bs["info"]["lo"], bs["info"]["hi"]
    out["cost_lo"], out["cost_hi"] = bs["cost"]["lo"], bs["cost"]["hi"]
    out["info_covered"] = bool(bs["info"]["lo"] <= exact[f"info_end_h{H}"] <= bs["info"]["hi"])
    out["cost_covered"] = bool(bs["cost"]["lo"] <= exact["cost"] <= bs["cost"]["hi"])
    if bs["eta"]["supported"]:
        out["est_eta"] = bs["eta"]["point"]; out["eta_lo"], out["eta_hi"] = bs["eta"]["lo"], bs["eta"]["hi"]
    out["exact_eta"] = exact[f"info_end_h{H}"] / exact["cost"] if exact["cost"] > 0 else float("nan")
    # direction: paired gain toward each target relative to silence at the endpoint
    for z in (0, 2):
        out[f"gain_toward_target{z}"] = float((comp.controlled[z][:, H, z] - comp.silent[:, H, z]).mean() / sc.n_agents)
    return out


def run_benchmark(out_dir: Path, sizes=(40, 200), replicates=2, master_seed=20260921,
                  workers: int = 4, train: dict[str, Any] | None = None) -> pd.DataFrame:
    train = train or {"lr": 1e-3, "max_epochs": 1000, "seeds": (0, 1, 2)}
    tasks = [(s.name, m, r, master_seed, train) for s in scenarios() for m in sizes for r in range(replicates)]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "synthetic_results.json"
    done = json.loads(cache.read_text()) if cache.exists() else []
    have = {(d["scenario"], d["m"], d["replicate"]) for d in done}
    todo = [t for t in tasks if (t[0], t[1], t[2]) not in have]
    if todo:
        with mp.get_context("spawn").Pool(workers) as pool:
            for res in pool.imap_unordered(_run_one, todo):
                done.append(res)
                cache.write_text(json.dumps(done, indent=1, default=str))
                print(f"synthetic {res['scenario']} m={res['m']} r={res['replicate']} done", flush=True)
    df = pd.DataFrame(done)
    df.to_csv(out_dir / "synthetic_results.csv", index=False)
    exact = pd.DataFrame([{"scenario": s.name, **exact_quantities(s)} for s in scenarios()])
    exact.to_csv(out_dir / "synthetic_exact.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/bound_estimators/synthetic"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replicates", type=int, default=2)
    a = ap.parse_args()
    run_benchmark(a.out, replicates=a.replicates, workers=a.workers)
