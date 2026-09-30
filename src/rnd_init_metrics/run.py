"""Orchestrator: archive -> observables -> fitted estimators -> tables -> report.

Stages
    data    validate the archive and write the audit
    obs     closed-form observables at every horizon (fast, no fitting)
    fit     cross-fitted information and KL estimators (parallel, cached)
    swaps   target-label permutation nulls
    tables  assemble CSV tables and efficiencies
    report  build the LaTeX report

Fitted jobs are cached one pickle per job under ``<output_dir>/jobs`` and skipped
on a re-run, so the pipeline is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bound_estimators.crossfit import Candidate
from bound_estimators.folds import FoldPlan, derive_seed, make_fold_plan
from bound_estimators.models import TrainConfig

from . import data as D
from . import estimators as E
from . import observables as OB

_COMPS: dict[str, D.Comparison] = {}


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_config_path"] = str(path)
    return cfg


def train_config(cfg: dict[str, Any]) -> TrainConfig:
    return TrainConfig(**cfg["training"])


def candidates(spec: list[dict[str, Any]]) -> list[Candidate]:
    out: list[Candidate] = []
    for s in spec:
        for lam in (s.get("lam", [0.0]) if isinstance(s.get("lam", [0.0]), list) else [s["lam"]]):
            for w in (s.get("width", [0]) if isinstance(s.get("width", [0]), list) else [s["width"]]):
                out.append(Candidate(family=s["family"], lam=float(lam), width=int(w),
                                     cap=s.get("cap")))
    return out


# --------------------------------------------------------------------- jobs
def _job_path(out_dir: Path, job: dict[str, Any]) -> Path:
    name = f"{job['comparison']}__{job['problem']}__h{job['horizon']}__{job['set']}"
    if job.get("kind") == "swap":
        name = "swap__" + name
    return out_dir / "jobs" / f"{name}.pkl"


def _plan(comp: D.Comparison, seed: int, folds: dict[str, int]) -> FoldPlan:
    return make_fold_plan(comp.inits, seed, repeat=0, n_outer=folds["outer"],
                          n_inner=folds["inner"], tag=comp.key)


def _init_worker(comps: dict[str, D.Comparison]) -> None:
    global _COMPS
    _COMPS = comps
    import torch
    torch.set_num_threads(1)


def _swap_targets(comp: D.Comparison, rng: np.random.Generator) -> D.Comparison:
    """Exchangeability null: swap the two target branches within an initialization."""
    flip = rng.random(comp.m) < 0.5
    c0, c2 = comp.controlled[0].copy(), comp.controlled[2].copy()
    c0[flip], c2[flip] = comp.controlled[2][flip], comp.controlled[0][flip]
    new = D.Comparison(**{**asdict(comp), "controlled": {0: c0, 2: c2}})
    return new


def run_job(job: dict[str, Any]) -> tuple[str, float]:
    t0 = time.time()
    path = Path(job["path"])
    if path.exists():
        return job["name"], 0.0
    comp = _COMPS[job["comparison"]]
    cands = [Candidate(**c) for c in job["candidates"]]
    cfg = TrainConfig(**job["train"])
    plan = _plan(comp, job["seed"], job["folds"])
    if job.get("kind") == "swap":
        rng = np.random.default_rng(derive_seed(job["seed"], "swap", comp.key, job["problem"], job["horizon"]))
        observed = E.run_problem(comp, job["problem"], job["horizon"], cands, plan, cfg)["selected"]["score_bits"]
        null = [E.run_problem(_swap_targets(comp, rng), job["problem"], job["horizon"], cands, plan, cfg)
                ["selected"]["score_bits"] for _ in range(job["n_swaps"])]
        result = {"kind": "swap", "comparison": comp.key, "problem": job["problem"],
                  "horizon": job["horizon"], "observed": observed, "null": np.array(null)}
    else:
        result = E.run_problem(comp, job["problem"], job["horizon"], cands, plan, cfg,
                               keep_curves=job.get("keep_curves", False))
        result["set"] = job["set"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    return job["name"], time.time() - t0


def build_jobs(cfg: dict[str, Any], comps: dict[str, D.Comparison], out_dir: Path) -> list[dict]:
    jobs: list[dict[str, Any]] = []
    base = {"seed": cfg["seed"], "folds": cfg["folds"], "train": asdict(train_config(cfg))}

    def add(key, problem, h, set_name, cands, **extra):
        job = {**base, "comparison": key, "problem": problem, "horizon": int(h), "set": set_name,
               "candidates": [asdict(c) for c in cands], **extra}
        job["name"] = f"{key}/{problem}/h{h}/{set_name}"
        job["path"] = str(_job_path(out_dir, job))
        jobs.append(job)

    ep = candidates(cfg["endpoint"]["candidates"])
    pa = candidates(cfg["path"]["candidates"])
    for key in comps:
        for h in cfg["endpoint"]["horizons"]:
            add(key, "endpoint", h, "main", ep, keep_curves=(h == max(cfg["endpoint"]["horizons"])))
        for h in cfg["path"]["horizons"]:
            for problem in ("cost0", "cost2", "pathinfo", "mixture"):
                add(key, problem, h, "main", pa)
    return jobs


def build_swap_jobs(cfg: dict[str, Any], comps: dict[str, D.Comparison], out_dir: Path) -> list[dict]:
    sw = cfg.get("swaps")
    if not sw:
        return []
    jobs: list[dict[str, Any]] = []
    cands = candidates(sw["candidates"])
    for key in comps:
        for h in sw["horizons"]:
            job = {"seed": cfg["seed"], "folds": cfg["folds"], "train": asdict(train_config(cfg)),
                   "comparison": key, "problem": "endpoint", "horizon": int(h), "set": "swap",
                   "candidates": [asdict(c) for c in cands], "kind": "swap", "n_swaps": int(sw["n"])}
            job["name"] = f"{key}/endpoint/h{h}/swap"
            job["path"] = str(_job_path(out_dir, job))
            jobs.append(job)
    return jobs


def run_jobs(jobs: list[dict], comps: dict[str, D.Comparison], workers: int, log: Path) -> None:
    pending = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"{len(jobs)} jobs, {len(pending)} pending, {workers} workers", flush=True)
    if not pending:
        return
    pending.sort(key=lambda j: (-len(j["candidates"]), -j["horizon"]))
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh, ProcessPoolExecutor(workers, initializer=_init_worker,
                                                   initargs=(comps,)) as pool:
        for i, (name, secs) in enumerate(pool.map(run_job, pending), 1):
            line = f"[{i}/{len(pending)}] {name} {secs:.1f}s elapsed={time.time()-t0:.0f}s"
            fh.write(line + "\n"); fh.flush()
            if i % 10 == 0 or i == len(pending):
                print(line, flush=True)


# ------------------------------------------------------------------- stages
def stage_data(cfg: dict[str, Any]) -> tuple[dict[str, D.Comparison], list[D.Trajectory], D.Audit]:
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    comps, trajs, audit = D.load()
    audit.provenance["config"] = cfg["_config_path"]
    (out_dir / "audit.json").write_text(json.dumps(audit.to_dict(), indent=1, default=str))
    for key, comp in comps.items():
        _plan(comp, cfg["seed"], cfg["folds"]).to_json(out_dir / "folds" / f"{key}.json")
    print(f"audit: {audit.n_trajectories} trajectories, {audit.n_inits} initializations, "
          f"{len(audit.exclusions)} exclusions; comparisons: {audit.comparison_counts}")
    return comps, trajs, audit


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stage", default="all",
                    choices=["data", "obs", "fit", "swaps", "tables", "report", "all"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out_dir = Path(cfg["output_dir"])
    workers = a.workers or int(cfg.get("workers", max(1, (os.cpu_count() or 2) - 2)))
    comps, trajs, audit = stage_data(cfg)
    if a.stage in ("obs", "tables", "report", "all"):
        from .tables import build_observables
        build_observables(cfg, comps, trajs, out_dir)
    if a.stage in ("fit", "all"):
        jobs = build_jobs(cfg, comps, out_dir)
        if a.only:
            jobs = [j for j in jobs if a.only in j["name"]]
        run_jobs(jobs, comps, workers, out_dir / "jobs.log")
    if a.stage in ("swaps", "all"):
        jobs = build_swap_jobs(cfg, comps, out_dir)
        if a.only:
            jobs = [j for j in jobs if a.only in j["name"]]
        run_jobs(jobs, comps, workers, out_dir / "jobs.log")
    if a.stage in ("tables", "report", "all"):
        from .tables import build_fitted_tables
        build_fitted_tables(cfg, comps, out_dir)
    if a.stage in ("report", "all"):
        from .report_tex import build
        build(cfg, out_dir)


if __name__ == "__main__":
    main()
