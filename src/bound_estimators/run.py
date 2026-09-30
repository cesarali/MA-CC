"""Orchestrator: config -> data audit -> fold plans -> parallel jobs -> tables -> report.

    python -m bound_estimators.run --config configs/analysis/bound_estimators/blackboard_checkpoint_ensemble_01.yaml

Stages (``--stage``): ``data``, ``fit``, ``swaps``, ``tables``, ``report``, ``all``.
Every job's full output (per-parent contributions, out-of-fold logits, selection
records) is pickled under ``<output_dir>/jobs`` and skipped on re-run, so the
pipeline can be resumed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import __version__
from .crossfit import Candidate
from .data import Comparison, load_comparisons, save_canonical, verify_archive
from .folds import FoldPlan, derive_seed, make_fold_plan
from .models import TrainConfig
from .pipelines import run_problem
from .uncertainty import null_summary, swap_branch, swap_targets

_COMPS: dict[str, Comparison] = {}


def parse_candidates(specs: list[dict[str, Any]]) -> list[Candidate]:
    out = []
    for s in specs:
        s = dict(s)
        lams = s.pop("lam", [0.0])
        widths = s.pop("width", [0])
        lams = lams if isinstance(lams, list) else [lams]
        widths = widths if isinstance(widths, list) else [widths]
        for lam in lams:
            for w in widths:
                out.append(Candidate(family=s["family"], lam=float(lam), width=int(w),
                                     history=s.get("history"), cap=s.get("cap")))
    return out


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    return cfg


def train_config(cfg: dict[str, Any]) -> TrainConfig:
    t = dict(cfg.get("training", {}))
    if "seeds" in t:
        t["seeds"] = tuple(int(s) for s in t["seeds"])
    return TrainConfig(**t)


# --------------------------------------------------------------------------- jobs
def _init_worker(comps: dict[str, Comparison]) -> None:
    global _COMPS
    _COMPS = comps
    import torch
    torch.set_num_threads(1)


def _job_path(out_dir: Path, job: dict[str, Any]) -> Path:
    name = f"{job['comparison']}__{job['problem']}__h{job['horizon']}__{job['set']}__r{job['repeat']}"
    if job.get("kind") == "swap":
        name = "swap__" + name
    return out_dir / "jobs" / f"{name}.pkl"


def _plan_for(comp: Comparison, master_seed: int, repeat: int, folds: dict[str, int]) -> FoldPlan:
    return make_fold_plan(comp.parent_ids, master_seed, repeat=repeat, n_outer=folds["outer"],
                          n_inner=folds["inner"], tag=comp.key)


def run_job(job: dict[str, Any]) -> tuple[str, float]:
    t0 = time.time()
    path = Path(job["path"])
    if path.exists():
        return job["name"], 0.0
    comp = _COMPS[job["comparison"]]
    cands = [Candidate(**c) for c in job["candidates"]]
    tcfg = TrainConfig(**job["train"])
    plan = _plan_for(comp, job["master_seed"], job["repeat"], job["folds"])
    if job.get("kind") == "swap":
        rng = np.random.default_rng(derive_seed(job["master_seed"], "swap", comp.key, job["problem"], job["horizon"]))
        observed = run_problem(comp, job["problem"], job["horizon"], cands, plan, tcfg)["selected"]["score"]
        scores = []
        for _ in range(job["n_swaps"]):
            if job["problem"] == "endpoint":
                sc = swap_targets(comp, rng)
            else:
                sc = swap_branch(comp, 0 if job["problem"] == "cost0" else 2, rng)
            r = run_problem(sc, job["problem"], job["horizon"], cands, plan, tcfg)
            scores.append(r["selected"]["score"])
        result: dict[str, Any] = {"kind": "swap", "comparison": comp.key, "problem": job["problem"],
                                  "horizon": job["horizon"], "null_scores": np.array(scores),
                                  "observed": observed, "candidates": [c.name for c in cands]}
    else:
        result = run_problem(comp, job["problem"], job["horizon"], cands, plan, tcfg,
                             keep_curves=job.get("keep_curves", False))
        result["set"] = job["set"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    return job["name"], time.time() - t0


def build_jobs(cfg: dict[str, Any], comps: dict[str, Comparison], out_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    base = {"master_seed": cfg["master_seed"], "folds": cfg["folds"], "train": asdict(train_config(cfg))}
    repeats = int(cfg["folds"].get("repeats", 1))

    def add(comp_key, problem, horizon, set_name, cands, repeat=0, **extra):
        job = {**base, "comparison": comp_key, "problem": problem, "horizon": int(horizon), "set": set_name,
               "candidates": [asdict(c) for c in cands], "repeat": repeat, **extra}
        job["name"] = f"{comp_key}/{problem}/h{horizon}/{set_name}/r{repeat}" + ("/swap" if extra.get("kind") == "swap" else "")
        job["path"] = str(_job_path(out_dir, job))
        jobs.append(job)

    keys = list(comps)
    ep, co, dg = cfg["endpoint"], cfg["cost"], cfg.get("diagnostics", {})
    ep_c, co_c = parse_candidates(ep["candidates"]), parse_candidates(co["candidates"])
    for r in range(repeats):
        for k in keys:
            for h in ep["horizons"]:
                add(k, "endpoint", h, "main", ep_c, repeat=r, keep_curves=(h == 10 and r == 0))
            for h in co["horizons"]:
                for problem in ("cost0", "cost2"):
                    add(k, problem, h, "main", co_c, repeat=r, keep_curves=(h == 10 and r == 0))
            # direct variational critics are a separate output (selected by inner NWJ), never
            # mixed with BCE-trained classifiers under the one-standard-error rule
            direct = parse_candidates(co.get("direct_candidates", []))
            if direct:
                for h in co.get("direct_horizons", []):
                    for problem in ("cost0", "cost2"):
                        add(k, problem, h, "direct", [Candidate(family="constant")] + direct, repeat=r)
    if dg:
        dg_c = parse_candidates(dg["candidates"])
        for k in keys:
            for h in dg.get("pathinfo_horizons", []):
                add(k, "pathinfo", h, "diag", dg_c)
            for h in dg.get("mixture_horizons", []):
                add(k, "mixture", h, "diag", dg_c)
    mem = co.get("memory")
    if mem:
        for k in mem.get("comparisons", keys):
            for hist in mem["histories"]:
                mc = [Candidate(family=c.family, lam=c.lam, width=c.width, history=hist)
                      for c in parse_candidates(mem["candidates"])]
                for problem in ("cost0", "cost2"):
                    add(k, problem, mem["horizon"], f"mem{hist}", mc)
    return jobs


def build_swap_jobs(cfg: dict[str, Any], comps: dict[str, Comparison], out_dir: Path) -> list[dict[str, Any]]:
    sw = cfg.get("label_swaps")
    if not sw:
        return []
    jobs: list[dict[str, Any]] = []
    base = {"master_seed": cfg["master_seed"], "folds": cfg["folds"], "train": asdict(train_config(cfg))}
    cands = parse_candidates(sw["candidates"])
    for k in comps:
        for h in sw["horizons"]:
            for problem in sw.get("problems", ["endpoint", "cost0", "cost2"]):
                job = {**base, "comparison": k, "problem": problem, "horizon": int(h), "set": "swap",
                       "candidates": [asdict(c) for c in cands], "repeat": 0, "kind": "swap",
                       "n_swaps": int(sw["n"])}
                job["name"] = f"{k}/{problem}/h{h}/swap"
                job["path"] = str(_job_path(out_dir, job))
                jobs.append(job)
    return jobs


def run_jobs(jobs: list[dict[str, Any]], comps: dict[str, Comparison], workers: int, log: Path) -> None:
    pending = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"{len(jobs)} jobs, {len(pending)} pending, {workers} workers", flush=True)
    if not pending:
        return
    # heavy jobs first so the pool tail is short
    pending.sort(key=lambda j: (-len(j["candidates"]), -j["horizon"]))
    t0 = time.time()
    with open(log, "a") as fh, mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(comps,)) as pool:
        for i, (name, dt) in enumerate(pool.imap_unordered(run_job, pending), 1):
            line = f"[{i}/{len(pending)}] {name} {dt:.1f}s elapsed={time.time() - t0:.0f}s"
            fh.write(line + "\n"); fh.flush()
            if i % 10 == 0 or i == len(pending):
                print(line, flush=True)


# --------------------------------------------------------------------------- stages
def stage_data(cfg: dict[str, Any]) -> tuple[dict[str, Comparison], dict[str, Any]]:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    comps, audit, digest = load_comparisons(Path(cfg["archive"]), Path(cfg["work_dir"]),
                                            cfg.get("archive_sha256"), horizon=int(cfg.get("horizon", 10)))
    provenance = {"package_version": __version__, "archive": cfg["archive"], "archive_sha256": digest,
                  "config": cfg["_config_path"]}
    if cfg.get("frozen_inputs_archive"):
        provenance["frozen_inputs_sha256"] = verify_archive(Path(cfg["frozen_inputs_archive"]),
                                                            cfg.get("frozen_inputs_sha256"))
    (out_dir / "audit.json").write_text(json.dumps({**audit.to_dict(), "provenance": provenance}, indent=1))
    save_canonical(comps, out_dir / "canonical_paths.parquet")
    for r in range(int(cfg["folds"].get("repeats", 1))):
        for comp in comps.values():
            _plan_for(comp, cfg["master_seed"], r, cfg["folds"]).to_json(out_dir / "folds" / f"{comp.key}_r{r}.json")
    print(f"audit: {audit.n_parents_total} parents, {audit.complete_paths} complete paths, "
          f"{len(audit.exclusions)} exclusions; comparisons: {audit.comparison_counts}")
    return comps, audit.to_dict()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stage", default="all", choices=["data", "fit", "swaps", "tables", "report", "all"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", default=None, help="substring filter on job names (debugging)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    workers = args.workers or int(cfg.get("workers", max(1, (os.cpu_count() or 2) - 2)))
    comps, _ = stage_data(cfg)
    if args.stage in ("fit", "all"):
        jobs = build_jobs(cfg, comps, out_dir)
        if args.only:
            jobs = [j for j in jobs if args.only in j["name"]]
        run_jobs(jobs, comps, workers, out_dir / "jobs.log")
    if args.stage in ("swaps", "all"):
        jobs = build_swap_jobs(cfg, comps, out_dir)
        if args.only:
            jobs = [j for j in jobs if args.only in j["name"]]
        run_jobs(jobs, comps, workers, out_dir / "jobs.log")
    if args.stage in ("tables", "report", "all"):
        from .tables import build_tables
        build_tables(cfg, comps, out_dir)
    if args.stage in ("report", "all"):
        from .report_tex import build
        build(cfg, out_dir, Path(cfg.get("synthetic_dir", "results/bound_estimators/synthetic")))


if __name__ == "__main__":
    main()
