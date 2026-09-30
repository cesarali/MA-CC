"""Collect job outputs into result tables (CSV + JSON) under ``<output_dir>/tables``."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import Comparison, TARGETS
from .folds import derive_seed
from .uncertainty import null_summary, paired_bootstrap

KEY_COLS = ["comparison", "q", "rho", "schedule", "budget", "horizon", "repeat"]


def _load_jobs(out_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for p in sorted((out_dir / "jobs").glob("*.pkl")):
        with open(p, "rb") as fh:
            jobs.append(pickle.load(fh))
    return jobs


def _mode(names: list[str]) -> str:
    return Counter(names).most_common(1)[0][0]


def _row_common(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r[k] for k in ("comparison", "q", "rho", "schedule", "budget", "horizon", "repeat", "m", "problem")}


def estimates_table(jobs: list[dict[str, Any]], n_boot: int, seed: int, max_epochs: int = 1000) -> pd.DataFrame:
    rows = []
    for r in jobs:
        if r.get("kind") == "swap":
            continue
        row = _row_common(r)
        row["set"] = r["set"]
        sel = r["selected"]
        row["selected_mode"] = _mode(r["selection"]["selected_by_fold"])
        row["selected_by_fold"] = "|".join(r["selection"]["selected_by_fold"])
        row["score"] = sel["score"]
        row["log_loss"] = sel["log_loss"]; row["accuracy"] = sel["accuracy"]
        if r["problem"] in ("endpoint", "pathinfo"):
            row["brier"] = sel["brier"]
            for eps in ("1e-06", "0.0001", "1e-08"):
                row[f"score_eps{eps}"] = float(sel[f"per_parent_eps{eps}"].mean())
                row[f"saturation_eps{eps}"] = sel[f"saturation_eps{eps}"]
            pp = sel["per_parent_eps1e-06"]
        else:
            for tag in ("raw", "cap2", "cap5", "cap10"):
                row[f"nwj_{tag}"] = sel[f"nwj_{tag}"]; row[f"dv_{tag}"] = sel[f"dv_{tag}"]
                row[f"plugin_{tag}"] = sel[f"plugin_{tag}"]
                row[f"ess_{tag}"] = sel[f"tail_{tag}"]["ess"]; row[f"maxshare_{tag}"] = sel[f"tail_{tag}"]["max_weight_share"]
            pp = sel["per_parent_nwj_raw"]
        bs = paired_bootstrap({"score": pp}, n_boot, derive_seed(seed, "boot", r["comparison"], r["problem"], r["horizon"]))
        row["score_lo"], row["score_hi"], row["score_sd"] = bs["score"]["lo"], bs["score"]["hi"], bs["score"]["sd"]
        for name, res in r["by_candidate"].items():
            row[f"cand:{name}"] = res["score"]
            row[f"loss:{name}"] = res["log_loss"]
        # n_params from fold 0; fraction of neural refits that hit the epoch cap
        f0 = r["selection"]["folds"][0]
        for name, npar in f0["n_params"].items():
            row[f"nparams:{name}"] = npar
        caps = [f["refit_epochs"][n] for f in r["selection"]["folds"] for n in f["refit_epochs"] if f["refit_epochs"][n] is not None]
        row["epoch_cap_fraction"] = float(np.mean([e >= max_epochs for e in caps])) if caps else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def frequency_table(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in jobs:
        if r.get("problem") != "endpoint" or "frequency" not in r:
            continue
        row = _row_common(r)
        for k, v in r["frequency"].items():
            if not k.startswith("per_parent"):
                row[k] = v
        row["classifier_score"] = r["selected"]["score"]
        rows.append(row)
    return pd.DataFrame(rows)


def efficiency_table(jobs: list[dict[str, Any]], n_boot: int, seed: int) -> pd.DataFrame:
    """Join endpoint information with the two cost problems at shared horizons; paired bootstrap."""
    index: dict[tuple, dict[str, Any]] = {}
    for r in jobs:
        if r.get("kind") == "swap" or r.get("set") != "main":
            continue
        index[(r["comparison"], r["horizon"], r["repeat"], r["problem"])] = r
    rows = []
    for (comp, h, rep, prob), r in index.items():
        if prob != "endpoint":
            continue
        c0, c2 = index.get((comp, h, rep, "cost0")), index.get((comp, h, rep, "cost2"))
        if c0 is None or c2 is None:
            continue
        row = _row_common(r); row.pop("problem")
        info = r["selected"]["per_parent_eps1e-06"]
        for tag in ("raw", "cap5"):
            cost = 0.5 * (c0["selected"][f"per_parent_nwj_{tag}"] + c2["selected"][f"per_parent_nwj_{tag}"])
            bs = paired_bootstrap({"info": info, "cost": cost}, n_boot, derive_seed(seed, "eff", comp, h, rep))
            row[f"info"] = bs["info"]["point"]; row["info_lo"], row["info_hi"] = bs["info"]["lo"], bs["info"]["hi"]
            row[f"cost_{tag}"] = bs["cost"]["point"]
            row[f"cost_{tag}_lo"], row[f"cost_{tag}_hi"] = bs["cost"]["lo"], bs["cost"]["hi"]
            row[f"cost_{tag}_nonpositive_frac"] = bs["cost_nonpositive_fraction"]
            row[f"eta_{tag}_supported"] = bs["eta"]["supported"]
            if bs["eta"]["supported"]:
                row[f"eta_{tag}"] = bs["eta"]["point"]; row[f"eta_{tag}_lo"], row[f"eta_{tag}_hi"] = bs["eta"]["lo"], bs["eta"]["hi"]
            else:
                row[f"eta_{tag}"] = np.nan; row[f"eta_{tag}_lo"] = np.nan; row[f"eta_{tag}_hi"] = np.nan
        row["cost0_nwj_raw"], row["cost2_nwj_raw"] = c0["selected"]["nwj_raw"], c2["selected"]["nwj_raw"]
        row["cost0_dv_raw"], row["cost2_dv_raw"] = c0["selected"]["dv_raw"], c2["selected"]["dv_raw"]
        row["cost0_model"], row["cost2_model"] = _mode(c0["selection"]["selected_by_fold"]), _mode(c2["selection"]["selected_by_fold"])
        row["info_model"] = _mode(r["selection"]["selected_by_fold"])
        row["cost0_ess"], row["cost2_ess"] = c0["selected"]["tail_raw"]["ess"], c2["selected"]["tail_raw"]["ess"]
        rows.append(row)
    return pd.DataFrame(rows)


def swaps_table(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in jobs:
        if r.get("kind") != "swap":
            continue
        s = null_summary(r["null_scores"], r["observed"])
        rows.append({"comparison": r["comparison"], "problem": r["problem"], "horizon": r["horizon"],
                     "candidates": "|".join(r["candidates"]), **s})
    return pd.DataFrame(rows)


def memory_table(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in jobs:
        if r.get("kind") == "swap" or not str(r.get("set", "")).startswith("mem"):
            continue
        row = _row_common(r); row["history"] = int(r["set"][3:])
        row["nwj_raw"] = r["selected"]["nwj_raw"]; row["nwj_cap5"] = r["selected"]["nwj_cap5"]
        row["dv_raw"] = r["selected"]["dv_raw"]; row["selected_mode"] = _mode(r["selection"]["selected_by_fold"])
        for name, res in r["by_candidate"].items():
            row[f"cand:{name}"] = res["score"]
        rows.append(row)
    return pd.DataFrame(rows)


def direction_table(comps: dict[str, Comparison], horizons: list[int]) -> pd.DataFrame:
    """Mean vote vectors by branch and paired signed gain toward each target relative to silence."""
    rows = []
    for comp in comps.values():
        for h in horizons:
            row = {"comparison": comp.key, "q": comp.q, "rho": comp.rho, "schedule": comp.schedule,
                   "budget": comp.budget, "horizon": h, "m": comp.m}
            sil = comp.silent[:, h] / 24.0
            for a in range(3):
                row[f"silent_mean_N{a}"] = float(sil[:, a].mean())
            for z in TARGETS:
                ctl = comp.controlled[z][:, h] / 24.0
                for a in range(3):
                    row[f"target{z}_mean_N{a}"] = float(ctl[:, a].mean())
                gain = ctl[:, z] - sil[:, z]
                row[f"gain_toward_target{z}"] = float(gain.mean())
                row[f"gain_toward_target{z}_se"] = float(gain.std(ddof=1) / np.sqrt(len(gain)))
            rows.append(row)
    return pd.DataFrame(rows)


def training_curves(cfg: dict[str, Any], comps: dict[str, Comparison], horizon: int = 10) -> dict[str, Any]:
    """Train-vs-inner-validation curves for a few neural candidates (outer fold 0, inner fold 0, seed 0)."""
    from .crossfit import Candidate, fit_candidate
    from .models import TrainConfig
    from .pipelines import build_dataset
    from .folds import make_fold_plan
    t = dict(cfg.get("training", {})); t["seeds"] = (0,)
    tcfg = TrainConfig(**t)
    keys = [k for k in ("q12_rho1.00_sensing_b12", "q3_rho0.70_always_b3") if k in comps] or list(comps)[:2]
    specs = [("endpoint", Candidate("mlp", 0.1, 4)), ("cost0", Candidate("flat", 0.01, 4)),
             ("cost0", Candidate("gru", 0.01, 4)), ("cost2", Candidate("gru", 0.01, 4))]
    out = {}
    for k in keys:
        comp = comps[k]
        plan = make_fold_plan(comp.parent_ids, cfg["master_seed"], n_outer=cfg["folds"]["outer"], n_inner=cfg["folds"]["inner"], tag=comp.key)
        fit_p, val_p = plan.inner_split(0, 0)
        for problem, cand in specs:
            ds = build_dataset(comp, problem, horizon)
            model = fit_candidate(ds, cand, ds.subset(fit_p), tcfg, val=ds.subset(val_p))
            out[f"{k}/{problem}/{cand.name}"] = model.curves[0]
    return out


def build_tables(cfg: dict[str, Any], comps: dict[str, Comparison], out_dir: Path) -> dict[str, pd.DataFrame]:
    jobs = _load_jobs(out_dir)
    tdir = out_dir / "tables"; tdir.mkdir(exist_ok=True)
    n_boot = int(cfg.get("bootstrap", {}).get("resamples", 2000))
    seed = int(cfg["master_seed"])
    tables = {
        "estimates": estimates_table(jobs, n_boot, seed, int(cfg.get("training", {}).get("max_epochs", 1000))),
        "frequency": frequency_table(jobs),
        "efficiency": efficiency_table(jobs, n_boot, seed),
        "label_swaps": swaps_table(jobs),
        "memory": memory_table(jobs),
        "direction": direction_table(comps, sorted(set(cfg["endpoint"]["horizons"]))),
    }
    for name, df in tables.items():
        df.to_csv(tdir / f"{name}.csv", index=False)
    (tdir / "curves.json").write_text(json.dumps(training_curves(cfg, comps)))
    print({k: len(v) for k, v in tables.items()})
    return tables
