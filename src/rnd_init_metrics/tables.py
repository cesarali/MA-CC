"""Assemble observables and fitted estimates into CSV tables with intervals.

All uncertainty comes from resampling whole initializations: every arm, horizon
and outcome coordinate of a resampled initialization travels together, which
preserves the covariance induced by the shared silent branch.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import observables as OB
from .data import POPULATION, TARGETS, Comparison, Trajectory

LOG2 = np.log(2.0)
OUTCOMES = {"truth": 0, "alloc1": 1, "false_target": 2}


def _boot(per_init: dict[str, np.ndarray], n: int, seed: int) -> dict[str, tuple[float, float]]:
    """Percentile intervals from a shared resample of initializations."""
    m = len(next(iter(per_init.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    out = {}
    for k, v in per_init.items():
        draws = np.asarray(v)[idx].mean(axis=1)
        out[k] = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    return out


# ------------------------------------------------------------- observables
def build_observables(cfg: dict[str, Any], comps: dict[str, Comparison],
                      trajs: list[Trajectory], out_dir: Path) -> None:
    tdir = out_dir / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    nb, seed = int(cfg["bootstrap"]["resamples"]), int(cfg["seed"])
    horizons = list(range(1, cfg["horizon"] + 1))

    # ---- effects: gains, M/D, per outcome coordinate, every horizon
    rows = []
    for key, c in comps.items():
        for h in horizons:
            per: dict[str, np.ndarray] = {}
            rec: dict[str, Any] = {"comparison": key, "profile": c.profile, "rho": c.rho,
                                   "budget": c.budget, "horizon": h, "m": c.m}
            for name, a in OUTCOMES.items():
                md = OB.md_decomposition(c, a, h)
                for stat in ("g_plus", "g_minus", "M", "D", "switch"):
                    rec[f"{name}_{stat}"] = float(md[stat].mean())
                    per[f"{name}_{stat}"] = md[stat]
                rec[f"{name}_base"] = float(OB.fractions(c.silent[:, h, a]).mean())
            # total incorrect support is the exact complement of truth
            rec["err_M"] = -rec["truth_M"]; rec["err_D"] = -rec["truth_D"]
            ci = _boot(per, nb, seed)
            for k, (lo, hi) in ci.items():
                rec[f"{k}_lo"], rec[f"{k}_hi"] = lo, hi
            rows.append(rec)
    pd.DataFrame(rows).to_csv(tdir / "effects.csv", index=False)

    # ---- command channel, following, resources, information per resource
    rows = []
    for key, c in comps.items():
        for h in horizons:
            ch = OB.command_channel(c, h)
            rec = {"comparison": key, "profile": c.profile, "rho": c.rho, "budget": c.budget,
                   "horizon": h, "m": c.m, "mi_bits": ch["mi_bits"],
                   "eta_command": ch["eta_command"], "following": ch["following"],
                   "following_base": ch["following_base"], "following_gain": ch["following_gain"]}
            for al in (0.1, 1.0):
                rec[f"mi_bits_alpha{al:g}"] = OB.command_channel(c, h, alpha=al)["mi_bits"]
            for v in range(3):
                rec[f"p{v}_given_t0"] = ch["p_v_given_0"][v]
                rec[f"p{v}_given_t2"] = ch["p_v_given_2"][v]
                rec[f"p{v}_base"] = ch["p_base"][v]
            for lam in cfg["resource_lambdas"]:
                ipr = OB.info_per_resource(c, h, float(lam))
                rec[f"resource_lam{lam:g}"] = ipr["resource_mean"]
                rec[f"bits_per_resource_lam{lam:g}"] = ipr["bits_per_resource"]
                rec[f"fgain_per_resource_lam{lam:g}"] = ipr["following_gain_per_resource"]
            rec["posts_t0"] = float(OB.posting_cost(c, 0, h).mean())
            rec["posts_t2"] = float(OB.posting_cost(c, 2, h).mean())
            rec["senses_t0"] = float(OB.sensing_cost(c, 0, h).mean())
            rec["senses_t2"] = float(OB.sensing_cost(c, 2, h).mean())
            rec.update(OB.budget_efficiency(c, 0, h))
            rows.append(rec)
    pd.DataFrame(rows).to_csv(tdir / "command.csv", index=False)

    # ---- susceptibility families and activation information
    rows = []
    for key, c in comps.items():
        for z in TARGETS:
            for name, a in OUTCOMES.items():
                sm = OB.state_matched_susceptibility(c, a, z)
                ipw = OB.ipw_causal_response(c, a, z)
                av = OB.available_susceptibility(c, a, z)
                rows.append({"comparison": key, "profile": c.profile, "rho": c.rho,
                             "budget": c.budget, "target": z, "outcome": name,
                             "chi_bar": sm["chi_bar"], "chi_support_mass": sm["support_mass"],
                             "chi_unconditioned": sm["unconditioned"],
                             "activation_rate": sm["activation_rate"],
                             **{k: v for k, v in ipw.items() if not k.startswith("per_init")},
                             "avail_row_ratio": av["row_ratio_mean"],
                             "avail_cell_ratio": av["cell_ratio"]})
    pd.DataFrame(rows).to_csv(tdir / "susceptibility.csv", index=False)

    # ---- activation / assigned-policy information and their efficiencies, by horizon
    rows = []
    for key, c in comps.items():
        for h in horizons:
            for z in TARGETS:
                act = OB.activation_information(c, 0, z, h)
                ir = OB.information_response_bound(c, 0, z, h)
                ap = OB.assigned_policy_information(c, 0, z, h)
                rows.append({"comparison": key, "profile": c.profile, "rho": c.rho,
                             "budget": c.budget, "horizon": h, "target": z,
                             **act, **ir, **ap})
    pd.DataFrame(rows).to_csv(tdir / "activation.csv", index=False)

    # ---- sensing channel and evidence observables
    rows = []
    for key, c in comps.items():
        for z in TARGETS:
            rows.append({"comparison": key, "target": z, **OB.sensing_information(c, z)})
    pd.DataFrame(rows).to_csv(tdir / "sensing.csv", index=False)

    rows = []
    for key, c in comps.items():
        for h in horizons:
            rows.append({"comparison": key, "profile": c.profile, "rho": c.rho,
                         "budget": c.budget, "horizon": h, **OB.evidence_observables(c, h)})
    pd.DataFrame(rows).to_csv(tdir / "evidence.csv", index=False)

    # ---- minimum task cost K_min for the truth coordinate
    rows = []
    for key, c in comps.items():
        for h in horizons:
            for z in TARGETS:
                for name, a in OUTCOMES.items():
                    rows.append({"comparison": key, "horizon": h, "target": z, "outcome": name,
                                 **OB.k_min(c, a, h, z)})
    pd.DataFrame(rows).to_csv(tdir / "kmin.csv", index=False)

    # ---- descriptive trajectories (mean vote shares by arm and round)
    rows = []
    for key, c in comps.items():
        for h in range(0, cfg["horizon"] + 1):
            rec = {"comparison": key, "profile": c.profile, "rho": c.rho,
                   "budget": c.budget, "horizon": h}
            for arm, arr in (("silent", c.silent), ("t0", c.controlled[0]), ("t2", c.controlled[2])):
                fr = OB.fractions(arr[:, h, :]).mean(axis=0)
                for v in range(3):
                    rec[f"{arm}_x{v}"] = fr[v]
            rows.append(rec)
    pd.DataFrame(rows).to_csv(tdir / "trajectories.csv", index=False)
    print("observables written to", tdir)


# ------------------------------------------------------------ fitted tables
def _load_jobs(out_dir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((out_dir / "jobs").glob("*.pkl")):
        with open(p, "rb") as fh:
            out.append(pickle.load(fh))
    return out


def build_fitted_tables(cfg: dict[str, Any], comps: dict[str, Comparison], out_dir: Path) -> None:
    tdir = out_dir / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    jobs = _load_jobs(out_dir)
    nb, seed = int(cfg["bootstrap"]["resamples"]), int(cfg["seed"])

    est_rows, curves, swap_rows = [], {}, []
    per_init: dict[tuple, np.ndarray] = {}
    for j in jobs:
        if j.get("kind") == "swap":
            null = np.asarray(j["null"])
            swap_rows.append({"comparison": j["comparison"], "horizon": j["horizon"],
                              "observed": j["observed"], "null_mean": float(null.mean()),
                              "null_p95": float(np.percentile(null, 95)), "n_swaps": len(null),
                              "p_value": float((1 + (null >= j["observed"]).sum()) / (1 + len(null)))})
            continue
        sel = j["selected"]
        rec = {k: j[k] for k in ("comparison", "profile", "rho", "budget", "problem", "horizon", "m")}
        rec["selected_mode"] = pd.Series(j["selected_by_fold"]).mode().iloc[0]
        for k, v in sel.items():
            if not k.startswith("per_init") and not isinstance(v, np.ndarray):
                rec[k] = v
        for name, sc in j["by_candidate"].items():
            rec[f"cand:{name}"] = sc.get("score_bits", sc.get("score_nats"))
        if j["problem"] == "endpoint":
            rec.update({f"freq_{k}": v for k, v in j.get("frequency", {}).items()})
        est_rows.append(rec)
        per_init[(j["comparison"], j["problem"], j["horizon"])] = sel["per_init"]
        if "curves" in j:
            curves[f"{j['comparison']}/h{j['horizon']}"] = j["curves"]
    pd.DataFrame(est_rows).to_csv(tdir / "estimates.csv", index=False)
    if swap_rows:
        pd.DataFrame(swap_rows).to_csv(tdir / "swaps.csv", index=False)

    # ---- efficiencies: cost C = (K0+K2)/2 and the ratios of section 11
    rows = []
    for key, c in comps.items():
        for h in sorted({k[2] for k in per_init if k[0] == key and k[1] == "cost0"}):
            need = [(key, p, h) for p in ("cost0", "cost2", "pathinfo", "mixture")]
            if not all(n in per_init for n in need):
                continue
            k0, k2 = per_init[(key, "cost0", h)], per_init[(key, "cost2", h)]
            cost = 0.5 * (k0 + k2)
            pinfo = per_init[(key, "pathinfo", h)] * LOG2          # bits -> nats
            mix = per_init[(key, "mixture", h)]
            end = per_init.get((key, "endpoint", h))
            rec: dict[str, Any] = {"comparison": key, "profile": c.profile, "rho": c.rho,
                                   "budget": c.budget, "horizon": h, "m": c.m,
                                   "K0_nats": float(k0.mean()), "K2_nats": float(k2.mean()),
                                   "cost_nats": float(cost.mean()),
                                   "I_path_nats": float(pinfo.mean()),
                                   "D_mixture_nats": float(mix.mean()),
                                   "decomp_sum_nats": float(pinfo.mean() + mix.mean())}
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, c.m, size=(nb, c.m))
            cd = cost[idx].mean(axis=1)
            rec["cost_nonpositive_frac"] = float((cd <= 0).mean())
            rec["cost_lo"], rec["cost_hi"] = np.percentile(cd, [2.5, 97.5])
            supported = rec["cost_nonpositive_frac"] == 0.0
            rec["eta_supported"] = bool(supported)
            for nm, num in (("eta_ctl", pinfo), ("eta_end", end if end is not None else None)):
                if num is None:
                    continue
                nn = num * (LOG2 if nm == "eta_end" else 1.0)
                rec[nm] = float(nn.mean() / cost.mean()) if supported else np.nan
                if supported:
                    d = nn[idx].mean(axis=1) / cd
                    rec[f"{nm}_lo"], rec[f"{nm}_hi"] = np.percentile(d, [2.5, 97.5])
            if end is not None:
                rec["I_end_nats"] = float(end.mean() * LOG2)
                rec["I_end_bits"] = float(end.mean())
            # eta_task uses the closed-form minimum cost for the truth coordinate.
            # It is only formed when both targets' K_min are identified: an
            # unidentified K_min is a baseline-support failure, not a large cost.
            kms = [OB.k_min(c, 0, h, z) for z in (0, 2)]
            ident = all(x["K_min_identified"] for x in kms)
            rec["K_min_identified"] = ident
            rec["K_min_mean_nats"] = float(np.mean([x["K_min_nats"] for x in kms])) if ident else np.nan
            rec["eta_task"] = (float(rec["K_min_mean_nats"] / rec["cost_nats"])
                               if ident and supported and rec["cost_nats"] > 0 else np.nan)
            rows.append(rec)
    pd.DataFrame(rows).to_csv(tdir / "efficiency.csv", index=False)
    if curves:
        (tdir / "curves.json").write_text(json.dumps(
            {k: {str(f): c for f, c in v.items()} for k, v in curves.items()}, default=float))
    print("fitted tables written to", tdir)
