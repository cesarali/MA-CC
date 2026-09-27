"""Compact, resumable full-grid Santa Fe v3 execution on Cygnus.

Every physical cell retains all episode/round observables, all 64 exact initial
states, and three pre-action snapshot rounds. One declared cell per shard also
retains its full identity-level round and micro ledgers. Theory retains paired
initial-block means and within-block path dispersion; only that raw subset
retains individual theory paths. No information estimator is replaced here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .cluster import _v3_code_hashes
from .config import load_config
from .runner import run
from .theory_integration import export_cell
from santa_fe_theory.core import Parameters, Solver, VERSION
from santa_fe_theory.io import load_snapshots, run_ensemble


ROUND_COLUMNS=(
    "cell_id","seed","round","beta_regime","N","F_plus","q","rho","budget",
    "truth_share","target_share","kappa_plus","kappa_minus","kappa_ctrl",
    "peer_board_target_count","controller_U","controller_effective_U",
    "controller_p_act","controller_sensed_messages","controller_sensor_target_count",
    "controller_observed_target_share","budget_used","B_peer_counts_json",
    "B_total_counts_json","kappa_mean_coverage","kappa_population_coverage",
    "kappa_mean_evidence_balance","epistemic_r_vote_correlation",
    "epistemic_s_vote_correlation",
)
SNAPSHOT_COLUMNS=(
    "cell_id","seed","round","beta_regime","N","F_plus","q","rho","budget",
    "agent_states_json","fact_weights_json","peer_board_json",
    "controller_fact_pool_json","rng_state_next_day_json","next_board_json",
    "peer_board_target_count","controller_U","controller_p_act","target_share",
    "kappa_plus","kappa_minus",
)
THEORY_VARIANTS={
    "sampling_matched_reduced":("without_replacement","paper_diagonal"),
    "paper_reduced":("with_replacement","paper_diagonal"),
}
THEORY_METRICS=("target_share","kappa_plus","kappa_minus","peer_target_share",
                "action_next","posts_next","front_next_plus_fact","front_next_minus_fact")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _root(config):
    return config.results_dir


def _cell(root,cell_id):
    return root/"cells"/f"cell-{cell_id:04d}"


def _code_hash():
    return sha(Path(__file__))


def _write_json(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")


def prepare(config):
    root=_root(config)
    root.mkdir(parents=True,exist_ok=True)
    (root/"cells").mkdir(exist_ok=True)
    (root/"logs").mkdir(exist_ok=True)
    copy=root/"config.yaml"
    if copy.is_file() and copy.read_bytes()!=config.path.read_bytes():
        raise ValueError("result root already belongs to a different config")
    copy.write_bytes(config.path.read_bytes())
    plan={"schema_version":1,"config_sha256":sha(config.path),
          "compact_runner_sha256":_code_hash(),"simulator_code_hashes":_v3_code_hashes(),
          "parameter_cells":len(config.cells),"episodes_per_cell":config.episodes,
          "expected_episodes":len(config.cells)*config.episodes,
          "raw_retention_rule":"first local cell of each shard only",
          "snapshot_rounds":[15,30,45],"theory_initial_blocks":16,
          "theory_paths_per_block":4,"theory_substeps":48,
          "theory_variants":THEORY_VARIANTS,
          "trajectory_scope":"full 61-round scalar/board summaries per episode; no all-cell micro ledger"}
    _write_json(root/"execution_plan.json",plan)
    return plan


def _check(config):
    path=_root(config)/"execution_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"run prepare first: {path}")
    p=json.loads(path.read_text())
    if p["config_sha256"]!=sha(config.path) or p["compact_runner_sha256"]!=_code_hash():
        raise ValueError("config or compact runner changed since prepare")
    if p["simulator_code_hashes"]!=_v3_code_hashes():
        raise ValueError("simulator semantics changed since prepare")
    return p


def _valid_seal(path,expected):
    if not path.is_file():
        return False
    try:
        d=json.loads(path.read_text())
        return all(d.get(k)==v for k,v in expected.items()) and all(
            (path.parent/name).is_file() and sha(path.parent/name)==digest
            for name,digest in d["files"].items())
    except (OSError,ValueError,KeyError):
        return False


def run_cell(config,cell_id):
    plan=_check(config)
    if not 0<=cell_id<len(config.cells):
        raise ValueError("cell ID outside resolved config")
    final_destination=_cell(_root(config),cell_id)
    expected={"config_sha256":plan["config_sha256"],"cell_id":cell_id,
              "episodes":config.episodes,"round_rows":config.episodes*61,
              "simulator_code_hashes":plan["simulator_code_hashes"]}
    seal=final_destination/"simulation_complete.json"
    if _valid_seal(seal,expected):
        return {"cell_id":cell_id,"status":"already_complete"}
    if final_destination.exists():
        raise ValueError(f"unsealed output exists; inspect before retry: {final_destination}")
    destination=final_destination.with_name(final_destination.name+".inprogress")
    if destination.exists():
        raise ValueError(f"in-progress output exists; inspect before retry: {destination}")
    destination.mkdir(parents=True)
    seal=destination/"simulation_complete.json"
    cell_config=replace(config,cells=(config.cells[cell_id],),
                        processes=min(config.processes,int(os.environ.get("SLURM_CPUS_PER_TASK",config.processes))))
    rounds,micro=run(cell_config)
    if len(rounds)!=expected["round_rows"] or rounds.seed.nunique()!=config.episodes:
        raise ValueError("unexpected episode/round count")
    if not set(ROUND_COLUMNS).issubset(rounds) or not set(SNAPSHOT_COLUMNS).issubset(rounds):
        raise ValueError("compact retained schema is absent from simulator output")
    files=[]
    metrics=rounds[list(ROUND_COLUMNS)]
    metrics.to_parquet(destination/"round_metrics.parquet",index=False)
    files.append("round_metrics.parquet")
    snapshots=rounds.loc[rounds["round"].isin((15,30,45)),list(SNAPSHOT_COLUMNS)]
    snapshots.to_parquet(destination/"pre_action_snapshots.parquet",index=False)
    files.append("pre_action_snapshots.parquet")
    late=rounds.loc[rounds["round"].between(41,60)].groupby("seed",as_index=False).agg(
        late_target_support=("target_share","mean"),
        late_target_temporal_sd=("target_share","std"),
        late_kappa_plus=("kappa_plus","mean"),
        late_kappa_minus=("kappa_minus","mean"),
        intervention_frequency=("controller_U","mean"),
        posts_late=("budget_used","sum"))
    final=rounds.loc[rounds["round"]==60,["seed","target_share"]].rename(
        columns={"target_share":"final_target_support"})
    late=late.merge(final,on="seed",validate="one_to_one")
    late["final_target_majority"]=(late.final_target_support>.5).astype(int)
    late["cell_id"]=cell_id
    late["cumulative_posts"]=rounds.groupby("seed").budget_used.sum().reindex(late.seed).to_numpy()
    late.to_parquet(destination/"episode_summary.parquet",index=False)
    files.append("episode_summary.parquet")
    export_cell(config,rounds,cell_id,destination/"theory_input",
                initial_blocks=config.episodes,theory_replicas=4)
    files.extend(["theory_input/initials.json","theory_input/simulator.csv",
                  "theory_input/simulation_manifest.json","theory_input/block_manifest.json"])
    if cell_id==0:
        rounds.to_parquet(destination/"raw_rounds.parquet",index=False)
        micro.to_parquet(destination/"raw_micro.parquet",index=False)
        files.extend(("raw_rounds.parquet","raw_micro.parquet"))
    _write_json(seal,{**expected,"files":{f:sha(destination/f) for f in files},
                      "raw_identity_ledger_retained":cell_id==0,
                      "snapshot_rounds":[15,30,45]})
    os.replace(destination,final_destination)
    return {"cell_id":cell_id,"status":"complete","files":len(files)}


def theory_cell(config,cell_id):
    plan=_check(config)
    destination=_cell(_root(config),cell_id)
    seal=destination/"simulation_complete.json"
    if not _valid_seal(seal,{"config_sha256":plan["config_sha256"],
                              "cell_id":cell_id,"episodes":config.episodes,
                              "round_rows":config.episodes*61,
                              "simulator_code_hashes":plan["simulator_code_hashes"]}):
        raise ValueError("simulation cell not sealed")
    out=destination/"theory_compact"
    out.mkdir(exist_ok=True)
    complete=out/"theory_complete.json"
    expected={"config_sha256":plan["config_sha256"],"cell_id":cell_id,
              "theory_runner_version":VERSION,"initial_blocks":16,"paths_per_block":4,
              "substeps":48,"variants":list(THEORY_VARIANTS)}
    if _valid_seal(complete,expected):
        return {"cell_id":cell_id,"status":"already_complete"}
    if any(out.iterdir()):
        raise ValueError(f"unsealed theory output exists; inspect before retry: {out}")
    all_initials=load_snapshots(destination/"theory_input/initials.json")
    initials=all_initials[:16]
    if len(initials)!=16:
        raise ValueError("fewer than 16 independent initial states")
    resolved=json.loads((destination/"theory_input/simulation_manifest.json").read_text())
    p=Parameters(**resolved["parameters"])
    sim=pd.read_csv(destination/"theory_input/simulator.csv")
    selected_ids=[x["snapshot_id"] for x in initials]
    sim=sim.loc[sim.initial_id.isin(selected_ids)]
    all_blocks=[];all_times=[];all_diagnostics=[];all_summaries=[]
    for variant,(sampling,covariance) in THEORY_VARIANTS.items():
        solver=Solver(rounds=60,replicas_per_initial=4,
            seed=20261250+config.seed*1000+cell_id,substeps=48,
            sampling=sampling,covariance=covariance,
            daytime_noise=True,boundary="project")
        rows,diagnostics=run_ensemble(p,solver,initials)
        theory=pd.DataFrame(rows)
        for metric in THEORY_METRICS:
            theory[metric]=pd.to_numeric(theory[metric],errors="coerce")
        means=theory.groupby(["initial_id","round"],as_index=False)[list(THEORY_METRICS)].mean()
        spread=theory.groupby(["initial_id","round"],as_index=False)[list(THEORY_METRICS)].std()
        for r in means.itertuples(index=False):
            corresponding=spread.loc[(spread.initial_id==r.initial_id)&(spread["round"]==r.round)].iloc[0]
            for metric in THEORY_METRICS:
                all_times.append({"variant":variant,"initial_id":r.initial_id,"round":r.round,
                    "metric":metric,"path_mean":getattr(r,metric),
                    "path_sd":corresponding[metric],"paths":4})
        diag=pd.DataFrame(diagnostics)
        cols=[c for c in diag if "_projections_" in c]
        diag["projection_count"]=diag[cols].sum(axis=1)
        agg=diag.groupby("initial_id",as_index=False).agg(
            projections=("projection_count","sum"),
            day_proposals=("day_proposals","sum"),night_proposals=("night_proposals","sum"))
        agg["variant"]=variant
        all_diagnostics.extend(agg.to_dict("records"))
        for metric in ("target_share","kappa_plus","kappa_minus"):
            differences=[]
            for initial in selected_ids:
                th=means.loc[(means.initial_id==initial)&means["round"].between(41,60),metric].mean()
                mi=sim.loc[(sim.initial_id==initial)&sim["round"].between(41,60),metric].mean()
                differences.append({"variant":variant,"initial_id":initial,"metric":metric,
                                    "theory_late":th,"micro_late":mi,"theory_minus_micro":th-mi})
            all_blocks.extend(differences)
            v=np.array([r["theory_minus_micro"] for r in differences])
            se=v.std(ddof=1)/np.sqrt(len(v))
            all_summaries.append({"variant":variant,"metric":metric,"late_bias":v.mean(),
                "paired_block_sd":v.std(ddof=1),"paired_block_se":se,
                "ci95_low_t":v.mean()-t.ppf(.975,len(v)-1)*se,
                "ci95_high_t":v.mean()+t.ppf(.975,len(v)-1)*se,
                "n_independent_initial_blocks":len(v),"nested_paths_per_block":4,
                "precision_met_0_04":bool(t.ppf(.975,len(v)-1)*se<=.04) if metric=="target_share" else None})
        if cell_id==0:
            theory.to_parquet(out/f"{variant}_illustrative_paths.parquet",index=False)
    pd.DataFrame(all_times).to_parquet(out/"block_timeseries.parquet",index=False)
    pd.DataFrame(all_blocks).to_parquet(out/"paired_blocks.parquet",index=False)
    pd.DataFrame(all_diagnostics).to_parquet(out/"projection_diagnostics.parquet",index=False)
    pd.DataFrame(all_summaries).to_csv(out/"variant_summary.csv",index=False)
    files=[p.name for p in out.iterdir() if p.is_file() and p.name!="theory_complete.json"]
    _write_json(complete,{**expected,"files":{f:sha(out/f) for f in files},
        "theory_level":"reduced_langevin","full_class_density_theory":False,
        "block_weighting":"equal independent saved initial states"})
    return {"cell_id":cell_id,"status":"complete","variants":list(THEORY_VARIANTS)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run-cell","theory-cell"))
    parser.add_argument("--config",required=True)
    parser.add_argument("--cell-id",type=int)
    args=parser.parse_args(argv)
    config=load_config(args.config)
    if config.params.model_version!="santa_fe_epistemic_feedback_v3":
        raise ValueError("compact runner requires Santa Fe v3")
    if args.command=="prepare":
        result=prepare(config)
    else:
        if args.cell_id is None:
            raise ValueError("--cell-id required")
        result=(run_cell(config,args.cell_id) if args.command=="run-cell" else
                theory_cell(config,args.cell_id))
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
