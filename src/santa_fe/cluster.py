"""Resumable cell-array execution for synthetic Santa Fe studies.

The commands here can run locally for checks or as Slurm array tasks. This
module does not submit jobs; ``plan`` writes a reviewable Cygnus submit script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shlex

import pandas as pd
import math
import numpy as np
from tqdm.auto import tqdm

from .config import Config, load_config
from .beta_cmi import produce_beta_outputs
from .llm_parallel import _analyze_group, _report, adapt_trajectories
from .measurements import final_summary, information_summary, susceptibility_summary
from .plotting import save_phase_diagrams
from .runner import run, save_trajectories


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v3_code_hashes(model_version: str = "santa_fe_epistemic_feedback_v3") -> dict[str, str]:
    source = Path(__file__).resolve().parent
    repository = source.parent
    def digest(paths):
        h = hashlib.sha256()
        for path in paths:
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()
    simulator = ([source / "state.py", source / "game.py", source / "v3_game.py"]
                 if model_version == "santa_fe_epistemic_feedback_v3" else
                 [source / "state.py", source / "game.py", source / "live_board.py",
                  source / "config.py", source / "runner.py"])
    return {
        "simulator_sha256": digest(simulator),
        "information_engine_sha256": digest([source / "llm_parallel.py", repository / "mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py"]),
    }


def _v3(config: Config) -> bool:
    return config.params.model_version in {"santa_fe_epistemic_feedback_v3", "santa_fe_live_board_v4"}


def _json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parquet(path: Path, data: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    data.to_parquet(temporary, index=False, compression="zstd", compression_level=6)
    os.replace(temporary, path)


def _cell_dir(config: Config, cell_id: int) -> Path:
    return config.results_dir / "cells" / f"cell-{cell_id:04d}"


def _load_manifest(config: Config) -> dict:
    path = config.results_dir / "execution_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"run `santa_fe.cluster prepare` first: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != _sha(config.path):
        raise ValueError("config changed after cluster preparation")
    if manifest["parameter_cells"] != len(config.cells):
        raise ValueError("cell count changed after cluster preparation")
    if _v3(config):
        for key, value in _v3_code_hashes(config.params.model_version).items():
            if manifest.get(key) != value:
                raise ValueError(f"v3 {key} changed after preparation; run prepare again")
    return manifest


def prepare(config: Config) -> dict:
    root = config.results_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "cells").mkdir(exist_ok=True)
    copy = root / "config.yaml"
    if copy.is_file() and copy.read_bytes() != config.path.read_bytes():
        raise ValueError(f"result root already belongs to a different config: {root}")
    copy.write_bytes(config.path.read_bytes())
    policy = config.raw.get("slurm", {})
    if not isinstance(policy, dict):
        raise ValueError("slurm policy must be a mapping")
    for stage in ("simulation", "information", "extras", "finalize"):
        value = policy.get(stage, {})
        if not isinstance(value, dict):
            raise ValueError(f"slurm.{stage} must be a mapping")
        if int(value.get("cpus_per_task", 1)) < 1:
            raise ValueError(f"slurm.{stage}.cpus_per_task must be positive")
        if stage != "finalize" and int(value.get("throttle", 1)) < 1:
            raise ValueError(f"slurm.{stage}.throttle must be positive")
    simulation_cpus = int(policy.get("simulation", {}).get("cpus_per_task", 4))
    information_cpus = int(policy.get("information", {}).get("cpus_per_task", 4))
    if config.processes > simulation_cpus:
        raise ValueError("experiment.processes exceeds Slurm simulation CPUs per task")
    if int(config.raw.get("analysis", {}).get("llm_parallel", {}).get("processes", 1)) > information_cpus:
        raise ValueError("llm_parallel.processes exceeds Slurm information CPUs per task")
    manifest = {"schema_version": 1, "config_path": str(config.path),
                "config_sha256": _sha(config.path), "results_dir": str(root),
                "parameter_cells": len(config.cells), "episodes_per_cell": config.episodes,
                "total_episodes": len(config.cells) * config.episodes,
                "round_rows_expected": len(config.cells) * config.episodes * (config.params.rounds + 1),
                "model_semantics": {key: getattr(config.params, key) for key in ("model_version", "persistence_clock",
                    "peer_posting_mode", "controller_message_mode", "board_clock", "controller_fact_selection")},
                "slurm": policy}
    if _v3(config):
        manifest.update(_v3_code_hashes(config.params.model_version))
    _json(root / "execution_plan.json", manifest)
    return manifest


def _valid_seal(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    try:
        seal = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if seal.get(key) != value:
                return False
        for name, digest in seal["files"].items():
            if not (path.parent / name).is_file() or _sha(path.parent / name) != digest:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_cell(config: Config, cell_id: int) -> dict:
    manifest = _load_manifest(config)
    if not 0 <= cell_id < len(config.cells):
        raise ValueError(f"cell ID outside 0..{len(config.cells)-1}")
    destination = _cell_dir(config, cell_id)
    destination.mkdir(parents=True, exist_ok=True)
    if config.params.model_version == "santa_fe_live_board_v4" and (destination / "retained" / "retention_complete.json").is_file():
        from .live_board_retention import verified_run_summary
        verified_run_summary(config, cell_id)
        return {"cell_id": cell_id, "status": "already_compacted", "path": str(destination)}
    seal_path = destination / "cell_complete.json"
    expected = {"config_sha256": manifest["config_sha256"], "cell_id": cell_id,
                "episodes": config.episodes, "round_rows": config.episodes * (config.params.rounds + 1)}
    if _v3(config):
        expected["simulator_sha256"] = manifest["simulator_sha256"]
    if _valid_seal(seal_path, expected):
        return {"cell_id": cell_id, "status": "already_complete", "path": str(destination)}
    cell_config = replace(config, cells=(config.cells[cell_id],), processes=min(config.processes, int(os.environ.get("SLURM_CPUS_PER_TASK", config.processes))))
    rounds, micro = run(cell_config)
    if rounds.seed.nunique() != config.episodes or len(rounds) != expected["round_rows"]:
        raise ValueError("generated cell has unexpected episode or round count")
    round_path = destination / "rounds.parquet"
    _parquet(round_path, rounds)
    files = {round_path.name: _sha(round_path)}
    if config.save_micro:
        micro_path = destination / "micro.parquet"
        _parquet(micro_path, micro)
        files[micro_path.name] = _sha(micro_path)
    _json(seal_path, {**expected, "files": files,
                      "model_version": config.cells[cell_id].params.model_version,
                      "beta_evidence": config.cells[cell_id].params.beta_evidence,
                      "beta_social": config.cells[cell_id].params.beta_social,
                      "beta_regime": config.cells[cell_id].beta_regime,
                      "rho": config.cells[cell_id].params.rho,
                      "budget_fraction": config.cells[cell_id].params.budget_fraction,
                      "budget": config.cells[cell_id].params.budget})
    return {"cell_id": cell_id, "status": "complete", "path": str(destination)}


def _sealed_rounds(config: Config, cell_id: int) -> pd.DataFrame:
    manifest = _load_manifest(config)
    destination = _cell_dir(config, cell_id)
    expected = {"config_sha256": manifest["config_sha256"], "cell_id": cell_id,
                "episodes": config.episodes, "round_rows": config.episodes * (config.params.rounds + 1)}
    if _v3(config):
        expected["simulator_sha256"] = manifest["simulator_sha256"]
    if not _valid_seal(destination / "cell_complete.json", expected):
        raise ValueError(f"cell {cell_id} is incomplete or its files changed")
    rounds = pd.read_parquet(destination / "rounds.parquet")
    if len(rounds) != expected["round_rows"] or rounds.seed.nunique() != config.episodes:
        raise ValueError(f"cell {cell_id} contents do not match its seal")
    return rounds


def aggregate_cells(config: Config) -> dict:
    if config.params.model_version == "santa_fe_live_board_v4":
        return aggregate_live_board_cells(config)
    _load_manifest(config)
    frames = [_sealed_rounds(config, cell.cell_id) for cell in tqdm(config.cells, desc="sealed cells")]
    rounds = pd.concat(frames, ignore_index=True)
    if rounds.duplicated(["cell_id", "seed", "round"]).any():
        raise ValueError("duplicate scientific round coordinates")
    micro_frames = []
    if config.save_micro:
        micro_frames = [pd.read_parquet(_cell_dir(config, cell.cell_id) / "micro.parquet") for cell in config.cells]
    micro = pd.concat(micro_frames, ignore_index=True) if micro_frames else pd.DataFrame()
    save_trajectories(config, rounds, micro)
    root = config.results_dir
    summary = root / "summaries"
    summary.mkdir(exist_ok=True)
    final = final_summary(rounds)
    final.to_csv(summary / "final_summary.csv", index=False)
    if config.information and config.params.model_version != "santa_fe_live_board_v4":
        information_summary(rounds, config.bins, config.coordinate).to_csv(summary / "information_summary.csv", index=False)
        susceptibility_summary(rounds, config.bins, config.coordinate).to_csv(summary / "susceptibility.csv", index=False)
    if config.save_plots:
        save_phase_diagrams(final, None, root / "plots" / "phase_diagrams")
    lines = ["# Santa Fe full-grid summary", "",
             f"Cells: {len(config.cells)}; episodes per cell: {config.episodes}; round rows: {len(rounds)}.",
             "", "Phase diagrams are in `plots/phase_diagrams/`; information diagrams are added after information aggregation.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _json(root / "cells_aggregation.json", {"config_sha256": _sha(config.path),
                                            "cells": len(config.cells), "episodes": len(config.cells) * config.episodes,
                                            "round_rows": len(rounds), "round_trajectory_sha256": _sha(root / "trajectories" / "round_trajectories.parquet")})
    return {"cells": len(config.cells), "episodes": len(config.cells) * config.episodes,
            "round_rows": len(rounds), "results_dir": str(root)}


def aggregate_live_board_cells(config: Config) -> dict:
    """Combine sealed v4 cells or their independently sealed per-run summaries."""
    _load_manifest(config)
    summaries=[]
    run_tables=[]
    metric_names=("late_target_share","late_truth_share","late_kappa_plus",
                  "late_kappa_minus","late_phi_plus","late_overloaded_share",
                  "late_target_temporal_sd","late_board_target_vote_fraction",
                  "late_board_positive_fact_fraction","late_board_message_count",
                  "late_unique_controller_fact_ids",
                  "final_target_share","final_truth_share","final_target_majority",
                  "intervention_frequency","cumulative_spending")
    for cell in tqdm(config.cells,desc="sealed live-board cells"):
        retained=_cell_dir(config,cell.cell_id)/"retained"/"per_run_summary.parquet"
        if retained.is_file():
            from .live_board_retention import verified_run_summary
            table=verified_run_summary(config,cell.cell_id)
        else:
            rounds=_sealed_rounds(config,cell.cell_id)
            table=_live_board_run_table(config,cell,rounds)
        run_tables.append(table)
        for name in metric_names:
            values=table[name]
            sd=float(values.std(ddof=1)) if len(values)>1 else math.nan
            se=sd/math.sqrt(len(values)) if len(values)>1 else math.nan
            halfwidth=1.96*se if math.isfinite(se) else math.nan
            if name.startswith("late_"):
                window=("intervention_rounds_41_to_60" if "board" in name or
                        "controller_fact" in name else "X_41_to_X_60")
            elif name.startswith("final_"):
                window="terminal_X_60"
            else:
                window="intervention_rounds_1_to_60"
            summaries.append({"cell_id":cell.cell_id,"controller_target":cell.params.controller_target,
                "beta_regime":cell.beta_regime,"rho":cell.params.rho,"budget":cell.params.budget,
                "q":cell.params.q,"q_c":round(cell.params.sensing_fraction*cell.params.N),
                "metric":name,"window":window,
                "estimate":float(values.mean()),"independent_episodes":len(values),
                "sd_across_episodes":sd,"se":se,
                "ci_low":float(values.mean()-halfwidth),
                "ci_high":float(values.mean()+halfwidth),
                "predeclared_halfwidth_target":0.02 if name=="late_target_share" else math.nan,
                "precision_sufficient":bool(halfwidth<=0.02) if name=="late_target_share" else None})
    summary=config.results_dir/"summaries"
    summary.mkdir(exist_ok=True)
    runs=pd.concat(run_tables,ignore_index=True)
    _parquet(summary/"per_run_summary.parquet",runs)
    pd.DataFrame(summaries).to_csv(summary/"cell_metric_summary.csv",index=False)
    baselines={tuple(key):group for key,group in runs.loc[runs.budget==0].groupby(
        ["beta_regime","controller_target","rho","q","q_c"])}
    gains=[]
    for cell_id,group in runs.groupby("cell_id"):
        first=group.iloc[0]
        key=(first.beta_regime,first.controller_target,first.rho,first.q,first.q_c)
        baseline=baselines.get(key)
        if baseline is None:continue
        n=len(group);n0=len(baseline)
        x=group.late_target_share.to_numpy(dtype=float)
        x0=baseline.late_target_share.to_numpy(dtype=float)
        cost=group.cumulative_spending.to_numpy(dtype=float)
        gain=float(x.mean()-x0.mean())
        var_gain=float(np.var(x,ddof=1)/n+np.var(x0,ddof=1)/n0) if n>1 and n0>1 else math.nan
        se_gain=math.sqrt(var_gain) if math.isfinite(var_gain) else math.nan
        if int(first.budget)==0:
            gain=0.0
            var_gain=0.0
            se_gain=0.0
        cost_mean=float(cost.mean())
        cost_se=float(np.std(cost,ddof=1)/math.sqrt(n)) if n>1 else math.nan
        stable=bool(cost_mean>0 and cost_mean>2*cost_se)
        ratio=gain/cost_mean if stable else math.nan
        if stable:
            covariance=float(np.cov(x,cost,ddof=1)[0,1]/n)
            ratio_variance=(var_gain/cost_mean**2+
                gain**2*np.var(cost,ddof=1)/(n*cost_mean**4)-
                2*gain*covariance/cost_mean**3)
            ratio_se=math.sqrt(max(0,ratio_variance))
        else:ratio_se=math.nan
        gains.append({"cell_id":cell_id,"baseline_cell_id":int(baseline.cell_id.iloc[0]),
            "beta_regime":first.beta_regime,"controller_target":first.controller_target,
            "rho":first.rho,"q":first.q,"q_c":first.q_c,"budget":first.budget,
            "gain_late_target_share":gain,"gain_se_unpaired":se_gain,
            "gain_ci_low":gain-1.96*se_gain,"gain_ci_high":gain+1.96*se_gain,
            "actual_mean_cumulative_spending":cost_mean,"cost_se":cost_se,
            "gain_per_actual_post":ratio,"gain_per_post_se_delta":ratio_se,
            "ratio_stable_denominator":stable,"replicates_paired_across_budget":False,
            "interpretation":"descriptive finite-horizon ratio; not an efficiency bound"})
    pd.DataFrame(gains).to_csv(summary/"gain_vs_b0.csv",index=False)
    _json(config.results_dir/"cells_aggregation.json",{"model_version":config.params.model_version,
        "config_sha256":_sha(config.path),"cells":len(config.cells),
        "independent_episodes":len(config.cells)*config.episodes,
        "raw_location":"cells/cell-NNNN/{rounds,micro}.parquet or retained examples after compaction",
        "summary":"summaries/cell_metric_summary.csv",
        "precision_insufficient_late_support_cells":int(sum(
            row["metric"]=="late_target_share" and not row["precision_sufficient"]
            for row in summaries))})
    return {"cells":len(config.cells),"episodes":len(config.cells)*config.episodes,
            "results_dir":str(config.results_dir)}


def _live_board_run_table(config: Config, cell, rounds: pd.DataFrame) -> pd.DataFrame:
    """Compute episode-level finite-horizon metrics before optional raw retention."""
    if rounds.duplicated(["seed","round"]).any():
        raise ValueError(f"duplicate scientific round coordinates in cell {cell.cell_id}")
    active=rounds.loc[rounds.time_stage=="pre_action"]
    # Outcomes of intervention rounds 41..60 are X_41..X_60.
    late=rounds.loc[rounds["round"].between(41,60)]
    board_late=active.loc[active["round"].between(40,59)].copy()
    composition=[json.loads(value) for value in board_late.B_total_counts_json]
    plus_votes=np.array([sum(count for key,count in item.items() if key.startswith("+1_"))
                         for item in composition],dtype=float)
    minus_votes=np.array([sum(count for key,count in item.items() if key.startswith("-1_"))
                          for item in composition],dtype=float)
    positive_facts=np.array([sum(count for key,count in item.items() if key.endswith("_+1"))
                             for item in composition],dtype=float)
    board_size=plus_votes+minus_votes
    board_late["target_board_fraction"]=(plus_votes if cell.params.controller_target==1
                                          else minus_votes)/board_size
    board_late["positive_fact_fraction"]=positive_facts/board_size
    board_late["board_size"]=board_size
    board_grouped=board_late.groupby("seed")
    terminal=rounds.loc[rounds.time_stage=="terminal"].set_index("seed")
    if len(active)!=config.episodes*config.params.rounds or len(terminal)!=config.episodes:
        raise ValueError(f"v4 cell {cell.cell_id} has an incomplete round clock")
    grouped=late.groupby("seed")
    table=pd.DataFrame({
        "late_target_share":grouped.target_share.mean(),
        "late_truth_share":grouped.truth_share.mean(),
        "late_kappa_plus":grouped.kappa_plus.mean(),
        "late_kappa_minus":grouped.kappa_minus.mean(),
        "late_phi_plus":grouped.phi_plus.mean(),
        "late_overloaded_share":grouped.overloaded_share.mean(),
        "late_target_temporal_sd":grouped.target_share.std(ddof=1),
        "late_board_target_vote_fraction":board_grouped.target_board_fraction.mean(),
        "late_board_positive_fact_fraction":board_grouped.positive_fact_fraction.mean(),
        "late_board_message_count":board_grouped.board_size.mean(),
        "late_unique_controller_fact_ids":board_grouped.unique_controller_fact_ids.mean(),
        "final_target_share":terminal.target_share,
        "final_truth_share":terminal.truth_share,
        "final_target_majority":(terminal.target_share>.5).astype(float),
        "intervention_frequency":active.groupby("seed").controller_U.mean(),
        "cumulative_spending":active.groupby("seed").budget_used.sum(),
    }).reset_index()
    table.insert(0,"cell_id",cell.cell_id)
    table["controller_target"]=cell.params.controller_target
    table["beta_regime"]=cell.beta_regime
    table["rho"]=cell.params.rho
    table["budget"]=cell.params.budget
    table["q"]=cell.params.q
    table["q_c"]=round(cell.params.sensing_fraction*cell.params.N)
    return table


def run_information_cell(config: Config, cell_id: int) -> dict:
    _load_manifest(config)
    destination = _cell_dir(config, cell_id) / "information"
    if config.params.model_version == "santa_fe_live_board_v4" and (_cell_dir(config, cell_id) / "retained" / "retention_complete.json").is_file():
        from .live_board_retention import verified_run_summary
        verified_run_summary(config, cell_id)
        original = json.loads((_cell_dir(config, cell_id) / "cell_complete.json").read_text())
        expected = {"config_sha256": _sha(config.path),
                    "source_sha256": original["files"]["rounds.parquet"],
                    "cell_id": cell_id,
                    "information_engine_sha256": _v3_code_hashes(config.params.model_version)["information_engine_sha256"]}
        if not _valid_seal(destination / "information_complete.json", expected):
            raise ValueError(f"compacted information cell {cell_id} has an invalid seal")
        return {"cell_id": cell_id, "status": "already_compacted", "path": str(destination)}
    rounds = _sealed_rounds(config, cell_id)
    source = _cell_dir(config, cell_id) / "rounds.parquet"
    settings = config.raw.get("analysis", {}).get("llm_parallel", {})
    statistics = tuple(settings.get("statistics", ()))
    if not statistics:
        raise ValueError("analysis.llm_parallel.statistics is required")
    bootstrap = int(settings.get("bootstrap_resamples", 1000))
    permutations = int(settings.get("null_permutations", 1000))
    confidence = float(settings.get("confidence", .95))
    seed = int(settings.get("seed", config.seed))
    processes = min(int(settings.get("processes", 1)), int(os.environ.get("SLURM_CPUS_PER_TASK", settings.get("processes", 1))))
    destination.mkdir(exist_ok=True)
    seal_path = destination / "information_complete.json"
    expected = {"config_sha256": _sha(config.path), "source_sha256": _sha(source), "cell_id": cell_id}
    if _v3(config):
        expected["information_engine_sha256"] = _v3_code_hashes(config.params.model_version)["information_engine_sha256"]
    if _valid_seal(seal_path, expected):
        return {"cell_id": cell_id, "status": "already_complete", "path": str(destination)}
    events = adapt_trajectories(rounds, bins=config.bins)
    groups: dict[int, list] = {}
    for event in events:
        groups.setdefault(event.round_index, []).append(event)
    payloads = [(cell_id, "pooled", None, events, statistics, bootstrap, permutations, confidence, seed)]
    if settings.get("per_round", True):
        per_round_statistics=tuple(settings.get("per_round_statistics",statistics))
        payloads += [(cell_id, "per_round", time, rows, per_round_statistics, bootstrap, permutations, confidence, seed)
                     for time, rows in sorted(groups.items())]
    estimates, nulls = [], []
    if processes == 1:
        iterator = map(_analyze_group, payloads)
        for estimate_rows, null_rows in tqdm(iterator, total=len(payloads), desc=f"information cell {cell_id}"):
            estimates.extend(estimate_rows)
            nulls.extend(null_rows)
    else:
        with ProcessPoolExecutor(max_workers=processes, mp_context=get_context("spawn")) as pool:
            iterator = pool.map(_analyze_group, payloads, chunksize=1)
            for estimate_rows, null_rows in tqdm(iterator, total=len(payloads), desc=f"information cell {cell_id}"):
                estimates.extend(estimate_rows)
                nulls.extend(null_rows)
    estimate_path = destination / "estimates.parquet"
    null_path = destination / "nulls.parquet"
    _parquet(estimate_path, pd.DataFrame(estimates))
    _parquet(null_path, pd.DataFrame(nulls))
    _json(seal_path, {**expected, "estimates": len(estimates), "null_draws": len(nulls),
                      "files": {estimate_path.name: _sha(estimate_path), null_path.name: _sha(null_path)}})
    return {"cell_id": cell_id, "status": "complete", "estimates": len(estimates),
            "null_draws": len(nulls), "path": str(destination)}


def aggregate_information(config: Config) -> dict:
    _load_manifest(config)
    estimates = []
    total_null_draws = 0
    for cell in tqdm(config.cells, desc="sealed information cells"):
        cell_id = cell.cell_id
        source = _cell_dir(config, cell_id) / "rounds.parquet"
        destination = _cell_dir(config, cell_id) / "information"
        if source.is_file():
            source_hash = _sha(source)
        else:
            from .live_board_retention import verified_run_summary
            verified_run_summary(config, cell_id)
            original = json.loads((_cell_dir(config, cell_id) / "cell_complete.json").read_text())
            source_hash = original["files"]["rounds.parquet"]
        expected = {"config_sha256": _sha(config.path), "source_sha256": source_hash, "cell_id": cell_id}
        if _v3(config):
            expected["information_engine_sha256"] = _v3_code_hashes(config.params.model_version)["information_engine_sha256"]
        seal = destination / "information_complete.json"
        if not _valid_seal(seal, expected):
            raise ValueError(f"information cell {cell_id} is incomplete or changed")
        payload = json.loads(seal.read_text(encoding="utf-8"))
        total_null_draws += int(payload["null_draws"])
        estimates.append(pd.read_parquet(destination / "estimates.parquet"))
    table = pd.concat(estimates, ignore_index=True)
    output = config.results_dir / "information" / "llm_parallel"
    output.mkdir(parents=True, exist_ok=True)
    _parquet(output / "round_information_estimates.parquet", table)
    table.to_csv(output / "round_information_estimates.csv", index=False)
    _report(table, config, output)
    if config.params.model_version != "santa_fe_live_board_v4":
        final = pd.read_csv(config.results_dir / "summaries" / "final_summary.csv")
        if config.save_plots:
            save_phase_diagrams(final, table.loc[table.scope == "pooled"], config.results_dir / "plots" / "phase_diagrams")
    if config.params.model_version != "santa_fe_live_board_v4" and config.raw.get("analysis", {}).get("beta_cmi", {}).get("enabled", False):
        rounds = pd.read_parquet(config.results_dir / "trajectories" / "round_trajectories.parquet")
        susceptibility = pd.read_csv(config.results_dir / "summaries" / "susceptibility.csv")
        produce_beta_outputs(config, rounds, final, susceptibility, table)
    _json(output / "analysis_config.json", {"config_sha256": _sha(config.path),
                                            "source": "sealed per-cell information outputs",
                                            "cells": len(config.cells), "estimates": len(table),
                                            "null_draws": total_null_draws})
    (output / "analysis_recipe.yaml").write_bytes(config.path.read_bytes())
    return {"cells": len(config.cells), "estimates": len(table), "null_draws": total_null_draws,
            "results_dir": str(output)}


def _resources(policy: dict, stage: str) -> tuple[int, str, str, int]:
    defaults = {"simulation": (4, "8G", "01:00:00", 12),
                "information": (4, "12G", "04:00:00", 12),
                "extras": (1, "8G", "02:00:00", 16),
                "finalize": (8, "32G", "02:00:00", 1)}
    section = policy.get(stage, {})
    cpus, memory, limit, throttle = defaults[stage]
    return (int(section.get("cpus_per_task", cpus)), str(section.get("memory", memory)),
            str(section.get("time_limit", limit)), int(section.get("throttle", throttle)))


def write_cygnus_commands(config: Config, *, python: str, repository_root: str) -> Path:
    manifest = prepare(config)
    policy = manifest["slurm"]
    root = config.results_dir
    config_arg = shlex.quote(str(config.path))
    py = shlex.quote(python)
    repo = shlex.quote(repository_root)
    logs = shlex.quote(str(root / "logs"))
    cells = len(config.cells)
    def sbatch(stage: str, job_name: str, command: str, *, array_size: int | None = None,
               dependency: str | None = None) -> str:
        cpus, memory, limit, throttle = _resources(policy, stage)
        array_arg = f" --array=0-{array_size-1}%{throttle}" if array_size is not None else ""
        index = "%A_%a" if array_size is not None else "%j"
        dependent = f" --dependency=afterok:${dependency}" if dependency else ""
        return (f"sbatch --parsable --job-name={job_name}{array_arg}{dependent} --cpus-per-task={cpus} "
                f"--mem={shlex.quote(memory)} --time={shlex.quote(limit)} "
                f"--output={logs}/{job_name}-{index}.out --error={logs}/{job_name}-{index}.err "
                f"--wrap={shlex.quote(command)}")
    base = f"cd {repo} && export PYTHONPATH={repo}/src && export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
    cell_finalize = base + f"{py} -m santa_fe.cluster aggregate-cells --config {config_arg}"
    info_finalize = base + f"{py} -m santa_fe.cluster aggregate-information --config {config_arg}"
    commands = ["#!/usr/bin/env bash", "set -euo pipefail",
                "# Slurm MaxArraySize=1001: chunk every cell stage into at most 1000 indices.",
                "# Review paths/resources before running; no provider calls are made."]
    def array_stage(stage: str, job_name: str, operation: str, prefix: str,
                    dependency: str | None) -> str:
        preceding=dependency
        for index,start in enumerate(range(0,cells,1000)):
            count=min(1000,cells-start)
            variable=f"{prefix}_{index}"
            command=(base+f"{py} -m santa_fe.cluster {operation} --config {config_arg} "
                     f"--cell-id $((SLURM_ARRAY_TASK_ID+{start}))")
            commands.append(f"{variable}=$({sbatch(stage,job_name,command,array_size=count,dependency=preceding)})")
            preceding=variable
        if preceding is None:raise ValueError("empty cell array")
        return preceding
    sim_last=array_stage("simulation","santa-fe-sim","run-cell","sim_job",None)
    commands.append(f"cells_job=$({sbatch('finalize','santa-fe-cells',cell_finalize,dependency=sim_last)})")
    info_last=array_stage("information","santa-fe-info","run-information-cell","info_job","cells_job")
    commands.append(f"final_job=$({sbatch('finalize','santa-fe-final',info_finalize,dependency=info_last)})")
    if config.params.model_version == "santa_fe_live_board_v4":
        maps = base + f"{py} -m santa_fe.cluster plot-live-board --config {config_arg}"
        extra_last=array_stage("extras","santa-fe-extra","postprocess-cell","extra_job","final_job")
        commands += [f"maps_job=$({sbatch('finalize','santa-fe-maps',maps,dependency=extra_last)})",
                     f'printf "simulation_last=%s cell_aggregation=%s information_last=%s information_aggregation=%s extras_last=%s maps=%s\\n" "${sim_last}" "$cells_job" "${info_last}" "$final_job" "${extra_last}" "$maps_job"']
    else:
        commands.append(f'printf "simulation_last=%s cell_aggregation=%s information_last=%s final=%s\\n" "${sim_last}" "$cells_job" "${info_last}" "$final_job"')
    commands.append("")
    path = root / "submit_cygnus.sh"
    path.write_text("\n".join(commands), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run-cell", "aggregate-cells", "run-information-cell", "aggregate-information", "postprocess-cell", "plot-live-board", "plan-cygnus"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell-id", type=int)
    parser.add_argument("--python", help="absolute Cygnus Python path for the plan")
    parser.add_argument("--repository-root", help="absolute Cygnus checkout path for the plan")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.operation in {"run-cell", "run-information-cell", "postprocess-cell"}:
        cell_id = args.cell_id if args.cell_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        if args.operation == "postprocess-cell":
            from .live_board_postprocess import run_cell as postprocess_cell
            result = postprocess_cell(str(config.path),cell_id)
        else:
            result = run_cell(config, cell_id) if args.operation == "run-cell" else run_information_cell(config, cell_id)
    elif args.operation == "prepare":
        result = prepare(config)
    elif args.operation == "aggregate-cells":
        result = aggregate_cells(config)
    elif args.operation == "aggregate-information":
        result = aggregate_information(config)
    elif args.operation == "plot-live-board":
        from .live_board_plots import plot_maps
        result = plot_maps(str(config.path))
    else:
        if not args.python or not args.repository_root:
            parser.error("plan-cygnus needs --python and --repository-root")
        script = write_cygnus_commands(config, python=args.python, repository_root=args.repository_root)
        result = {"script": str(script), "results_dir": str(config.results_dir),
                  "cells": len(config.cells), "episodes": len(config.cells) * config.episodes}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
