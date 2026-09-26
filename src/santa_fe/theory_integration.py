"""Adapt retained Santa Fe v3 trajectories to the supplied reduced theory runner.

This module never changes or regenerates MICRO trajectories. Each comparison
uses actual round-0 population/board records and the runner's own CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from santa_fe_theory.core import VERSION, project_snapshot
from .config import load_config
from .v3_game import replay_next_day, _json
from .v3_validation import sensor_averaged_propensity

VARIANTS = {
    "paper_reduced": ("with_replacement", "paper_diagonal"),
    "sampling_matched_reduced": ("without_replacement", "paper_diagonal"),
    "sampling_clock_corrected_reduced": ("without_replacement", "fixed_clock"),
}


def _records(value):
    return json.loads(value) if isinstance(value, str) else value


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _category_counts(board):
    # Runner's canonical project_snapshot validates IDs and sign alignment.
    counts = [0, 0, 0, 0]
    for message in board:
        vote = int(message["vote"])
        fact = message["fact_id"]
        counts[(0 if vote == 1 else 2) + (fact is None)] += 1
    return counts


def _project(row, board_field):
    agents = _records(row.agent_states_json)
    weights = _records(row.fact_weights_json)
    board = _records(getattr(row, board_field))
    projected = project_snapshot(agents, dict(enumerate(weights)), board, N=int(row.N))
    return projected


def _runner(command: str, **options):
    argv = [sys.executable, "-m", "santa_fe_theory", command]
    for key, value in options.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(argv, check=True)


def export_cell(config, rounds: pd.DataFrame, cell_id: int, out: Path, *, initial_blocks: int,
                theory_replicas: int):
    """Export one physical cell with strict initial and stage alignment."""
    if config.cells[cell_id].params.model_version != "santa_fe_epistemic_feedback_v3":
        raise ValueError("theory adapter requires Santa Fe v3")
    data = rounds.loc[rounds.cell_id == cell_id].sort_values(["seed", "round"])
    if data.empty:
        raise ValueError(f"missing cell {cell_id}")
    seeds = sorted(map(int, data.seed.unique()))[:initial_blocks]
    if len(seeds) < initial_blocks:
        raise ValueError(f"cell {cell_id} has only {len(seeds)} saved initial blocks")
    data = data.loc[data.seed.isin(seeds)]
    first = data.loc[data["round"] > 0].iloc[0]
    weights = _records(first.fact_weights_json)
    Fplus = weights.count(1)
    params = config.cells[cell_id].params
    resolved = {"model_version": "santa_fe_epistemic_feedback_v3",
                "N": int(first.N), "F_plus": Fplus, "F_minus": weights.count(-1),
                "q": int(first.q), "q_c": int(first.controller_sensed_messages),
                "b": int(first.controller_requested_budget), "rho": float(first.rho),
                "beta_evidence": float(first.beta_evidence), "beta_social": float(first.beta_social),
                "policy_beta": float(first.policy_beta),
                "policy_threshold": float(first.policy_threshold),
                "controller_target": int(first.controller_target)}
    if resolved["b"] != params.budget or resolved["N"] != params.N:
        raise ValueError("saved integer budget/population does not match resolved cell")
    if resolved["q_c"] < 1 or resolved["F_plus"] + resolved["F_minus"] != params.F:
        raise ValueError("invalid saved sensor/fact counts")
    rounds_expected = params.rounds
    initials, simulation, blocks = [], [], []
    for seed in seeds:
        panel = data.loc[data.seed == seed].sort_values("round")
        if panel["round"].tolist() != list(range(rounds_expected + 1)):
            raise ValueError(f"cell {cell_id} seed {seed} lacks a complete round panel")
        initial = panel.iloc[0]
        projected = _project(initial, "front_page_json")
        for key in ("N", "F_plus", "F_minus"):
            if projected[key] != resolved[key]:
                raise ValueError(f"round-0 projected {key} differs from cell parameter")
        if not np.isclose(projected["x"], initial.truth_share, atol=1e-12):
            raise ValueError("round-0 truth share mismatch")
        snapshot_id = f"cell{cell_id:04d}_seed{seed}"
        snapshot = {"snapshot_id": snapshot_id, "round": 0,
                    "x": projected["x"], "kappa_plus": projected["kappa_plus"],
                    "kappa_minus": projected["kappa_minus"],
                    "front_page_counts": projected["front_page_counts"]}
        initials.append(snapshot)
        blocks.append({"initial_id": snapshot_id, "simulator_seed": seed,
                       "theory_replicas": theory_replicas})
        for row in panel.itertuples():
            if _records(row.fact_weights_json).count(1) != Fplus:
                raise ValueError("fact split changed within an episode")
            d = int(row.round)
            if d == 0:
                peer_target = action = effective = posts = ""
                peer_counts = ["", "", "", ""]
                front = snapshot["front_page_counts"]
            else:
                peer = _records(row.peer_board_json)
                peer_counts = _category_counts(peer)
                peer_target = sum(m["vote"] == resolved["controller_target"] for m in peer) / resolved["N"]
                if not np.isclose(peer_target, row.peer_board_target_count / resolved["N"]):
                    raise ValueError("peer target count is inconsistent with saved messages")
                action = int(row.controller_U)
                effective = int(row.controller_effective_U)
                posts = int(row.budget_used)
                front = _category_counts(_records(row.next_board_json))
                if posts != resolved["b"] * action:
                    raise ValueError("action/post budget mismatch")
            simulation.append({"model": "MICRO", "initial_id": snapshot_id, "replica": 0,
                "round": d, "x": float(row.truth_share), "target_share": float(row.target_share),
                "kappa_plus": float(row.kappa_plus), "kappa_minus": float(row.kappa_minus),
                "peer_target_share": peer_target, "action_next": action,
                "effective_action_next": effective, "posts_next": posts,
                **{f"front_next_{key}": value for key, value in zip(
                    ("plus_fact", "plus_empty", "minus_fact", "minus_empty"), front)},
                **{f"peer_{key}": value for key, value in zip(
                    ("plus_fact", "plus_empty", "minus_fact", "minus_empty"), peer_counts)}})
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "initials.json", {"schema_version": 1, "kind": "initial", "snapshots": initials})
    _write_csv(out / "simulator.csv", simulation)
    _write_json(out / "simulation_manifest.json", {
        "parameters": resolved, "sampling": "without_replacement",
        "round_stage": "end_of_day_before_controller_effect", "data_kind": "simulation",
        "initial_snapshots": initials})
    _write_json(out / "block_manifest.json", {"cell_id": cell_id, "blocks": blocks,
        "stage": "round 0 initial front page; round d post-day population and U_d for next board",
        "initial_fact_redundancy": params.initial_fact_redundancy,
        "truth_fact_fraction_input": params.truth_fact_fraction,
        "model_version": params.model_version})
    return resolved, data


def _wasserstein_1d(a, b):
    a = np.sort(np.asarray(a, dtype=float)); b = np.sort(np.asarray(b, dtype=float))
    support = np.unique(np.concatenate((a, b)))
    if len(support) < 2:
        return 0.0
    return float(np.sum(np.abs(np.searchsorted(a, support[:-1], side="right") / len(a) -
                               np.searchsorted(b, support[:-1], side="right") / len(b)) *
                        np.diff(support)))


def _selected_distribution(theory: pd.DataFrame, micro: pd.DataFrame,
                           common: list[str], out: Path, *, draws: int = 399):
    """Paired initial-block bootstrap for selected empirical distribution distances."""
    horizon = int(micro["round"].max())
    rounds = sorted({1, max(1, horizon//2), horizon})
    rows = []
    rng = np.random.default_rng(20260927)
    for d in rounds:
        for metric in ("target_share", "kappa_plus", "kappa_minus"):
            t_by = [theory.loc[(theory.initial_id == block) &
                               (theory["round"] == d), metric].to_numpy(dtype=float)
                    for block in common]
            m_by = [micro.loc[(micro.initial_id == block) &
                              (micro["round"] == d), metric].to_numpy(dtype=float)
                    for block in common]
            if any(len(a)==0 or len(b)==0 for a,b in zip(t_by,m_by)):
                raise ValueError("distribution comparison has an incomplete block panel")
            all_theory = np.concatenate(t_by); all_micro = np.concatenate(m_by)
            distance = _wasserstein_1d(all_theory, all_micro)
            variance_theory = float(np.var(all_theory, ddof=1)) if len(all_theory)>1 else float("nan")
            variance_micro = float(np.var(all_micro, ddof=1)) if len(all_micro)>1 else float("nan")
            bootstrap = []
            if len(common)>1:
                for _ in range(draws):
                    indices = rng.integers(len(common), size=len(common))
                    bootstrap.append(_wasserstein_1d(np.concatenate([t_by[i] for i in indices]),
                                                     np.concatenate([m_by[i] for i in indices])))
            rows.append({"round":d,"metric":metric,"n_initial_blocks":len(common),
                         "theory_replicas_per_block":len(t_by[0]),
                         "micro_replicas_per_block":len(m_by[0]),
                         "wasserstein_1":distance,
                         "bootstrap_ci95_low":float(np.quantile(bootstrap,.025)) if bootstrap else "",
                         "bootstrap_ci95_high":float(np.quantile(bootstrap,.975)) if bootstrap else "",
                         "bootstrap_draws":draws if bootstrap else 0,
                         "theory_variance":variance_theory,
                         "micro_variance":variance_micro,
                         "variance_ratio_theory_over_micro":variance_theory/variance_micro
                            if variance_micro>0 else "",
                         "bootstrap_unit":"paired_initial_block"})
    _write_csv(out / "selected_distribution_bootstrap.csv", rows)


def _extra_comparison(theory_dir: Path, sim_csv: Path, compare_dir: Path, *, late_start: int):
    theory = pd.read_csv(theory_dir / "trajectories.csv")
    micro = pd.read_csv(sim_csv)
    common = sorted(set(theory.initial_id) & set(micro.initial_id))
    _selected_distribution(theory, micro, common, compare_dir)
    rows = []
    for d in sorted(micro["round"].unique()):
        for metric in ("target_share", "kappa_plus", "kappa_minus"):
            a = [] ; b = []
            for block in common:
                a.append(theory.loc[(theory.initial_id == block) & (theory["round"] == d), metric].mean())
                b.append(micro.loc[(micro.initial_id == block) & (micro["round"] == d), metric].mean())
            delta = np.asarray(a)-np.asarray(b)
            theory_values = theory.loc[theory["round"] == d, metric].to_numpy()
            micro_values = micro.loc[micro["round"] == d, metric].to_numpy()
            rows.append({"round": d, "metric": metric, "n_blocks": len(common),
                "theory_minus_micro": float(delta.mean()),
                "se_paired_blocks": float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else "",
                "theory_variance": float(np.var(theory_values, ddof=1)) if len(theory_values)>1 else "",
                "micro_variance": float(np.var(micro_values, ddof=1)) if len(micro_values)>1 else "",
                "wasserstein_1": _wasserstein_1d(theory_values, micro_values)})
    _write_csv(compare_dir / "distribution_diagnostics.csv", rows)
    selected = [r for r in rows if late_start <= r["round"] <= int(micro["round"].max())]
    late = []
    for metric in ("target_share", "kappa_plus", "kappa_minus"):
        values = np.array([r["theory_minus_micro"] for r in selected if r["metric"] == metric])
        block_theory = np.array([
            theory.loc[(theory.initial_id == block) & (theory["round"] >= late_start)]
                  .groupby("round")[metric].mean().mean()
            for block in common])
        block_micro = np.array([micro.loc[(micro.initial_id == block) &
            (micro["round"] >= late_start), metric].mean() for block in common])
        paired = block_theory-block_micro
        se = float(paired.std(ddof=1)/np.sqrt(len(common))) if len(common)>1 else ""
        late.append({"metric": metric, "round_start": late_start,
                     "round_end": int(micro["round"].max()), "n_rounds": len(values),
                     "late_bias": float(values.mean()),
                     "late_rmse": float(np.sqrt(np.mean(values*values))),
                     "micro_late_mean": float(block_micro.mean()),
                     "theory_late_mean": float(block_theory.mean()),
                     "paired_block_se": se,
                     "paired_ci95_low": float(paired.mean()-1.96*se) if se != "" else "",
                     "paired_ci95_high": float(paired.mean()+1.96*se) if se != "" else ""})
    _write_csv(compare_dir / "late_window_errors.csv", late)
    return late


def _branch_records(data: pd.DataFrame, resolved: dict, out: Path, *, snapshots_per_cell: int):
    candidates = data.loc[(data["round"] > 0) & (data["round"] < data["round"].max())]
    selected = candidates.sort_values(["seed", "round"]).groupby("seed", sort=True).head(1).head(snapshots_per_cell)
    if selected.empty:
        raise ValueError("forced next-day branches require a saved pre-action round before the horizon")
    snapshots, source = [], []
    for row in selected.itertuples():
        projection = _project(row, "peer_board_json")
        if any(projection[key] != resolved[key] for key in ("N", "F_plus", "F_minus")):
            raise ValueError("pre-action projection does not match cell parameters")
        if sum(projection["front_page_counts"]) != resolved["N"]:
            raise ValueError("closed peer board must have N messages")
        snapshot_id = f"cell{int(row.cell_id):04d}_seed{int(row.seed)}_round{int(row.round)}"
        snapshots.append({"snapshot_id": snapshot_id, "round": int(row.round),
                          "x": projection["x"], "kappa_plus": projection["kappa_plus"],
                          "kappa_minus": projection["kappa_minus"],
                          "peer_counts": projection["front_page_counts"]})
        source.append(row)
    _write_json(out / "pre_action.json", {"schema_version": 1,
                                         "kind": "pre_action", "snapshots": snapshots})
    return snapshots, source


def _micro_branch(source, snapshots, resolved, config, *, pairs: int, seed: int, out: Path):
    rows, summaries, replay = [], [], []
    cell_id = int(source[0].cell_id)
    params = config.cells[cell_id].params
    all_data = pd.read_parquet(config.results_dir / "trajectories" / "round_trajectories.parquet")
    successor = {(int(r.cell_id), int(r.seed), int(r.round)): r for r in all_data.itertuples()}
    for snapshot, row in zip(snapshots, source):
        agents = _records(row.agent_states_json)
        weights = _records(row.fact_weights_json)
        peer = _records(row.peer_board_json)
        factual_board = _records(row.next_board_json)
        next_agents, next_peer = replay_next_day(agents, factual_board, weights, params,
            _records(row.rng_state_next_day_json), int(row.round)+1)
        saved = successor[(cell_id, int(row.seed), int(row.round)+1)]
        ok = (_json(next_agents) == saved.agent_states_json and
              _json(next_peer) == saved.peer_board_json)
        replay.append({"snapshot_id": snapshot["snapshot_id"], "factual_replay_match": ok})
        if not ok:
            raise ValueError(f"factual next-day replay failed for {snapshot['snapshot_id']}")
        pool = _records(row.controller_fact_pool_json)
        differences = []
        for pair in range(pairs):
            streams = np.random.SeedSequence([seed, cell_id, int(row.seed), int(row.round), pair]).spawn(2)
            downstream = np.random.default_rng(streams[0]).bit_generator.state
            fact_rng = np.random.default_rng(streams[1])
            forced = [{"message_id": f"forced-{pair}-{j}", "author": params.N,
                       "vote": params.controller_target, "fact_id": int(fact_rng.choice(pool)),
                       "fact_sign": params.controller_target, "source": "controller"}
                      for j in range(resolved["b"])]
            targets = []
            for action, board in ((0, peer), (1, peer+forced)):
                next_agents, _ = replay_next_day(agents, board, weights, params,
                                                 downstream, int(row.round)+1)
                target = sum(a["vote"] == resolved["controller_target"] for a in next_agents)/params.N
                targets.append(target)
                rows.append({"snapshot_id": snapshot["snapshot_id"], "pair": pair,
                             "forced_action": action, "pre_action_round": int(row.round),
                             "outcome_round": int(row.round)+1, "target_next": target,
                             "x_next": sum(a["vote"] == 1 for a in next_agents)/params.N})
            differences.append(targets[1]-targets[0])
        d = np.asarray(differences)
        propensity = sensor_averaged_propensity(int(row.peer_board_target_count), params.N,
            resolved["q_c"], params.policy_beta, params.policy_threshold)
        summaries.append({"snapshot_id": snapshot["snapshot_id"], "pairs": pairs,
            "a_sensor_averaged": propensity, "chi_target": float(d.mean()),
            "se_paired": float(d.std(ddof=1)/np.sqrt(pairs)) if pairs>1 else ""})
    _write_csv(out / "micro_branch_samples.csv", rows)
    _write_csv(out / "micro_branch_summary.csv", summaries)
    _write_csv(out / "micro_factual_replay.csv", replay)
    return summaries


def _branch_compare(theory_dir: Path, cell_dir: Path):
    micro = pd.read_csv(cell_dir / "micro_branch_samples.csv")
    theory = pd.read_csv(theory_dir / "branch_samples.csv")
    result = []
    for snapshot_id in sorted(micro.snapshot_id.unique()):
        for action in (0, 1):
            m = micro.loc[(micro.snapshot_id == snapshot_id) & (micro.forced_action == action), "target_next"].to_numpy()
            t = theory.loc[(theory.snapshot_id == snapshot_id) & (theory.forced_action == action), "target_next"].to_numpy()
            result.append({"snapshot_id": snapshot_id, "forced_action": action,
                "n_micro": len(m), "n_theory": len(t), "micro_mean": float(m.mean()),
                "theory_mean": float(t.mean()), "theory_minus_micro": float(t.mean()-m.mean()),
                "micro_variance": float(m.var(ddof=1)), "theory_variance": float(t.var(ddof=1)),
                "wasserstein_1": _wasserstein_1d(t,m)})
    _write_csv(theory_dir / "branch_comparison.csv", result)
    ms = pd.read_csv(cell_dir / "micro_branch_summary.csv")
    ts = pd.read_csv(theory_dir / "branch_summary.csv")
    chi = ts.merge(ms, on="snapshot_id", suffixes=("_theory", "_micro"))
    chi["theory_minus_micro_chi"] = chi.chi_target_theory - chi.chi_target_micro
    chi.to_csv(theory_dir / "branch_chi_comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = sorted(micro.snapshot_id.unique())
    fig, axes = plt.subplots(len(ids), 2, figsize=(10, 3.2*len(ids)), squeeze=False,
                             constrained_layout=True)
    bins = np.linspace(0, 1, 13)
    for i, snapshot_id in enumerate(ids):
        for action, color in ((0, "#0072b2"), (1, "#d55e00")):
            for dataset, linestyle, label in ((micro, "solid", "MICRO"),
                                               (theory, "dashed", "REDUCED")):
                values = dataset.loc[(dataset.snapshot_id == snapshot_id) &
                                     (dataset.forced_action == action), "target_next"].to_numpy()
                counts, edges = np.histogram(values, bins=bins)
                axes[i,0].step(edges[:-1], counts/len(values), where="post",
                               color=color, linestyle=linestyle,
                               label=f"{label}, U={action}")
        axes[i,0].set(xlabel="next-day target share (common bins)",
                      ylabel="probability per bin", title=snapshot_id)
        axes[i,0].legend(fontsize=7)
        record = chi.loc[chi.snapshot_id == snapshot_id].iloc[0]
        axes[i,1].errorbar([0,1], [record.chi_target_micro, record.chi_target_theory],
                           yerr=[1.96*record.se_paired_micro, 1.96*record.se_paired_theory],
                           fmt="o", capsize=3, color="#333333")
        axes[i,1].set(xlim=(-.5,1.5), xticks=[0,1], xticklabels=["MICRO","REDUCED"],
                      ylabel="paired chi_target ± 1.96 SE", title="next-day causal response")
    fig.suptitle("Forced-action next-day distributions; 12 common bins for display")
    fig.savefig(theory_dir / "branch_comparison.pdf")
    fig.savefig(theory_dir / "branch_comparison.png", dpi=150)
    plt.close(fig)
    return chi


def _regime_slice(output: Path, summary: list[dict]):
    """Categorical pilot tiles; sparse cells remain empty rather than interpolated."""
    rows = [r for r in summary if r["metric"] == "target_share"]
    _write_csv(output / "regime_coordinates.csv", rows)
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rhos = sorted({r["rho"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    variants = list(VARIANTS)
    residual_max = max(.05, max(abs(r["late_bias"]) for r in rows))
    fig, axes = plt.subplots(len(variants), 3, figsize=(12, 2.7*len(variants)),
                             squeeze=False, constrained_layout=True)
    for i, variant in enumerate(variants):
        subset = [r for r in rows if r["variant"] == variant]
        for j, (key, color, limits) in enumerate((
                ("micro_late_mean", "viridis", (0,1)),
                ("theory_late_mean", "viridis", (0,1)),
                ("late_bias", "coolwarm", (-residual_max,residual_max)))):
            tile = np.full((len(rhos), len(budgets)), np.nan)
            for row in subset:
                tile[rhos.index(row["rho"]), budgets.index(row["budget"])] = row[key]
            ax = axes[i,j]
            view = ax.imshow(tile, cmap=color, vmin=limits[0], vmax=limits[1],
                             aspect="auto", origin="lower")
            ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
            ax.set_yticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
            label = {"paper_reduced": "paper reduced", "sampling_matched_reduced": "sampling matched",
                     "sampling_clock_corrected_reduced": "sampling + clock"}[variant]
            ax.set(xlabel="integer budget b", ylabel=(label + "\nrho") if j == 0 else "rho",
                   title=("MICRO", "REDUCED", "Theory − MICRO")[j] if i == 0 else "")
            for y in range(len(rhos)):
                for x in range(len(budgets)):
                    if np.isfinite(tile[y,x]):
                        ax.text(x,y,f"{tile[y,x]:.3f}",ha="center",va="center",color="black",
                                bbox={"facecolor":"white","alpha":.7,"edgecolor":"none"})
            fig.colorbar(view, ax=ax, shrink=.8)
    fig.suptitle("Saved-pilot late-window target share; categorical cells only")
    fig.savefig(output / "regime_pilot.pdf")
    fig.savefig(output / "regime_pilot.png", dpi=150)
    plt.close(fig)


def _write_report(output: Path, summary: list[dict], *, initial_blocks: int,
                  theory_replicas: int, substeps: int, branch_pairs: int):
    lines = ["# Santa Fe v3: reduced theory versus saved MICRO pilot", "",
        "This is a matched comparison from retained v3 trajectories. The theory is the",
        "supplied three-variable reduced Langevin runner, not full heterogeneous HMF.",
        f"Each cell uses {initial_blocks} actual round-0 initial blocks, {theory_replicas} "
        f"theory replicas per block, {substeps} solver substeps per day, and {branch_pairs} "
        "forced branch pairs per selected pre-action snapshot.", "",
        "## Late-window target share", "",
        "| Cell | rho | b | Variant | MICRO | Theory | Theory − MICRO | Paired-block SE |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        if row["metric"] != "target_share":
            continue
        lines.append(f"| {row['cell_id']} | {row['rho']:g} | {row['budget']} | {row['variant']} | "
                     f"{row['micro_late_mean']:.3f} | {row['theory_late_mean']:.3f} | "
                     f"{row['late_bias']:+.3f} | {row['paired_block_se']:.3f} |")
    lines += ["", "For a horizon below round 41, the late window is its last ten rounds.",
        "The paired-block SE uses independent saved initial blocks; normal 95% intervals",
        "with only a few blocks are exploratory. Theory replica variation is nested within",
        "each block. The per-round RMSE and distribution diagnostics are in each variant's",
        "comparison directory. Selected early/middle/final Wasserstein distances and",
        "paired initial-block bootstrap intervals are in `selected_distribution_bootstrap.csv`.",
        "With few blocks, those intervals are descriptive. The categorical pilot slice is",
        "`regime_pilot.pdf`; it does not establish a phase boundary.", "", "## Alignment and causal checks", "",
        "Round 0 uses each saved population and first front page. Row d contains the",
        "post-day population; U_d determines the next front page and can first affect row d+1.",
        "The built-in comparison verifies exact initial snapshots, physical parameters, and",
        "complete round panels. Each selected pre-action snapshot passes factual MICRO replay.",
        "At b=0, forced U=0 and U=1 branches have exactly equal next-day population",
        "outcomes under common streams. Active-branch continuous theory and discrete MICRO",
        "samples, paired response tables, and common-bin display histograms are retained.",
        "No T_pi or eta_IR is inferred from the continuous branches; a shared partition and",
        "bias/uncertainty protocol would be required.", "", "## Interpretation limits", "",
        "The corrected variant still assumes independent-binomial epistemic counts,",
        "vote/count independence, exchangeable fact IDs, Gaussian aggregate night thinning,",
        "and independent population/board noise. It is not an exact simulator surrogate.",
        "The outputs are a small saved-pilot comparison at available cells; more independent",
        "initial blocks and a wider rho/b grid are required for a regime map. Projection",
        "counts in theory diagnostics and 24-versus-48 substep sensitivity should be checked",
        "before attributing discrepancies solely to closure error.", ""]
    (output / "pilot_report.md").write_text("\n".join(lines))


def integrate(config_path: str | Path, out: str | Path, *, cell_ids: list[int],
              initial_blocks=2, theory_replicas=16, substeps=24,
              branch_pairs=16, branch_snapshots=1, seed=20260927):
    config = load_config(config_path)
    if config.params.model_version != "santa_fe_epistemic_feedback_v3":
        raise ValueError("only v3 trajectories can be compared with the supplied theory")
    if min(initial_blocks, theory_replicas, substeps, branch_pairs, branch_snapshots) < 1:
        raise ValueError("replication and numerical settings must be positive")
    output = Path(out).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("choose a new or empty integration output directory")
    output.mkdir(parents=True, exist_ok=True)
    saved_config = config.results_dir / "config.yaml"
    if not saved_config.is_file() or saved_config.read_bytes() != config.path.read_bytes():
        raise ValueError("saved simulator config must exactly match the requested YAML")
    rounds = pd.read_parquet(config.results_dir / "trajectories" / "round_trajectories.parquet")
    if not cell_ids or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("choose unique physical cell IDs")
    summary = []
    for cell_id in cell_ids:
        if not 0 <= cell_id < len(config.cells):
            raise ValueError(f"unknown physical cell {cell_id}")
        cell_dir = output / f"cell_{cell_id:04d}"
        resolved, selected = export_cell(config, rounds, cell_id, cell_dir,
            initial_blocks=initial_blocks, theory_replicas=theory_replicas)
        snapshots, source = _branch_records(selected, resolved, cell_dir,
                                             snapshots_per_cell=branch_snapshots)
        _micro_branch(source, snapshots, resolved, config, pairs=branch_pairs,
                      seed=seed, out=cell_dir)
        for variant, (sampling, covariance) in VARIANTS.items():
            variant_dir = cell_dir / variant
            variant_dir.mkdir()
            cfg = {"schema_version": 1, "parameters": resolved,
                   "solver": {"rounds": config.cells[cell_id].params.rounds,
                              "replicas_per_initial": theory_replicas,
                              "seed": seed+cell_id*100003, "substeps": substeps,
                              "sampling": sampling, "covariance": covariance,
                              "daytime_noise": True, "boundary": "project"},
                   "provenance": {"source": str(config.path), "cell_id": cell_id,
                                  "initial_fact_redundancy": config.cells[cell_id].params.initial_fact_redundancy,
                                  "simulator_trajectory": str(config.results_dir),
                                  "variant": variant}}
            _write_json(variant_dir / "config.json", cfg)
            _runner("run", config=variant_dir / "config.json",
                    initials=cell_dir / "initials.json", out=variant_dir / "theory")
            _runner("compare", theory=variant_dir / "theory",
                    simulation=cell_dir / "simulator.csv",
                    simulation_manifest=cell_dir / "simulation_manifest.json",
                    out=variant_dir / "comparison")
            horizon = config.cells[cell_id].params.rounds
            late_start = 41 if horizon >= 41 else max(1, horizon - 9)
            late = _extra_comparison(variant_dir / "theory", cell_dir / "simulator.csv",
                                     variant_dir / "comparison", late_start=late_start)
            _runner("branch", config=variant_dir / "config.json",
                    snapshots=cell_dir / "pre_action.json", pairs=branch_pairs,
                    out=variant_dir / "branch")
            chi = _branch_compare(variant_dir / "branch", cell_dir)
            for item in late:
                summary.append({"cell_id": cell_id, "variant": variant,
                                "rho": resolved["rho"], "budget": resolved["b"],
                                **item, "branch_chi_absolute_error_mean":
                                float(chi.theory_minus_micro_chi.abs().mean())})
    _write_csv(output / "cell_variant_summary.csv", summary)
    _regime_slice(output, summary)
    _write_report(output, summary, initial_blocks=initial_blocks,
                  theory_replicas=theory_replicas, substeps=substeps,
                  branch_pairs=branch_pairs)
    package = Path(__file__).resolve().parent.parent / "santa_fe_theory"
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(package.glob("*.py"))}
    _write_json(output / "integration_manifest.json", {"schema_version": 1,
        "theory_runner_version": VERSION, "theory_level": "reduced_binomial_langevin",
        "theory_source_sha256": source_hashes,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=False).stdout.strip() or None,
        "simulator_config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
        "simulator_results_dir": str(config.results_dir), "cell_ids": cell_ids,
        "variants": VARIANTS, "initial_blocks_per_cell": initial_blocks,
        "theory_replicas_per_block": theory_replicas, "substeps": substeps,
        "branch_pairs": branch_pairs, "branch_snapshots_per_cell": branch_snapshots,
        "master_seed": seed, "block_weighting": "equal initial blocks",
        "comparison_scope": "local saved-pilot cells; not a phase map",
        "late_window_rule": "rounds 41..T if T>=41; otherwise final up-to-10 rounds",
        "limitations": ["independent-binomial epistemic closure", "fact identities exchangeable",
                        "Gaussian aggregate night", "population/board noise independent",
                        "boundary projection and finite time steps"]})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cell-ids", type=int, nargs="+", required=True)
    parser.add_argument("--initial-blocks", type=int, default=2)
    parser.add_argument("--theory-replicas", type=int, default=16)
    parser.add_argument("--substeps", type=int, default=24)
    parser.add_argument("--branch-pairs", type=int, default=16)
    parser.add_argument("--branch-snapshots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args(argv)
    print(integrate(args.config, args.out, cell_ids=args.cell_ids,
                    initial_blocks=args.initial_blocks, theory_replicas=args.theory_replicas,
                    substeps=args.substeps, branch_pairs=args.branch_pairs,
                    branch_snapshots=args.branch_snapshots, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
