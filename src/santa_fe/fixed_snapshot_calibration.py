"""Independent-arm Santa Fe v3 fixed-snapshot channel calibration.

This is a small, explicit channel experiment. It uses the shared MI estimator
on a propensity-weighted 2 x (N+1) table; all MI outputs are nats.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mas_cc.analysis.estimators import mutual_information_from_counts
from .config import load_config
from .v3_game import replay_next_day
from .v3_validation import sensor_averaged_propensity


def _records(value):
    return json.loads(value) if isinstance(value, str) else value


def channel_information_nats(arm0, arm1, propensity: float, population: int):
    """Plug-in T_pi for a fixed state and the known sensor-averaged policy."""
    if not 0 <= propensity <= 1 or not len(arm0) or not len(arm1):
        raise ValueError("invalid propensity or empty arm")
    counts = np.stack((np.bincount(arm0, minlength=population + 1) / len(arm0) * (1 - propensity),
                       np.bincount(arm1, minlength=population + 1) / len(arm1) * propensity))
    return float(mutual_information_from_counts(counts).unsmoothed * math.log(2))


def _rollouts(row, params, arm: int, count: int, seed: int):
    agents = _records(row.agent_states_json)
    peer = _records(row.peer_board_json)
    weights = _records(row.fact_weights_json)
    pool = _records(row.controller_fact_pool_json)
    outcomes = np.empty(count, dtype=int)
    for i in range(count):
        streams = np.random.SeedSequence([seed, int(row.cell_id), int(row.seed),
                                          int(row["round"]), arm, i]).spawn(2)
        downstream = np.random.default_rng(streams[0]).bit_generator.state
        fact_rng = np.random.default_rng(streams[1])
        forced = [{"message_id": f"calibration-{arm}-{i}-{j}", "author": params.N,
                   "vote": params.controller_target, "fact_id": int(fact_rng.choice(pool)),
                   "fact_sign": params.controller_target, "source": "controller"}
                  for j in range(params.budget * arm)]
        next_agents, _ = replay_next_day(agents, peer + forced, weights, params,
                                         downstream, int(row["round"]) + 1)
        outcomes[i] = sum(a["vote"] == params.controller_target for a in next_agents)
    return outcomes


def calibrate(config_path, out, *, cell_ids, per_arm=(16, 32, 64),
              reference_per_arm=128, repetitions=20, round_number=15, seed=20261101):
    config = load_config(config_path)
    rounds = pd.read_parquet(config.results_dir / "trajectories" / "round_trajectories.parquet")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for cell_id in cell_ids:
        params = config.cells[cell_id].params
        if params.model_version != "santa_fe_epistemic_feedback_v3":
            raise ValueError("requires Santa Fe v3")
        candidates = rounds.loc[(rounds.cell_id == cell_id) & (rounds["round"] == round_number)]
        if candidates.empty:
            raise ValueError(f"no pre-action round {round_number} for cell {cell_id}")
        row = candidates.sort_values("seed").iloc[0]
        q_c = min(params.N, max(1, round(params.sensing_fraction * params.N)))
        a = sensor_averaged_propensity(int(row.peer_board_target_count), params.N,
                                       q_c, params.policy_beta, params.policy_threshold)
        h = -sum(p * math.log(p) for p in (a, 1-a) if p > 0)
        ref0 = _rollouts(row, params, 0, reference_per_arm, seed + 1000001)
        ref1 = _rollouts(row, params, 1, reference_per_arm, seed + 2000001)
        reference = channel_information_nats(ref0, ref1, a, params.N)
        bootstrap_rng = np.random.default_rng(np.random.SeedSequence([seed, cell_id, 9000001]))
        reference_bootstrap = np.array([channel_information_nats(
            bootstrap_rng.choice(ref0, size=len(ref0), replace=True),
            bootstrap_rng.choice(ref1, size=len(ref1), replace=True), a, params.N)
            for _ in range(199)])
        reference_se = float(reference_bootstrap.std(ddof=1))
        for n in per_arm:
            for trial in range(repetitions):
                # Distinct downstream seeds for arms, trials, references and nulls.
                origin = seed + 10000000 + trial * 100000 + n * 1000
                arm0 = _rollouts(row, params, 0, n, origin)
                arm1 = _rollouts(row, params, 1, n, origin + 100000000)
                null0 = _rollouts(row, params, 0, n, origin + 200000000)
                null1 = _rollouts(row, params, 0, n, origin + 300000000)
                rows.append({"cell_id": cell_id, "snapshot_seed": int(row.seed),
                             "snapshot_round": round_number, "budget": params.budget,
                             "rho": params.rho, "q": params.q, "q_c": q_c,
                             "per_arm": n, "trial": trial, "propensity": a,
                             "H_action_nats_exact": h, "T_pi_reference_nats": reference, "T_pi_reference_bootstrap_se": reference_se,
                             "T_pi_physical_null_exact_nats": 0.0 if params.budget == 0 else float("nan"),
                             "reference_per_arm": reference_per_arm,
                             "T_pi_nats": channel_information_nats(arm0, arm1, a, params.N),
                             "T_pi_null_nats": channel_information_nats(null0, null1, a, params.N),
                             "chi_target": float((arm1.mean()-arm0.mean()) / params.N),
                             "null_chi_target": float((null1.mean()-null0.mean()) / params.N),
                             "conditioning": "complete_saved_pre_action_snapshot",
                             "output": "integer_next_day_target_count",
                             "reference_kind": "independent_finite_MC"})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "fixed_snapshot_trials.csv", index=False)
    summaries = []
    for (cell_id, n), group in frame.groupby(["cell_id", "per_arm"]):
        split = max(1, len(group)//2)
        null_design = group.iloc[:split]
        evaluation = group.iloc[split:]
        threshold = float(null_design.T_pi_null_nats.quantile(.95))
        errors = group.T_pi_nats - group.T_pi_reference_nats
        summaries.append({"cell_id":cell_id, "per_arm":n, "repetitions":len(group),
                          "reference_per_arm":reference_per_arm,
                          "T_pi_reference_nats":group.T_pi_reference_nats.iloc[0],
                          "bias_against_MC_reference":float(errors.mean()),
                          "rmse_against_MC_reference":float(np.sqrt(np.mean(errors**2))),
                          "null_95_percentile":threshold,
                          "null_threshold_trials":len(null_design),
                          "independent_evaluation_trials":len(evaluation),
                          "null_false_positive_fraction":float(np.mean(evaluation.T_pi_null_nats > threshold)) if len(evaluation) else float("nan"),
                          "alternative_detection_fraction":float(np.mean(evaluation.T_pi_nats > threshold)) if len(evaluation) else float("nan"),
                          "T_pi_reference_bootstrap_se":group.T_pi_reference_bootstrap_se.iloc[0],
                          "T_pi_physical_null_exact_nats":group.T_pi_physical_null_exact_nats.iloc[0],
                          "H_action_nats_exact":group.H_action_nats_exact.iloc[0]})
    pd.DataFrame(summaries).to_csv(out / "fixed_snapshot_summary.csv", index=False)
    (out / "design.json").write_text(json.dumps({"config":str(config_path),"cell_ids":cell_ids,
        "per_arm":list(per_arm),"reference_per_arm":reference_per_arm,"repetitions":repetitions,
        "round":round_number,"seed":seed,"null":"two independent U=0 arms",
        "reference_limitation":"finite Monte Carlo bank, not exact truth",
        "test_limitation":"threshold estimated from separate first-half null trials; pilot resolution limited by repetitions"}, indent=2) + "\n")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cell-ids", type=int, nargs="+", required=True)
    parser.add_argument("--per-arm", type=int, nargs="+", default=[16,32,64])
    parser.add_argument("--reference-per-arm", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--round", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args(argv)
    print(calibrate(args.config, args.out, cell_ids=args.cell_ids, per_arm=args.per_arm,
                    reference_per_arm=args.reference_per_arm, repetitions=args.repetitions,
                    round_number=args.round, seed=args.seed))


if __name__ == "__main__":
    main()
