"""Disjoint-episode estimator pilot for version 4 saved trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .llm_parallel import adapt_live_board_trajectories
from .llm_parallel import _summarize_nulls
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import round_information_analysis


STATISTICS=("round_target_sensing_mi","round_sensor_action_mi",
            "round_target_actuation_cmi","round_memory_target_actuation_cmi",
            "round_kappa_target_actuation_cmi","round_phi_target_actuation_cmi",
            "round_epistemic_target_actuation_cmi","round_target_susceptibility")


def run_cell(config_path:str,cell_id:int,*,null_draws:int=19,min_episodes:int=8)->dict:
    config=load_config(config_path)
    path=config.results_dir/"cells"/f"cell-{cell_id:04d}"/"rounds.parquet"
    if not path.is_file():raise FileNotFoundError(path)
    rounds=pd.read_parquet(path)
    episodes=np.array(sorted(rounds.seed.unique()))
    if len(episodes)!=config.episodes:raise ValueError("incomplete episode bank")
    order=np.random.default_rng(config.seed+cell_id*1009).permutation(episodes)
    counts=[n for n in (8,16,32,64,128,256,512) if min_episodes<=n<=len(order)]
    output=[]
    for n in counts:
        for repetition,start in enumerate(range(0,len(order)-n+1,n)):
            subset=rounds.loc[rounds.seed.isin(order[start:start+n])]
            events=adapt_live_board_trajectories(subset)
            estimates,nulls=round_information_analysis(events,statistics=STATISTICS,
                bootstrap_resamples=0,null_permutations=null_draws,
                seed=config.seed+cell_id*100000+n*100+repetition)
            for row in _summarize_nulls(estimates,nulls):
                output.append({"cell_id":cell_id,"sample_episodes":n,"repetition":repetition,
                    "disjoint_within_sample_size":True,"nested_across_sample_sizes":True,
                    "round_transitions":len(events),"null_draws":null_draws,
                    "statistic":row["statistic"],"estimate":row["estimate"],
                    "null_mean":row["null_mean"],"estimate_minus_null":row["estimate_minus_null"],
                    "null_p_value":row["null_p_value"],
                    "null_type":row["null_type"],"dual_action_event_fraction":row["round_dual_action_event_fraction"],
                    "conditioning_state_count":row["round_conditioning_state_count"]})
    destination=config.results_dir/("calibration" if null_draws==19 else f"calibration_{null_draws}")
    destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(output).to_parquet(destination/f"cell-{cell_id:04d}.parquet",index=False)
    seal={"cell_id":cell_id,"independent_episodes":len(episodes),
          "sample_sizes":counts,"repetitions_by_size":{str(n):len(order)//n for n in counts},
          "null_draws":null_draws,
          "interpretation":"Disjoint datasets within each sample size; sizes reuse one episode bank. This is an estimator pilot, not calibrated power or an independent high-precision reference."}
    (destination/f"cell-{cell_id:04d}.json").write_text(json.dumps(seal,indent=2)+"\n")
    return seal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--cell-id",required=True,type=int)
    parser.add_argument("--null-draws",type=int,default=19)
    parser.add_argument("--min-episodes",type=int,default=8)
    args=parser.parse_args()
    print(json.dumps(run_cell(args.config,args.cell_id,null_draws=args.null_draws,
                              min_episodes=args.min_episodes)))


if __name__=="__main__":main()
