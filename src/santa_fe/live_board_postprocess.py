"""Export v4 entropy, response, and state-local diagnostics from sealed cells."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from mas_cc.analysis.single_affinity import eta_ir, state_response_table, target_sensing_information

from .cluster import _sealed_rounds
from .config import load_config
from .live_board_metrics import extra_information, propensity_response, state_local_response
from .llm_parallel import adapt_live_board_trajectories


def run_cell(config_path:str,cell_id:int)->dict:
    config=load_config(config_path)
    if config.params.model_version!="santa_fe_live_board_v4":raise ValueError("v4 required")
    rounds=_sealed_rounds(config,cell_id)
    events=adapt_live_board_trajectories(rounds)
    n=config.cells[cell_id].params.N
    q_c=round(config.cells[cell_id].params.sensing_fraction*n)
    events=[replace(event,event={**event.event,"N":n,"sensor_sample_size":q_c,
                                  "controller_sensing_mode":"votes"}) for event in events]
    root=config.results_dir/"cells"/f"cell-{cell_id:04d}"/"live_board_analysis"
    root.mkdir(exist_ok=True)
    extra=extra_information(rounds)
    extra.insert(0,"cell_id",cell_id)
    extra.to_parquet(root/"entropy_information.parquet",index=False)
    response=propensity_response(rounds)
    response.insert(0,"cell_id",cell_id)
    response.to_parquet(root/"propensity_response.parquet",index=False)
    eta=eta_ir(events)
    sensing=target_sensing_information(events)
    code=hashlib.sha256()
    for path in (Path(__file__),Path(__file__).with_name("live_board_metrics.py")):
        code.update(path.read_bytes())
    diagnostics={"cell_id":cell_id,"eta_ir":eta,
                 "sensing_information":sensing,
                 "derived_code_sha256":code.hexdigest(),
                 "status":"descriptive_occupancy_estimate; check estimator calibration before interpretation"}
    (root/"derived_diagnostics.json").write_text(json.dumps(diagnostics,indent=2,allow_nan=True)+"\n")
    shared_states=pd.DataFrame(state_response_table(events).values())
    shared_states.insert(0,"cell_id",cell_id)
    shared_states.to_parquet(root/"shared_exact_K_response.parquet",index=False)
    states=state_local_response(rounds)
    states.insert(0,"cell_id",cell_id)
    states.to_parquet(root/"state_local_response.parquet",index=False)
    return {"cell_id":cell_id,"path":str(root),"events":len(events),"states":len(states)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--cell-id",required=True,type=int)
    args=parser.parse_args()
    print(json.dumps(run_cell(args.config,args.cell_id)))


if __name__=="__main__":main()
