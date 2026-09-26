"""Episode generation and persistence, independent of estimators."""
from __future__ import annotations

from dataclasses import asdict
from multiprocessing import get_context
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import Config
from .game import SyntheticGame
from .state import SimulationParameters


def _worker(payload: tuple[int, dict, int, str]) -> tuple[list[dict], list[dict]]:
    cell_id, params_data, seed, beta_regime = payload
    result = SyntheticGame(SimulationParameters(**params_data)).run_episode(seed)
    common = {"cell_id": cell_id, "seed": seed, "beta_regime": beta_regime, **result.params}
    return ([{**common, **row} for row in result.rounds],
            [{**common, **row} for row in result.micro])


def run(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounds: list[dict] = []
    micro: list[dict] = []
    for cell in tqdm(config.cells, desc="parameter cells"):
        seeds = np.random.SeedSequence([config.seed, cell.cell_id]).generate_state(config.episodes, dtype=np.uint64)
        payloads = [(cell.cell_id, asdict(cell.params), int(seed), cell.beta_regime) for seed in seeds]
        if config.processes == 1:
            iterator = map(_worker, payloads)
            for rr, mm in tqdm(iterator, total=config.episodes, desc=f"episodes cell {cell.cell_id}", leave=False):
                rounds.extend(rr)
                micro.extend(mm)
        else:
            with get_context("spawn").Pool(config.processes) as pool:
                iterator = pool.imap_unordered(_worker, payloads, chunksize=1)
                for rr, mm in tqdm(iterator, total=config.episodes, desc=f"episodes cell {cell.cell_id}", leave=False):
                    rounds.extend(rr)
                    micro.extend(mm)
    return (pd.DataFrame(rounds).sort_values(["cell_id", "seed", "round"]).reset_index(drop=True),
            pd.DataFrame(micro).sort_values(["cell_id", "seed", "round", "slot"]).reset_index(drop=True)
            if micro else pd.DataFrame())


def save_trajectories(config: Config, rounds: pd.DataFrame, micro: pd.DataFrame) -> None:
    root = config.results_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_bytes(config.path.read_bytes())
    metadata = {"source_config": str(config.path), "cells": len(config.cells),
                "episodes_per_cell": config.episodes, "total_episodes": len(config.cells) * config.episodes,
                "seed": config.seed, "schema_version": 1,
                "model_semantics": {key: getattr(config.params, key) for key in ("model_version", "persistence_clock",
                    "peer_posting_mode", "controller_message_mode", "board_clock", "controller_fact_selection")},
                "budget_map": {str(c.params.budget_fraction): c.params.budget for c in config.cells}}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    path = root / "trajectories"
    path.mkdir(exist_ok=True)
    if config.save_round:
        rounds.to_parquet(path / "round_trajectories.parquet", index=False)
    if config.save_micro and not micro.empty:
        micro.to_parquet(path / "micro_trajectories.parquet", index=False)
