"""Validated YAML configuration and deterministic cell planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from pathlib import Path
import math
import yaml

from .state import SimulationParameters

DEFAULT_BUDGETS = [i / 8 for i in range(9)]
COORDINATES = {"mean_coverage", "population_coverage", "both"}


@dataclass(frozen=True)
class Cell:
    cell_id: int
    params: SimulationParameters
    beta_regime: str = ""


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict
    params: SimulationParameters
    episodes: int
    processes: int
    seed: int
    results_dir: Path
    save_round: bool
    save_micro: bool
    save_plots: bool
    save_reports: bool
    bins: int
    coordinate: str
    information: bool
    permutation: dict
    bootstrap: dict
    histograms: dict
    sample_size: dict
    cells: tuple[Cell, ...]


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    experiment = _section(raw, "experiment")
    output = _section(raw, "output")
    model = _section(raw, "model")
    sweep = _section(raw, "sweep")
    analysis = _section(raw, "analysis")
    permutation = _section(analysis, "permutation_null")
    bootstrap = _section(analysis, "bootstrap")
    histograms = _section(analysis, "histograms")
    sample_size = _section(raw, "sample_size_study")
    information = _section(analysis, "mutual_information")
    allowed = {f.name for f in fields(SimulationParameters)} - {"save_micro"}
    unknown = set(model) - allowed
    if unknown:
        raise ValueError(f"unknown model parameters: {sorted(unknown)}")
    params = SimulationParameters(**model)
    for key in ("N", "F", "rounds"):
        _positive_int(getattr(params, key), f"model.{key}")
    _positive_int(params.q, "model.q", 0)
    _positive_int(params.initial_fact_redundancy, "model.initial_fact_redundancy", 0)
    for key in ("rho", "truth_fact_fraction", "sensing_fraction", "budget_fraction"):
        value = getattr(params, key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"model.{key} must be between 0 and 1")
    if params.controller_target not in (-1, 1):
        raise ValueError("model.controller_target must be -1 or 1")
    for key in ("beta_evidence", "beta_social", "policy_beta", "policy_threshold"):
        if not math.isfinite(getattr(params, key)):
            raise ValueError(f"model.{key} must be finite")
    episodes = _positive_int(experiment.get("episodes", 20), "experiment.episodes")
    processes = _positive_int(experiment.get("processes", 1), "experiment.processes")
    seed = _positive_int(experiment.get("seed", 20260925), "experiment.seed", 0)
    bins = _positive_int(analysis.get("bins", 6), "analysis.bins", 2)
    coordinate = analysis.get("epistemic_coordinate", "mean_coverage")
    if coordinate not in COORDINATES:
        raise ValueError(f"analysis.epistemic_coordinate must be one of {sorted(COORDINATES)}")
    if "results_dir" not in output:
        raise ValueError("output.results_dir is required")
    results_dir = Path(output["results_dir"]).expanduser()
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    save_round = bool(output.get("save_round_trajectories", True))
    save_micro = bool(output.get("save_micro_trajectories", False))
    enabled_info = bool(information.get("enabled", False))
    if not save_round and (enabled_info or permutation.get("enabled") or bootstrap.get("enabled") or sample_size.get("enabled")):
        raise ValueError("round trajectories must be saved when statistical analysis is enabled")
    if permutation.get("enabled", False):
        _positive_int(permutation.get("n_permutations", 1000), "permutation_null.n_permutations")
        if not permutation.get("preserve_conditioning_state", True):
            raise ValueError("conditional permutation null must preserve conditioning strata")
    if bootstrap.get("enabled", False):
        _positive_int(bootstrap.get("n_bootstrap", 1000), "bootstrap.n_bootstrap")
    confidence = bootstrap.get("confidence_level", 0.95)
    if not 0 < confidence < 1:
        raise ValueError("bootstrap.confidence_level must be in (0, 1)")
    if bootstrap.get("unit", "episode") != "episode":
        raise ValueError("bootstrap.unit must be episode")
    if sample_size.get("enabled", False):
        _positive_int(sample_size.get("reference_episodes", 1000), "sample_size_study.reference_episodes")
        _positive_int(sample_size.get("repetitions", 200), "sample_size_study.repetitions")
        for n in sample_size.get("episode_counts", [10, 20, 30, 50, 75, 100]):
            _positive_int(n, "sample_size_study.episode_counts entry")
            if n > sample_size.get("reference_episodes", 1000):
                raise ValueError("sample-size episode count exceeds reference episodes")
        _positive_int(sample_size.get("n_bootstrap", 100), "sample_size_study.n_bootstrap")
        _positive_int(sample_size.get("n_permutations", 100), "sample_size_study.n_permutations")
    budgets = sweep.get("budget_fraction", DEFAULT_BUDGETS)
    rhos = sweep.get("rho", [params.rho])
    for key, values in (("budget_fraction", budgets), ("rho", rhos)):
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"sweep.{key} must be a nonempty list without duplicates")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f"sweep.{key} entries must be finite values in [0, 1]")
    if "beta_pairs" in sweep:
        if "beta_evidence" in sweep or "beta_social" in sweep:
            raise ValueError("beta_pairs cannot be combined with beta_evidence or beta_social axes")
        pairs = sweep["beta_pairs"]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("sweep.beta_pairs must be a nonempty list")
        entries = []
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"name", "beta_evidence", "beta_social"}:
                raise ValueError("each beta pair needs name, beta_evidence and beta_social")
            name, be, bs = pair["name"], pair["beta_evidence"], pair["beta_social"]
            if not isinstance(name, str) or not name or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in (be, bs)):
                raise ValueError("invalid beta pair")
            entries.append((name, float(be), float(bs)))
        if len({e[0] for e in entries}) != len(entries) or len({e[1:] for e in entries}) != len(entries):
            raise ValueError("duplicate beta pair name or coordinates")
    else:
        axes = []
        for key in ("beta_evidence", "beta_social"):
            values = sweep.get(key, [getattr(params, key)])
            if not isinstance(values, list) or not values or len(set(values)) != len(values):
                raise ValueError(f"sweep.{key} must be a nonempty list without duplicates")
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values):
                raise ValueError(f"sweep.{key} entries must be finite")
            axes.append(values)
        entries = [(f"be_{be:g}_bs_{bs:g}", float(be), float(bs)) for be, bs in product(*axes)]
    cells = []
    for name, be, bs in entries:
        for rho, bf in product(rhos, budgets):
            cells.append(Cell(len(cells), replace(params, beta_evidence=be, beta_social=bs,
                                           rho=float(rho), budget_fraction=float(bf), save_micro=save_micro), name))
    return Config(path, raw, params, episodes, processes, seed, results_dir.resolve(), save_round, save_micro,
                  bool(output.get("save_plots", True)), bool(output.get("save_reports", True)), bins,
                  coordinate, enabled_info, permutation, bootstrap, histograms, sample_size, tuple(cells))


def plan(config: Config) -> dict:
    calibration = config.sample_size
    return {"parameter_cells": len(config.cells), "total_episodes": len(config.cells) * config.episodes,
            "sample_size_reference_episodes": (1 if "statistic" in calibration else 2) * len(config.cells) * int(calibration.get("reference_episodes", 1000)) if calibration.get("enabled", False) else 0,
            "sample_size_repetitions": len(config.cells) * len(calibration.get("episode_counts", [10, 20, 30, 50, 75, 100])) * int(calibration.get("repetitions", 200)) if calibration.get("enabled", False) else 0,
            "budgets": [{"fraction": f, "integer": round(f * config.params.N)} for f in
                        dict.fromkeys(c.params.budget_fraction for c in config.cells)],
            "results_dir": str(config.results_dir)}
