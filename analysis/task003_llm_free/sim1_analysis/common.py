"""Shared helpers for the Simulation 1 analysis (see ANALYSIS_PLAN.md).

Every table is built from the run folders' params.json and Parquet files, never
from folder names. Output goes to results/analysis/<date>/ (not in git); curated
figures and small tables are copied next to this file.
"""
from __future__ import annotations
import csv, datetime, json, pathlib, re, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                                     # analysis/task003_llm_free
SIM = ROOT / "simulator"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SIM))

from llmfree.world import World  # noqa: E402
from llmfree.setups import load_setup  # noqa: E402

WORLD_FILE = SIM / "world_task003.json"
SETUPS_DIR = ROOT / "experimental_setup"
SETUPS = ("task003_symmetric", "task003_nosolution")
VARIANTS = ("sim1_base", "sim1_pm", "sim1_pm_post_always", "sim1_uniform_post")
VARIANT_LABEL = {
    "sim1_base": "argmax, supporting posts",
    "sim1_pm": "prob. matching, supporting posts",
    "sim1_pm_post_always": "prob. matching, post always",
    "sim1_uniform_post": "argmax, uniform posts",
}
VARIANT_COLOR = {"sim1_base": "#1b6ca8", "sim1_pm": "#d1495b", "sim1_pm_post_always": "#edae49",
                 "sim1_uniform_post": "#3d9970"}
ALLOC_COLOR = {0: "#1b6ca8", 1: "#999999", 2: "#d1495b"}
POST_REASONS = {0: "in_context", 1: "fallback", 2: "random", 3: "uniform", 4: "abstained", 5: "no_memory"}


def out_dir(date: str | None = None) -> pathlib.Path:
    d = RESULTS / "analysis" / (date or f"{datetime.date.today():%Y-%m-%d}")
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def runs() -> pd.DataFrame:
    """The official runs, from the committed index."""
    return pd.read_csv(SIM / "runs_index.csv")


def run_folder(run: str) -> pathlib.Path:
    return RESULTS / run


def study_of_run(run: str) -> str:
    return run.split("_", 1)[1]                     # "2026-10-07_sim1_base" -> "sim1_base"


def cells_table() -> pd.DataFrame:
    """One row per cell per run, every factor a column, read from params.json."""
    rows = []
    for _, r in runs().iterrows():
        for p in sorted((run_folder(r.run) / "cells").glob("*/params.json")):
            prm = json.loads(p.read_text())
            study = r.study
            purpose = ("confirmation" if "_confirm_" in study else
                       "no_proof_stop" if study.endswith("_no_proof_stop") else "main")
            rows.append({
                "run": r.run, "study": study, "purpose": purpose,
                "variant": re.sub(r"(_no_proof_stop|_confirm_seed\d+)$", "", study),
                "stopping_rule": "no_proof_stop" if study.endswith("_no_proof_stop") else "stop_when_proved",
                "cell": p.parent.name, "path": str(p.parent),
                **{k: prm[k] for k in ("setup", "arm", "q", "qc", "b", "rho", "M", "agent_sampling_mode",
                                        "agent_post_rule", "agent_post_always", "silent_when_target_proved",
                                        "theta_vote", "seed")},
            })
    t = pd.DataFrame(rows)
    qc = t.qc.fillna(0)
    t["effective_gate"] = np.where(t.arm == "silent", np.nan,
                                   np.ceil(t.theta_vote * qc) / qc.replace(0, np.nan))
    return t


def bootstrap_mean(x: np.ndarray, n_boot: int = 1000, seed: int = 0, axis: int = 0):
    """Mean over episodes (axis 0) and a 95% percentile interval, resampling episodes."""
    rng = np.random.default_rng(seed)
    n = x.shape[axis]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)                       # (n_boot, ...)
    return x.mean(axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def write_index(d: pathlib.Path, entries: list[dict]):
    """Append figure/table entries to <out>/index.csv (question, file, what, cells)."""
    f = d / "index.csv"
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["question", "file", "what", "cells"], lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(entries)
