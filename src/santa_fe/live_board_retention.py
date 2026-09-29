"""Compact completed v4 cells after all raw-dependent analyses have finished.

The original cell seal records the full-data hashes. The retention seal records
which eight illustrative episodes remain and verifies all derived outputs.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .cluster import _cell_dir, _live_board_run_table, _load_manifest, _parquet, _sealed_rounds, _sha, _valid_seal, _json, _v3_code_hashes
from .config import load_config


EXTRA_FILES = (
    "entropy_information.parquet",
    "propensity_response.parquet",
    "shared_exact_K_response.parquet",
    "state_local_response.parquet",
    "derived_diagnostics.json",
)


def verified_run_summary(config, cell_id: int) -> pd.DataFrame:
    root = _cell_dir(config, cell_id) / "retained"
    seal = root / "retention_complete.json"
    if not _valid_seal(seal, {"cell_id": cell_id, "config_sha256": _sha(config.path)}):
        raise ValueError(f"retained cell {cell_id} has an invalid seal")
    table = pd.read_parquet(root / "per_run_summary.parquet")
    if len(table) != config.episodes or table.seed.nunique() != config.episodes:
        raise ValueError(f"retained cell {cell_id} has incomplete episode summaries")
    return table


def compact_cell(config, cell_id: int, *, delete_raw: bool = True) -> dict:
    if config.params.model_version != "santa_fe_live_board_v4":
        raise ValueError("v4 required")
    _load_manifest(config)
    root = _cell_dir(config, cell_id)
    retained = root / "retained"
    if (retained / "retention_complete.json").is_file():
        verified_run_summary(config, cell_id)
        seal_path = retained / "retention_complete.json"
        payload = json.loads(seal_path.read_text())
        if delete_raw and not payload["raw_removed"]:
            (root / "rounds.parquet").unlink()
            (root / "micro.parquet").unlink(missing_ok=True)
            payload["raw_removed"] = True
            _json(seal_path, payload)
        return {"cell_id": cell_id, "status": "already_compacted"}

    rounds = _sealed_rounds(config, cell_id)
    original = json.loads((root / "cell_complete.json").read_text())
    source_hash = original["files"]["rounds.parquet"]
    information = root / "information"
    expected = {"cell_id": cell_id, "config_sha256": _sha(config.path),
                "source_sha256": source_hash,
                "information_engine_sha256": _v3_code_hashes(config.params.model_version)["information_engine_sha256"]}
    if not _valid_seal(information / "information_complete.json", expected):
        raise ValueError(f"cell {cell_id}: information analysis is not sealed")
    extra = root / "live_board_analysis"
    for name in EXTRA_FILES:
        path = extra / name
        if not path.is_file():
            raise ValueError(f"cell {cell_id}: missing extra output {name}")
    diagnostics = json.loads((extra / "derived_diagnostics.json").read_text())
    if diagnostics.get("cell_id") != cell_id:
        raise ValueError(f"cell {cell_id}: extra diagnostics have wrong cell ID")

    table = _live_board_run_table(config, config.cells[cell_id], rounds)
    terminal = table.set_index("seed").final_target_share
    ordered = sorted(terminal.index)
    illustrative = list(dict.fromkeys(
        ordered[:4] + list(terminal.nsmallest(2).index) + list(terminal.nlargest(2).index)))
    for seed in ordered:
        if len(illustrative) >= 8:
            break
        if seed not in illustrative:
            illustrative.append(seed)
    illustrative = illustrative[:8]
    retained.mkdir(exist_ok=True)
    run_path = retained / "per_run_summary.parquet"
    round_path = retained / "example_rounds.parquet"
    micro_path = retained / "example_micro.parquet"
    _parquet(run_path, table)
    _parquet(round_path, rounds.loc[rounds.seed.isin(illustrative)])
    micro = root / "micro.parquet"
    if micro.is_file():
        _parquet(micro_path, pd.read_parquet(micro).loc[lambda x: x.seed.isin(illustrative)])
    files = {path.name: _sha(path) for path in (run_path, round_path, micro_path) if path.is_file()}
    payload = {"cell_id": cell_id, "config_sha256": _sha(config.path),
               "independent_episodes_analyzed": config.episodes,
               "illustrative_episodes_retained": len(illustrative),
               "illustrative_seeds": [int(s) for s in illustrative],
               "selection_rule": "first four ordered seeds; two lowest and two highest final target shares; fill duplicate slots with next ordered seeds",
               "original_file_sha256": original["files"],
               "information_seal_sha256": _sha(information / "information_complete.json"),
               "extra_file_sha256": {name: _sha(extra / name) for name in EXTRA_FILES},
               "files": files,
               "raw_removed": bool(delete_raw)}
    _json(retained / "retention_complete.json", payload)
    verified_run_summary(config, cell_id)
    if delete_raw:
        (root / "rounds.parquet").unlink()
        micro.unlink(missing_ok=True)
    return {"cell_id": cell_id, "status": "compacted" if delete_raw else "prepared",
            "retained_episodes": len(illustrative)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell-id", required=True, type=int)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compact_cell(load_config(args.config), args.cell_id,
                                  delete_raw=not args.keep_raw)))


if __name__ == "__main__":
    main()
