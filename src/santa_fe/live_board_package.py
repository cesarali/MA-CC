"""Build a compact, reviewable archive of the completed live-board grid.

This reads existing sealed outputs. It does not run episodes or re-estimate MI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .config import load_config


ROOT = Path("/shared/home/cesar/work/results/santa_fe_live_board_v4_full")
CAL = Path("/shared/home/cesar/work/results/santa_fe_live_board_v4_calibration_512")
PILOT = Path("/shared/home/cesar/work/results/santa_fe_live_board_v4_pilot")
REPO = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy(stage: Path, source: Path, name: str) -> None:
    target = stage / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def cell_metadata() -> pd.DataFrame:
    x = pd.read_csv(ROOT / "summaries/cell_metric_summary.csv", low_memory=False)
    return x.drop_duplicates("cell_id")[["cell_id", "controller_target", "beta_regime",
        "rho", "budget", "q", "q_c", "independent_episodes"]].set_index("cell_id")


def export_derived(stage: Path, metadata: pd.DataFrame) -> None:
    destination = stage / "derived"
    destination.mkdir(exist_ok=True)
    entropy_path = destination / "entropy_information.csv"
    response_path = destination / "propensity_response.csv"
    diagnostics_path = destination / "derived_diagnostics.jsonl"
    local_path = destination / "state_local_exact_K.csv"
    whole_path = destination / "whole_cell_susceptibility_by_conditioning.csv"
    entropy_first = response_first = local_first = whole_first = True
    with diagnostics_path.open("w") as diagnostics, local_path.open("w", newline="") as local_file, whole_path.open("w", newline="") as whole_file:
        local_writer = csv.writer(local_file)
        whole_writer = csv.writer(whole_file)
        local_columns = ["cell_id", "controller_target", "beta_regime", "rho", "budget", "q", "q_c",
            "conditioning", "target_count", "target_share", "events", "identified_events",
            "identified_occupancy_mass", "silence_events", "action_events",
            "chi_identified_weighted", "available_ipw_local_mean_weighted",
            "available_eligible_events", "status"]
        whole_columns = ["cell_id", "controller_target", "beta_regime", "rho", "budget", "q", "q_c",
            "conditioning", "events", "identified_events", "identified_occupancy_mass",
            "chi_identified_weighted", "status"]
        local_writer.writerow(local_columns)
        whole_writer.writerow(whole_columns)
        for cell_id, meta in metadata.iterrows():
            cell = ROOT / "cells" / f"cell-{cell_id:04d}" / "live_board_analysis"
            entropy = pd.read_parquet(cell / "entropy_information.parquet")
            response = pd.read_parquet(cell / "propensity_response.parquet")
            entropy.to_csv(entropy_path, mode="w" if entropy_first else "a", header=entropy_first, index=False)
            response.to_csv(response_path, mode="w" if response_first else "a", header=response_first, index=False)
            entropy_first = response_first = False
            d = json.loads((cell / "derived_diagnostics.json").read_text())
            diagnostics.write(json.dumps(d, allow_nan=True) + "\n")
            states = pd.read_parquet(cell / "state_local_response.parquet")
            parameter = [cell_id, int(meta.controller_target), meta.beta_regime, float(meta.rho),
                         int(meta.budget), int(meta.q), int(meta.q_c)]

            def summarize(group: pd.DataFrame, *, local: bool) -> list:
                total = int(group.events.sum())
                valid = group.loc[group.chi_identified & group.chi_target_fraction.notna()]
                identified = int(valid.events.sum())
                chi = float(np.average(valid.chi_target_fraction, weights=valid.events)) if identified else np.nan
                status = "identified_partially_or_fully" if identified else "no_dual_action_state"
                common = [total, identified, identified / total if total else np.nan]
                if not local:
                    return common + [chi, status]
                available = group.loc[group.available_ipw_local_mean.notna() &
                                      (group.available_eligible_events > 0)]
                eligible = int(available.available_eligible_events.sum())
                avail_mean = (float(np.average(available.available_ipw_local_mean,
                              weights=available.available_eligible_events)) if eligible else np.nan)
                return common + [int(group.silence_events.sum()), int(group.action_events.sum()),
                                 chi, avail_mean, eligible, status]

            for (conditioning, count), group in states.groupby(["conditioning", "target_count"], sort=False):
                local_writer.writerow(parameter + [conditioning, int(count), int(count) / 24] +
                                      summarize(group, local=True))
            for conditioning, group in states.groupby("conditioning", sort=False):
                whole_writer.writerow(parameter + [conditioning] + summarize(group, local=False))


def export_calibration(stage: Path) -> None:
    destination = stage / "calibration"
    destination.mkdir(exist_ok=True)
    chunks = []
    for count in (19, 99):
        directory = CAL / ("calibration" if count == 19 else "calibration_99")
        for file in sorted(directory.glob("cell-*.parquet")):
            frame = pd.read_parquet(file)
            frame["calibration_bank"] = "independent_512_episode_bank"
            chunks.append(frame)
    data = pd.concat(chunks, ignore_index=True)
    data.to_csv(destination / "episode_sensitivity.csv", index=False)
    summary = data.groupby(["cell_id", "null_draws", "sample_episodes", "statistic"], dropna=False).agg(
        disjoint_groups=("repetition", "nunique"),
        mean_estimate=("estimate", "mean"), sd_estimate=("estimate", "std"),
        mean_null=("null_mean", "mean"), mean_excess=("estimate_minus_null", "mean"),
        sd_excess=("estimate_minus_null", "std"),
        mean_null_p_value=("null_p_value", "mean"),
        dual_action_event_fraction=("dual_action_event_fraction", "mean"),
    ).reset_index()
    summary.to_csv(destination / "episode_sensitivity_summary.csv", index=False)
    seals = []
    for count in (19, 99):
        directory = CAL / ("calibration" if count == 19 else "calibration_99")
        for file in sorted(directory.glob("cell-*.json")):
            seals.append(json.loads(file.read_text()))
    (destination / "calibration_seals.json").write_text(json.dumps(seals, indent=2) + "\n")
    pilot = pd.read_parquet(PILOT / "information/llm_parallel/round_information_estimates.parquet")
    pilot.loc[pilot.scope == "pooled"].to_csv(destination / "independent_128_episode_pilot_pooled.csv", index=False)

    tpi = summary.loc[(summary.statistic == "round_target_actuation_cmi") & (summary.null_draws == 99)]
    with PdfPages(destination / "Tpi_episode_sensitivity.pdf") as pdf:
        for cell_id, group in tpi.groupby("cell_id"):
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            x = group.sample_episodes.to_numpy()
            ax.errorbar(x, group.mean_estimate, yerr=group.sd_estimate, marker="o", label="raw T_pi; SD across disjoint groups")
            ax.errorbar(x, group.mean_excess, yerr=group.sd_excess, marker="s", label="raw minus mean policy null")
            ax.axhline(0, color="black", linewidth=.7)
            ax.set(xscale="log", xlabel="independent episodes per dataset", ylabel="bits",
                   title=f"Cell {cell_id}: descriptive T_pi sensitivity (99 null draws)")
            ax.set_xticks(sorted(x)); ax.set_xticklabels([str(v) for v in sorted(x)])
            ax.legend(fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def selected_examples(stage: Path, metadata: pd.DataFrame) -> None:
    conditions = metadata.loc[metadata.q.eq(3) & metadata.q_c.eq(3) &
        metadata.rho.isin([.25, 1.0]) & metadata.budget.isin([0, 24])]
    index = []
    for cell_id, row in conditions.iterrows():
        source = ROOT / "cells" / f"cell-{cell_id:04d}" / "retained" / "example_rounds.parquet"
        name = f"examples/cell-{cell_id:04d}-example-rounds.parquet"
        copy(stage, source, name)
        index.append({"cell_id": cell_id, "controller_target": row.controller_target,
            "beta_regime": row.beta_regime, "rho": row.rho, "budget": row.budget,
            "q": row.q, "q_c": row.q_c, "file": name,
            "interpretation": "illustrative eight-episode subset; not an independent estimate"})
    pd.DataFrame(index).to_csv(stage / "examples/selection_index.csv", index=False)


def write_figure_index(stage: Path) -> None:
    """Index the map PDF in the same metric/regime/target order as the plotter."""
    data = pd.read_csv(ROOT / "plots/live_board_maps.csv", keep_default_na=False, low_memory=False)
    rows = []
    for (metric, view), group in data.groupby(["metric", "view"], sort=False):
        if pd.to_numeric(group.value, errors="coerce").notna().sum() == 0:
            continue
        for regime in ("balanced", "evidence_weighted", "social_weighted"):
            for target in (1, -1):
                if not group.loc[(group.beta_regime == regime) &
                                 (group.controller_target == target)].empty:
                    rows.append({"pdf_page": len(rows) + 1, "metric": metric, "view": view,
                                 "beta_regime": regime, "controller_target": target,
                                 "panels": "rows=q={3,12,24}; columns=q_c={3,12,24}; vertical=rho; horizontal=B"})
    if len(rows) != 390:
        raise ValueError(f"map figure count changed: expected 390, got {len(rows)}")
    (stage / "figures").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(stage / "figures/live_board_maps_index.csv", index=False)
    (stage / "figures/FIGURE_GUIDE.md").write_text("""# Figure guide

`live_board_maps.pdf` has 390 pages. Find a page by metric, view, beta regime and controller target in `live_board_maps_index.csv`. Each page has nine panels: rows are q=3,12,24 and columns are q_c=3,12,24. Within a panel, horizontal cells are B=0,3,...,24 and vertical cells are rho=0.25,0.4,0.55,0.7,0.85,1. Raw, policy-null and raw-minus-null information views have distinct pages. Blank cells denote unavailable values. These are finite-horizon 60-round maps.

Read `live_board_maps.csv` with `keep_default_na=False` in pandas if you want the literal `view="null"` label preserved; otherwise pandas treats that string as a missing value. The PDF is generated from the numeric CSV by `code/live_board_plots.py`.
""")


def write_docs(stage: Path, metadata: pd.DataFrame) -> None:
    physical = pd.read_csv(ROOT / "summaries/cell_metric_summary.csv")
    support = physical.loc[physical.metric == "late_target_share"]
    insufficient = int((~support.precision_sufficient.astype(bool)).sum())
    info = pd.read_parquet(ROOT / "information/llm_parallel/round_information_estimates.parquet",
                           columns=["cell_id", "scope", "statistic", "estimate"])
    pooled = info.loc[info.scope == "pooled"]
    (stage / "RESULTS_OVERVIEW.md").write_text(f"""# Santa Fe live-board v4: review overview

This is the completed **microscopic Santa Fe live-board simulation**, not a stochastic mean-field theory comparison. It comprises {len(metadata):,} physical cells and 128 independent 60-round episodes per cell ({len(metadata)*128:,} episodes). The grid crosses rho={{0.25,0.4,0.55,0.7,0.85,1}}, B={{0,3,...,24}}, q and q_c={{3,12,24}}, three beta regimes, and truth/false controller targets. Fixed settings and all resolved choices are in `config/full_grid.yaml` and `docs/study_spec.md`.

The 60-round finite-horizon regime maps are in `figures/live_board_maps.pdf`; their complete numeric cells are in `figures/live_board_maps.csv`. These are **not established phase transitions**. The primary late support window is X_41..X_60. Physical summaries cover every cell, with a row for each independent episode in `summaries/per_run_summary.parquet`. The pooled information export has {len(pooled):,} rows covering {pooled.statistic.nunique()} statistics at every cell; per-round estimates remain in the same Parquet table. The shared engine reports observational one-step `T_pi=I(U_t;K_(t+1)|K_t)` in bits, with a policy-redraw null, not a forced-branch causal information estimate.

The declared late-support 95% half-width target (0.02) was missed in **{insufficient} cells**. Read the per-cell `precision_sufficient` flag; do not treat all maps as equally precise. Raw plug-in CMI is positively biased at finite sample sizes in the pilot, and the exact epistemic conditioning can be sparse. Episode-bootstrap intervals from the shared engine are available in the information table, but their coverage was not independently calibrated for all metrics. Extra entropy and available-susceptibility outputs have no episode-level confidence interval.

`derived/state_local_exact_K.csv` aggregates the already computed state-local **response** table by exact target count and evidence conditioning, with action-arm counts and identified occupancy. It does not contain state-local information: outcome histograms per state were not retained. Per-cell full raw rounds and micro-slots were deleted after information and response analyses were sealed, following the later retention request. All 128 per-run physical summaries and eight deterministic example trajectories per cell remain in the full result root; this ZIP includes 24 selected example cells. The full result root is `{ROOT}`; the independent 512-episode calibration bank is `{CAL}`.

This archive provides estimates and plots, not a claim that every information effect or efficiency ratio is reliable. No full-grid theory curves or theory-minus-simulation panels belong to this live-board study.
""")
    (stage / "TPI_SENSITIVITY.md").write_text("""# What the T_pi sensitivity pilot tested

`T_pi` is the observational, one-round-lag `I(U_t;K_(t+1)|K_t)` in **bits**, implemented as `round_target_actuation_cmi`. It is not a forced action/silence branch estimate and is not identical to a causal response.

The independent 512-episode bank covers 12 fixed physical conditions. The 19-null-draw run partitions that bank into disjoint groups of 8, 16, 32, 64, 128, 256 and 512 complete episodes; the 99-null-draw run starts at 32 episodes and continues through 512. Sizes reuse the same bank, so curves are nested across sizes. The 128-episode pilot is a separate bank at the same 12 conditions. Estimators include raw T_pi, its policy-redraw null mean, raw-minus-null, epistemic CMI variants, sensing MI, and susceptibility. `calibration/episode_sensitivity.csv` preserves every disjoint-group estimate; `calibration/episode_sensitivity_summary.csv` gives descriptive means and standard deviations. The PDF plots the 99-draw T_pi series versus independent episode count.

This is **sensitivity analysis**, but it does not establish calibrated bias, RMSE, interval coverage, false-positive rate, power, or a universal required sample size: there is no independent high-precision alternative reference at every condition, and the number of independent groups shrinks at large sample sizes (only one at 512). The policy-redraw null keeps each logged state/sensor propensity while redrawing U; it is a conditional observational reference, not a physical no-effect intervention. At B=0 the gate is virtual and posting is absent, but raw observational T_pi can still reflect state selection and finite-sample bias. Interpret raw, null and excess separately.
""")
    (stage / "DEFINITIONS.md").write_text("""# Key definitions and units

- `K_t`: count of agents supporting the controller target at the pre-action state; `X_t=K_t/24`.
- `U_t`: logged Bernoulli controller gate; at `B=0` it is virtual and posts nothing. Its known propensity is sigmoid[4(0.5-Y_t/q_c)].
- `T_pi`: observational `I(U_t;K_(t+1)|K_t)` in base-2 bits. `round_target_actuation_cmi` is its shared-engine name. Additional CMI rows condition on exact memory histogram or declared coarse evidence partitions.
- `chi(K)`: mean one-round target-fraction change under U=1 minus that under U=0, matched on pre-action K. `round_target_susceptibility` is occupancy weighted over identified K states; `state_local_exact_K.csv` also reports local and evidence-conditioned response point estimates. Units: target-fraction change per round.
- `tau_ipw_lag_h`: mean `[U/e-(1-U)/(1-e)](X_(t+h)-X_t)` for h=1,2,3, with whole-episode standard errors. Later lags include subsequent usual-policy actions.
- `available_susceptibility_ratio`: sum of IPW one-round response contributions divided by sum of available mass `1-X_t` over nonsaturated rows. It has no confidence interval in this export.
- `eta_IR`: existing occupancy-level information-response ratio in `derived_diagnostics.jsonl`; the numerator uses identified state responses and action mix, and the denominator is raw T_pi. Undefined ratios stay null/NaN. Sensing information in that diagnostic is in nats, unlike the shared-engine MI in bits.
- `late_target_share`: per-episode mean of X_41..X_60, then mean/SD/SE across 128 independent episodes. Costs are actual posts, separate from nominal B. `gain_vs_b0.csv` uses matched physical cells but unpaired episode uncertainty.
- Raw/null/excess information: `estimate`, `null_mean`, and `estimate_minus_null` in the information Parquet. The null is state-dependent policy redraw for actuation CMI; see the included analyzer code for exact estimator settings and partitions.

Each cell combines one rho, B, q, q_c, beta regime, and controller target. Exact settings, horizon, initial fact allocation, live-board clock, forgetting, and sampling are in `config/full_grid.yaml` and `docs/study_spec.md`.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    stage = output.with_suffix(".stage")
    if output.exists() or stage.exists():
        raise FileExistsError("refusing to replace an existing archive or stage")
    config = load_config(REPO / "configs/santa_fe/live_board_full.yaml")
    if len(config.cells) != 2916:
        raise ValueError("unexpected physical grid")
    for name in ("cells_aggregation.json", "information/llm_parallel/analysis_config.json",
                 "plots/live_board_maps.pdf"):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)
    stage.mkdir(parents=True)
    metadata = cell_metadata()
    if len(metadata) != 2916:
        raise ValueError("incomplete physical summary")
    export_derived(stage, metadata)
    export_calibration(stage)
    selected_examples(stage, metadata)
    write_docs(stage, metadata)
    write_figure_index(stage)
    for source, name in (
        (ROOT / "config.yaml", "config/full_grid.yaml"),
        (CAL / "config.yaml", "config/calibration_512.yaml"),
        (ROOT / "execution_plan.json", "config/execution_plan.json"),
        (ROOT / "cells_aggregation.json", "config/cells_aggregation.json"),
        (ROOT / "information/llm_parallel/analysis_config.json", "config/information_analysis_config.json"),
        (ROOT / "summaries/cell_metric_summary.csv", "summaries/cell_metric_summary.csv"),
        (ROOT / "summaries/per_run_summary.parquet", "summaries/per_run_summary.parquet"),
        (ROOT / "summaries/gain_vs_b0.csv", "summaries/gain_vs_b0.csv"),
        (ROOT / "information/llm_parallel/round_information_estimates.parquet", "information/round_information_estimates.parquet"),
        (ROOT / "information/llm_parallel/round_information_report.md", "information/round_information_report.md"),
        (ROOT / "plots/live_board_maps.pdf", "figures/live_board_maps.pdf"),
        (ROOT / "plots/live_board_maps.csv", "figures/live_board_maps.csv"),
        (REPO / "docs/tdd/santa_fe/SANTA_FE_LIVE_BOARD_STUDY.md", "docs/study_spec.md"),
        (REPO / "docs/handoff/santa_fe_live_board_launch_20260927.md", "docs/launch_handoff.md"),
        (REPO / "docs/handoff/santa_fe_live_board_analysis_guide_20260928.md", "ANALYSIS_GUIDE.md"),
        (REPO / "src/santa_fe/live_board_package.py", "code/live_board_package.py"),
        (REPO / "src/santa_fe/live_board_plots.py", "code/live_board_plots.py"),
        (REPO / "src/santa_fe/live_board_metrics.py", "code/live_board_metrics.py"),
        (REPO / "src/santa_fe/live_board_calibration.py", "code/live_board_calibration.py"),
        (REPO / "src/santa_fe/live_board_postprocess.py", "code/live_board_postprocess.py"),
        (REPO / "src/santa_fe/llm_parallel.py", "code/llm_parallel.py"),
        (REPO / "src/santa_fe/cluster.py", "code/cluster.py"),
    ):
        copy(stage, source, name)
    manifest = {"package_format": 1, "model": "santa_fe_live_board_v4",
        "physical_cells": len(metadata), "independent_episodes_per_cell": 128,
        "theory_included": False, "full_result_root": str(ROOT),
        "full_calibration_root": str(CAL),
        "simulator_code_sha256": json.loads((ROOT / "execution_plan.json").read_text())["simulator_sha256"],
        "information_engine_sha256": json.loads((ROOT / "execution_plan.json").read_text())["information_engine_sha256"],
        "files": []}
    for file in sorted(stage.rglob("*")):
        if file.is_file():
            manifest["files"].append({"path": str(file.relative_to(stage)), "bytes": file.stat().st_size,
                                      "sha256": sha(file)})
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    temp = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", allowZip64=True) as archive:
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                mode = zipfile.ZIP_STORED if file.suffix in {".parquet", ".pdf"} else zipfile.ZIP_DEFLATED
                archive.write(file, file.relative_to(stage), compress_type=mode, compresslevel=6)
    with zipfile.ZipFile(temp) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP checksum failed: {bad}")
    temp.rename(output)
    shutil.rmtree(stage)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "sha256": sha(output), "files": len(manifest["files"]) + 1}))


if __name__ == "__main__":
    main()
