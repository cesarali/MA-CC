"""Read-only, manifest-driven reports over standardized study analyses."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml

from .table_io import read_scientific_table, retained_table_path


DEFAULT_FACETS = (
    "target_semantics",
    "epistemic_persistence",
    "social_group_size",
    "task_id",
    "beta",
    "threshold",
    "sensor_sample_size",
    "receiver_epistemic_disposition",
    "controller_evidence_strategy",
    "controller_actuation_mode",
    "controller_communication_policy",
)
TECHNICAL_COLUMNS = {
    "study_id",
    "source_run_id",
    "source_run_path",
    "source_cell_id",
    "cell_id",
    "metric",
    "source_metric",
    "estimate",
    "ci_low",
    "ci_high",
    "confidence",
    "null_type",
    "null_mean",
    "null_std",
    "p_value",
    "null_permutations",
    "bootstrap_resamples",
    "n_observations",
    "n_episodes",
    "units",
    "support_status",
    "phase_status",
    "analysis_hash",
    "grouping_json",
    "conditioning_json",
    "dependencies_json",
    "target_fraction_bin_index",
    "target_fraction_bin_lower",
    "target_fraction_bin_upper",
    "target_fraction_bin_center",
    "target_fraction_bin_count",
    "intervention_budget",
    "control.options.intervention_budget",
    "game.options.epistemic_persistence",
    "aggregation_scope",
    "aggregation_weight",
    "descriptive_only",
}


@dataclass(frozen=True)
class ReportResult:
    report_id: str
    source_analysis: Path
    output_dir: Path
    markdown: Path
    latex: Path
    pdf: Path
    manifest: Path
    figures: tuple[Path, ...]
    provisional: bool


@dataclass
class PhaseResult:
    metric_id: str
    label: str
    mode: str
    figures: list[Path]
    status: str
    reason: str | None
    displayed_values: int
    source_table: str | None
    panel_count: int


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list")
    return value


def _resolve_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid report YAML {path}: {exc}") from exc
    config = dict(_mapping(raw, "report configuration"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("report schema_version must be 1")
    report = _mapping(config.get("report"), "report")
    for field in ("id", "title", "source_analysis", "output_dir"):
        if not str(report.get(field, "")).strip():
            raise ValueError(f"report.{field} must be non-empty")
    sections = _sequence(config.get("sections", ()), "sections")
    ids = [str(_mapping(section, "section").get("id", "")) for section in sections]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("report section ids must be non-empty and unique")
    return config


def _analysis_root(path: Path) -> Path:
    candidate = path / "analysis" if (path / "analysis").is_dir() else path
    if not (candidate / "analysis_manifest.json").is_file():
        raise ValueError(
            f"source is not a standardized analysis directory: {candidate}"
        )
    if not (candidate / "validation.json").is_file():
        raise ValueError(f"source analysis has no validation.json: {candidate}")
    if not (candidate / "tables").is_dir():
        raise ValueError(f"source analysis has no tables directory: {candidate}")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    return dict(_mapping(value, str(path)))


def _normalize_semantics(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"false", "wrong", "deceptive"}:
        return "false"
    if text in {"truth", "true", "correct"}:
        return "truth"
    if text in {"none", "nan", "null"}:
        return "none"
    return text


class AnalysisPackage:
    def __init__(self, root: Path) -> None:
        self.root = _analysis_root(root)
        self.manifest = _read_json(self.root / "analysis_manifest.json")
        self.validation = _read_json(self.root / "validation.json")
        self.tables_dir = self.root / "tables"
        self._cache: dict[str, pd.DataFrame] = {}
        listed = self.manifest.get("tables", ())
        self.listed_tables = {Path(str(item)).stem for item in listed}

    def has_table(self, name: str) -> bool:
        return retained_table_path(self.tables_dir, name) is not None

    def table(self, name: str, *, required: bool = True) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name].copy()
        path = retained_table_path(self.tables_dir, name)
        if path is None:
            if required:
                raise ValueError(f"source analysis has no table named {name}")
            return pd.DataFrame()
        frame = read_scientific_table(path)
        if "target_semantics" in frame:
            frame["target_semantics"] = frame["target_semantics"].map(
                _normalize_semantics
            )
        self._cache[name] = frame
        return frame.copy()


class SourceLedger:
    """Internal proof that every displayed scientific value came from a row."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add(
        self,
        *,
        location: str,
        table: str,
        row_index: Any,
        field: str,
        value: Any,
        formula: str | None = None,
    ) -> None:
        if table == "" or row_index is None or field == "":
            raise ValueError(f"number at {location} lacks a source table row")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"number at {location} is not finite")
        self.entries.append(
            {
                "location": location,
                "table": table,
                "row_index": str(row_index),
                "field": field,
                "formula": formula,
                "rendered_value": number,
            }
        )


def _attach_cell_coordinates(frame: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "cell_id" not in frame or cells.empty:
        return frame
    existing = set(frame.columns)
    candidates = [
        column
        for column in cells.columns
        if column == "cell_id"
        or (
            column not in existing
            and column
            not in {
                "study_id",
                "source_run_id",
                "source_run_path",
                "source_cell_id",
                "config_hash",
                "resolved_config_hash",
                "recorded_resolved_config_hash",
                "expected_episodes",
                "completed_episodes",
                "failed_episodes",
                "sealed",
            }
        )
    ]
    return frame.merge(cells[candidates], on="cell_id", how="left")


def _metric_rows(
    package: AnalysisPackage,
    spec: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[pd.DataFrame, str | None, str | None]:
    if mode == "resolved":
        source = str(spec.get("source", ""))
        metric = spec.get("metric")
        value = str(spec.get("value", "estimate"))
    else:
        source = str(spec.get("aggregation_source", ""))
        metric = spec.get("aggregation_metric")
        value = str(spec.get("aggregation_value", spec.get("value", "estimate")))
    if not source:
        return pd.DataFrame(), None, f"no {mode} source was configured"
    frame = package.table(source, required=False)
    if frame.empty:
        return frame, source, f"{source}.parquet is unavailable or empty"
    if metric is not None:
        if "metric" not in frame:
            return pd.DataFrame(), source, f"{source}.parquet has no metric column"
        frame = frame[frame["metric"].astype(str) == str(metric)].copy()
    if frame.empty:
        return frame, source, f"metric {metric!r} is unavailable in {source}.parquet"
    if value not in frame:
        return pd.DataFrame(), source, f"value column {value!r} is unavailable"
    frame = _attach_cell_coordinates(frame, package.table("cells", required=False))
    frame["_report_value"] = pd.to_numeric(frame[value], errors="coerce")
    frame["_source_row"] = frame.index
    frame["_source_field"] = value
    return frame, source, None


def _bin_contract(
    frame: pd.DataFrame, *, minimum_bins: int
) -> tuple[int | None, str | None]:
    required = {
        "target_fraction_bin_index",
        "target_fraction_bin_lower",
        "target_fraction_bin_upper",
        "target_fraction_bin_center",
    }
    if not required.issubset(frame.columns):
        return None, "state-local target-fraction bin metadata are unavailable"
    indices = pd.to_numeric(frame["target_fraction_bin_index"], errors="coerce")
    if indices.notna().sum() == 0:
        return None, "the source contains no state-local rows"
    if "target_fraction_bin_count" in frame:
        counts = pd.to_numeric(frame["target_fraction_bin_count"], errors="coerce")
        finite_counts = sorted({int(value) for value in counts.dropna()})
        if len(finite_counts) != 1:
            return None, "state-local rows do not use one consistent bin count"
        bins = finite_counts[0]
    else:
        bins = int(indices.max()) + 1
    if bins < minimum_bins:
        return (
            None,
            f"only {bins} state bins exist; at least {minimum_bins} are required",
        )
    lower = pd.to_numeric(frame["target_fraction_bin_lower"], errors="coerce")
    upper = pd.to_numeric(frame["target_fraction_bin_upper"], errors="coerce")
    expected = set(range(bins))
    observed = {int(value) for value in indices.dropna()}
    if not observed.issubset(expected):
        return None, "state-bin indices fall outside the declared bin count"
    retained = pd.DataFrame({"index": indices, "lower": lower, "upper": upper}).dropna()
    if any(
        not math.isclose(float(row.lower), int(row.index) / bins, abs_tol=1e-12)
        or not math.isclose(
            float(row.upper), (int(row.index) + 1) / bins, abs_tol=1e-12
        )
        for row in retained.itertuples(index=False)
    ):
        return None, "retained state-bin bounds are inconsistent with the 0-to-1 domain"
    return bins, None


def _facet_columns(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    preferred: Sequence[str],
    budget: str,
) -> tuple[str, ...]:
    facets: list[str] = []
    for column in preferred:
        frame_varies = column in frame and frame[column].nunique(dropna=False) > 1
        cells_vary = column in cells and cells[column].nunique(dropna=False) > 1
        if (column in frame or column in cells) and (frame_varies or cells_vary):
            facets.append(column)
    key = [*facets, budget, "target_fraction_bin_index"]
    if not frame.duplicated(key).any():
        return tuple(facets)
    for column in frame.columns:
        if (
            column in facets
            or column in TECHNICAL_COLUMNS
            or column.startswith("_")
            or column not in frame
            or frame[column].nunique(dropna=False) <= 1
        ):
            continue
        candidate = [*facets, column, budget, "target_fraction_bin_index"]
        if not frame.duplicated(candidate).all():
            facets.append(column)
        if not frame.duplicated([*facets, budget, "target_fraction_bin_index"]).any():
            return tuple(facets)
    duplicates = int(
        frame.duplicated(
            [*facets, budget, "target_fraction_bin_index"], keep=False
        ).sum()
    )
    raise ValueError(
        f"{duplicates} source rows still share report coordinates after automatic faceting"
    )


def _normalized_panel_key(key: Any, size: int) -> tuple[str, ...]:
    values = key if isinstance(key, tuple) else (key,)
    if len(values) != size:
        raise ValueError("panel key does not match the selected facets")
    return tuple("<missing>" if pd.isna(value) else str(value) for value in values)


def _report_panel_groups(
    package: AnalysisPackage,
    frame: pd.DataFrame,
    facets: Sequence[str],
) -> list[tuple[Any, pd.DataFrame]]:
    actual = _panel_groups(frame, facets)
    if not facets:
        return actual
    by_key = {
        _normalized_panel_key(key, len(facets)): (key, group) for key, group in actual
    }
    cells = package.table("cells", required=False)
    if not cells.empty and all(column in cells for column in facets):
        for values in (
            cells[list(facets)].drop_duplicates().itertuples(index=False, name=None)
        ):
            key: Any = values[0] if len(values) == 1 else values
            normalized = _normalized_panel_key(key, len(facets))
            by_key.setdefault(normalized, (key, frame.iloc[0:0].copy()))
    return [by_key[key] for key in sorted(by_key)]


def _panel_label(facets: Sequence[str], key: Any) -> str:
    if not facets:
        return "all available cells"
    values = key if isinstance(key, tuple) else (key,)
    return ", ".join(
        f"{column}={value}" for column, value in zip(facets, values, strict=True)
    )


def _panel_groups(
    frame: pd.DataFrame, facets: Sequence[str]
) -> list[tuple[Any, pd.DataFrame]]:
    if not facets:
        return [("all", frame)]
    grouper: Any = facets[0] if len(facets) == 1 else list(facets)
    return list(frame.groupby(grouper, dropna=False, sort=True))


def _fallback_panels(
    package: AnalysisPackage, preferred: Sequence[str], *, mode: str
) -> list[str]:
    cells = package.table("cells", required=False)
    if cells.empty:
        return ["all available cells"]
    columns = [
        column
        for column in preferred
        if column in cells
        and cells[column].nunique(dropna=False) > 1
        and not (mode == "aggregated" and column == "epistemic_persistence")
    ]
    if not columns:
        return ["all available cells"]
    return [_panel_label(columns, key) for key, _ in _panel_groups(cells, columns)]


def _budgets(package: AnalysisPackage, frame: pd.DataFrame, budget: str) -> list[float]:
    values = (
        pd.to_numeric(frame.get(budget), errors="coerce")
        if budget in frame
        else pd.Series(dtype=float)
    )
    if values.notna().any():
        return sorted(float(value) for value in values.dropna().unique())
    cells = package.table("cells", required=False)
    if budget in cells:
        values = pd.to_numeric(cells[budget], errors="coerce")
        if values.notna().any():
            return sorted(float(value) for value in values.dropna().unique())
    return [0.0]


def _display_grid(
    group: pd.DataFrame,
    budgets: Sequence[float],
    bins: int,
    budget: str,
) -> tuple[np.ndarray, dict[tuple[int, int], pd.Series]]:
    grid = np.full((bins, len(budgets)), np.nan, dtype=float)
    rows: dict[tuple[int, int], pd.Series] = {}
    budget_index = {float(value): index for index, value in enumerate(budgets)}
    for _, row in group.iterrows():
        raw_budget = pd.to_numeric(pd.Series([row.get(budget)]), errors="coerce").iloc[
            0
        ]
        raw_bin = pd.to_numeric(
            pd.Series([row.get("target_fraction_bin_index")]), errors="coerce"
        ).iloc[0]
        if pd.isna(raw_budget) or pd.isna(raw_bin):
            continue
        coordinate = (int(raw_bin), budget_index.get(float(raw_budget), -1))
        if coordinate[1] < 0:
            continue
        if coordinate in rows:
            raise ValueError("duplicate source rows reached the display grid")
        rows[coordinate] = row
        support = str(row.get("support_status", "adequate"))
        phase = str(row.get("phase_status", "adequate"))
        value = float(row.get("_report_value", math.nan))
        if support == "unsupported" or phase in {
            "structural_cell_not_run",
            "state_not_visited",
            "insufficient_estimator_support",
        }:
            continue
        if math.isfinite(value):
            grid[coordinate] = value
    return grid, rows


def _normalization(values: np.ndarray, scale: Any):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, "viridis"
    kind = (
        str(scale.get("kind", "linear")) if isinstance(scale, Mapping) else str(scale)
    )
    if kind == "symlog":
        limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-12)
        linthresh = float(scale.get("linthresh", max(limit * 0.01, 1e-6)))
        return SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit), "RdBu_r"
    if kind == "diverging":
        limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-12)
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), "RdBu_r"
    return None, "viridis"


def _render_phase_metric(
    package: AnalysisPackage,
    spec: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    mode: str,
    figures_dir: Path,
    ledger: SourceLedger,
) -> PhaseResult:
    metric_id = str(spec.get("id", "metric"))
    label = str(spec.get("label", metric_id))
    minimum_bins = int(
        _mapping(suite.get("axes", {}), "phase axes").get("minimum_state_bins", 8)
    )
    if minimum_bins < 8:
        raise ValueError("state_budget_phase_suite requires at least eight state bins")
    budget = str(
        _mapping(suite.get("axes", {}), "phase axes").get(
            "budget", "intervention_budget"
        )
    )
    preferred = tuple(
        str(value)
        for value in _sequence(
            _mapping(
                suite.get("facets", {"preferred": list(DEFAULT_FACETS)}), "phase facets"
            ).get("preferred", list(DEFAULT_FACETS)),
            "phase preferred facets",
        )
    )
    if mode == "aggregated":
        preferred = tuple(
            value for value in preferred if value != "epistemic_persistence"
        )
    frame, source, reason = _metric_rows(package, spec, mode=mode)
    bins = minimum_bins
    if reason is None:
        bins, reason = _bin_contract(frame, minimum_bins=minimum_bins)
    budgets = _budgets(package, frame, budget)
    if reason is not None or bins is None:
        panel_labels = _fallback_panels(package, preferred, mode=mode)
        return _render_unavailable_phase(
            metric_id,
            label,
            mode,
            reason or "state-local values are unavailable",
            source,
            panel_labels,
            budgets,
            minimum_bins,
            figures_dir,
        )
    try:
        facets = _facet_columns(
            frame,
            package.table("cells", required=False),
            preferred=preferred,
            budget=budget,
        )
    except ValueError as exc:
        return _render_unavailable_phase(
            metric_id,
            label,
            mode,
            str(exc),
            source,
            _fallback_panels(package, preferred, mode=mode),
            budgets,
            bins,
            figures_dir,
        )
    groups = _report_panel_groups(package, frame, facets)
    max_panels = int(suite.get("max_panels_per_figure", 6))
    if max_panels < 1:
        raise ValueError("max_panels_per_figure must be positive")
    paths: list[Path] = []
    displayed = 0
    scale = spec.get("color_scale", "linear")
    for page, offset in enumerate(range(0, len(groups), max_panels), start=1):
        chunk = groups[offset : offset + max_panels]
        columns = min(3, len(chunk))
        rows_count = math.ceil(len(chunk) / columns)
        figure, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(4.3 * columns, 3.6 * rows_count),
            squeeze=False,
            constrained_layout=True,
        )
        page_grids: list[np.ndarray] = []
        prepared: list[tuple[Any, np.ndarray, dict[tuple[int, int], pd.Series]]] = []
        for key, group in chunk:
            grid, source_rows = _display_grid(group, budgets, bins, budget)
            page_grids.append(grid)
            prepared.append((key, grid, source_rows))
        joined = np.concatenate([grid.ravel() for grid in page_grids])
        norm, cmap_name = _normalization(joined, scale)
        cmap = plt.get_cmap(cmap_name).with_extremes(bad="#bdbdbd")
        image = None
        for axis, (key, grid, source_rows) in zip(axes.flat, prepared, strict=False):
            image = axis.imshow(
                np.ma.masked_invalid(grid),
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
            )
            axis.set_title(_panel_label(facets, key), fontsize=8)
            axis.set_xticks(range(len(budgets)), [f"{value:g}" for value in budgets])
            axis.set_xlabel("intervention budget b")
            axis.set_yticks(
                [-0.5, bins / 4 - 0.5, bins / 2 - 0.5, 3 * bins / 4 - 0.5, bins - 0.5],
                ["0", "0.25", "0.5", "0.75", "1"],
            )
            axis.set_ylabel("target fraction x")
            for (row_index, column_index), row in source_rows.items():
                value = grid[row_index, column_index]
                if not math.isfinite(float(value)):
                    continue
                displayed += 1
                ledger.add(
                    location=f"figure:{metric_id}:{mode}:page-{page}",
                    table=source or "",
                    row_index=row.get("_source_row"),
                    field=str(row.get("_source_field", "estimate")),
                    value=value,
                )
        for axis in axes.flat[len(chunk) :]:
            axis.set_visible(False)
        if image is not None:
            figure.colorbar(
                image, ax=list(axes.flat[: len(chunk)]), shrink=0.78, label=label
            )
        figure.suptitle(f"{label}: {mode} state-by-budget view")
        path = figures_dir / f"{metric_id}_{mode}_page_{page}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    if displayed == 0:
        for path in paths:
            path.unlink(missing_ok=True)
        return _render_unavailable_phase(
            metric_id,
            label,
            mode,
            "all source rows are missing or unsupported",
            source,
            [_panel_label(facets, key) for key, _ in groups],
            budgets,
            bins,
            figures_dir,
        )
    return PhaseResult(
        metric_id,
        label,
        mode,
        paths,
        "available",
        None,
        displayed,
        source,
        len(groups),
    )


def _render_unavailable_phase(
    metric_id: str,
    label: str,
    mode: str,
    reason: str,
    source: str | None,
    panel_labels: Sequence[str],
    budgets: Sequence[float],
    bins: int,
    figures_dir: Path,
) -> PhaseResult:
    labels = list(panel_labels) or ["all available cells"]
    columns = min(3, len(labels))
    rows_count = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(4.3 * columns, 3.5 * rows_count),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("gray").copy()
    for axis, panel in zip(axes.flat, labels, strict=False):
        grid = np.full((bins, len(budgets)), np.nan)
        axis.imshow(
            np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap
        )
        axis.set_title(panel, fontsize=8)
        axis.set_xticks(range(len(budgets)), [f"{value:g}" for value in budgets])
        axis.set_xlabel("intervention budget b")
        axis.set_yticks(
            [-0.5, bins / 4 - 0.5, bins / 2 - 0.5, 3 * bins / 4 - 0.5, bins - 0.5],
            ["0", "0.25", "0.5", "0.75", "1"],
        )
        axis.set_ylabel("target fraction x")
        axis.text(
            0.5,
            0.5,
            "UNAVAILABLE",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#555555",
            weight="bold",
        )
    for axis in axes.flat[len(labels) :]:
        axis.set_visible(False)
    figure.suptitle(f"{label}: {mode} state-by-budget view")
    path = figures_dir / f"{metric_id}_{mode}_unavailable.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return PhaseResult(
        metric_id,
        label,
        mode,
        [path],
        "unavailable",
        reason,
        0,
        source,
        len(labels),
    )


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _validation_markdown(validation: Mapping[str, Any], ledger: SourceLedger) -> str:
    counts = _mapping(validation.get("counts", {}), "validation counts")
    rows = []
    for field, label in (
        ("expected_cells", "Expected cells"),
        ("found_cells", "Found cells"),
        ("expected_episodes", "Expected episodes"),
        ("completed_episodes", "Completed episodes"),
        ("failed_episodes", "Failed episodes"),
        ("aborted_episodes", "Aborted episodes"),
        ("round_rows", "Round rows"),
        ("micro_slot_rows", "Micro-slot rows"),
    ):
        if field not in counts:
            continue
        value = int(counts[field])
        ledger.add(
            location=f"validation:{field}",
            table="validation.json",
            row_index="counts",
            field=field,
            value=value,
        )
        rows.append(f"| {label} | {value} |")
    return "\n".join(["| Validation item | Value |", "|---|---:|", *rows])


def _phase_markdown(results: Sequence[PhaseResult], output_dir: Path) -> str:
    blocks = []
    by_metric: dict[str, list[PhaseResult]] = {}
    for result in results:
        by_metric.setdefault(result.metric_id, []).append(result)
    for metric_results in by_metric.values():
        blocks.append(f"### {metric_results[0].label}")
        for result in metric_results:
            blocks.append(f"#### {result.mode.capitalize()} view")
            if result.status == "unavailable":
                blocks.append(
                    "This requested view is unavailable in the source aggregation package. "
                    + str(result.reason)
                    + ". The report did not reconstruct a scientific estimator from raw observations."
                )
            for path in result.figures:
                relative = path.relative_to(output_dir)
                blocks.append(
                    f"![{result.label}: {result.mode}]({relative.as_posix()})"
                )
            blocks.append(
                "Gray cells have no displayed estimate. They may be unvisited, "
                "unsupported, structurally absent, or otherwise missing; gray does not mean zero."
            )
    return "\n\n".join(blocks)


def _write_markdown(
    output: Path,
    config: Mapping[str, Any],
    package: AnalysisPackage,
    phase_results: Sequence[PhaseResult],
    ledger: SourceLedger,
) -> Path:
    provisional = not bool(package.validation.get("complete", False))
    status = "INCOMPLETE / PROVISIONAL" if provisional else "COMPLETE"
    errors = package.validation.get("errors", ())
    unavailable = [result for result in phase_results if result.status == "unavailable"]
    body = [
        f"# {_mapping(config['report'], 'report')['title']}",
        "",
        f"**Status: {status}**",
        "",
        f"Source analysis package: `{package.root}`",
        "",
        "## Executive summary",
        "",
        "This report is generated only from the standard aggregation package. "
        "It does not rerun episodes or reconstruct missing estimators from raw observations.",
        "",
        "## Validation and completeness",
        "",
        _validation_markdown(package.validation, ledger),
    ]
    if provisional:
        body.extend(
            [
                "",
                "The source study is incomplete. Every result in this report is provisional.",
            ]
        )
    if errors:
        body.extend(["", "Validation errors:", "", *[f"- {item}" for item in errors]])
    body.extend(
        [
            "",
            "## State-by-budget phase diagrams",
            "",
            "Each state-local target-state axis spans 0 to 1 and uses at least eight pre-existing "
            "aggregation bins. Resolved views retain every varying scientific coordinate. "
            "Aggregated views use only aggregation-produced tables; the report does not "
            "average estimator rows.",
            "",
            _phase_markdown(phase_results, output),
            "",
            "## Unavailable quantities",
            "",
        ]
    )
    if unavailable:
        body.extend(
            f"- **{item.label}, {item.mode}:** {item.reason}" for item in unavailable
        )
    else:
        body.append("- None.")
    body.extend(
        [
            "",
            "## Methods and limitations",
            "",
            "Scientific estimates, uncertainty intervals, null summaries, and support "
            "classifications come from aggregation tables. The report performs only "
            "selection, row-level arithmetic, support masking, and reshaping. A symmetric "
            "logarithmic color scale changes the display of susceptibility but not its values.",
            "",
            "No-control cells can provide occupancy and outcomes. Controller-action metrics "
            "are gray when the aggregation package contains no corresponding estimate.",
        ]
    )
    path = output / "report.md"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _write_latex(
    output: Path,
    config: Mapping[str, Any],
    package: AnalysisPackage,
    phase_results: Sequence[PhaseResult],
) -> Path:
    report = _mapping(config["report"], "report")
    provisional = not bool(package.validation.get("complete", False))
    status = "INCOMPLETE / PROVISIONAL" if provisional else "COMPLETE"
    counts = _mapping(package.validation.get("counts", {}), "validation counts")
    figures = []
    for result in phase_results:
        note = (
            f"Unavailable: {result.reason}. " if result.status == "unavailable" else ""
        )
        caption = (
            f"{result.label}, {result.mode} view. {note}Gray cells have no displayed "
            "estimate and do not mean zero."
        )
        for path in result.figures:
            figures.append(
                "\\begin{figure}[H]\n"
                "\\centering\n"
                f"\\includegraphics[width=0.98\\textwidth]{{figures/{_latex_escape(path.name)}}}\n"
                f"\\caption{{{_latex_escape(caption)}}}\n"
                "\\end{figure}"
            )
    count_rows = []
    for field, label in (
        ("expected_cells", "Expected cells"),
        ("found_cells", "Found cells"),
        ("expected_episodes", "Expected episodes"),
        ("completed_episodes", "Completed episodes"),
        ("failed_episodes", "Failed episodes"),
        ("aborted_episodes", "Aborted episodes"),
    ):
        if field in counts:
            count_rows.append(f"{_latex_escape(label)} & {int(counts[field])} \\\\")
    text = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{booktabs,graphicx,float,caption,microtype,xcolor,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.45em}}
\title{{{_latex_escape(report["title"])}}}
\author{{Standard MA-CC aggregation report}}
\date{{}}
\begin{{document}}
\maketitle
\textbf{{Status: {_latex_escape(status)}}}

Source analysis package: \texttt{{{_latex_escape(package.root)}}}

\section{{Executive summary}}
This report uses only the standard aggregation package. It does not rerun episodes or reconstruct missing scientific estimators from raw observations.

\section{{Validation and completeness}}
\begin{{tabular}}{{lr}}
\toprule
Validation item & Value \\
\midrule
{chr(10).join(count_rows)}
\bottomrule
\end{{tabular}}

{"The source study is incomplete, so every result is provisional." if provisional else "The source study is complete according to validation.json."}

\section{{State-by-budget phase diagrams}}
The target-state axis spans zero to one with at least eight pre-existing aggregation bins. Resolved views retain varying scientific coordinates. Aggregated views use only aggregation-produced tables.

{chr(10).join(figures)}

\section{{Methods and limitations}}
Scientific estimates, intervals, nulls, and support classifications come from aggregation tables. Gray cells are missing or unsupported, not zero. Symmetric-log susceptibility panels change only the display scale.
\end{{document}}
"""
    path = output / "report.tex"
    path.write_text(text, encoding="utf-8")
    return path


def _compile_pdf(output: Path, latex: Path) -> Path:
    executable = shutil.which("latexmk")
    if executable is None:
        raise ValueError("latexmk is required to compile the report PDF")
    completed = subprocess.run(
        [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error", latex.name],
        cwd=output,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        log = (completed.stdout + "\n" + completed.stderr)[-8000:]
        raise ValueError("LaTeX report compilation failed:\n" + log)
    pdf = output / "report.pdf"
    generated = output / f"{latex.stem}.pdf"
    if generated != pdf:
        generated.replace(pdf)
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise ValueError("LaTeX completed without producing report.pdf")
    for suffix in ("aux", "fdb_latexmk", "fls", "log", "out", "toc"):
        (output / f"{latex.stem}.{suffix}").unlink(missing_ok=True)
    return pdf


def build_study_report(config_path: str | Path) -> ReportResult:
    """Build Markdown, LaTeX, PDF, figures, and a manifest from aggregation rows."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"report configuration does not exist: {path}")
    config = _load_config(path)
    report = _mapping(config["report"], "report")
    source = _resolve_path(
        report["source_analysis"], base=path.parent, label="report.source_analysis"
    )
    output = _resolve_path(
        report["output_dir"], base=path.parent, label="report.output_dir"
    )
    package = AnalysisPackage(source)
    try:
        output.relative_to(package.root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "report.output_dir must not be inside the source analysis package"
        )

    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    figures = staging / "figures"
    figures.mkdir(parents=True)
    ledger = SourceLedger()
    phase_results: list[PhaseResult] = []
    sections = _sequence(config.get("sections", ()), "sections")
    for section_raw in sections:
        section = _mapping(section_raw, "section")
        kind = str(section.get("kind", ""))
        if kind == "validation_summary":
            continue
        if kind != "state_budget_phase_suite":
            raise ValueError(f"unsupported report section kind: {kind!r}")
        metrics = _sequence(section.get("metrics", ()), "phase metrics")
        if not metrics:
            raise ValueError("state_budget_phase_suite requires metrics")
        views_raw = section.get("views", ("resolved", "aggregated"))
        views = [
            str(item.get("mode")) if isinstance(item, Mapping) else str(item)
            for item in _sequence(views_raw, "phase views")
        ]
        unknown = sorted(set(views) - {"resolved", "aggregated"})
        if unknown:
            raise ValueError("unknown state-budget view(s): " + ", ".join(unknown))
        for metric_raw in metrics:
            metric = _mapping(metric_raw, "phase metric")
            for mode in views:
                phase_results.append(
                    _render_phase_metric(
                        package,
                        metric,
                        section,
                        mode=mode,
                        figures_dir=figures,
                        ledger=ledger,
                    )
                )

    markdown = _write_markdown(staging, config, package, phase_results, ledger)
    latex = _write_latex(staging, config, package, phase_results)
    pdf = _compile_pdf(staging, latex)
    config_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_payload = {
        "schema_version": 1,
        "report_id": str(report["id"]),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_analysis_package": str(package.root),
        "source_analysis_status": package.manifest.get(
            "status", "complete" if package.validation.get("complete") else "incomplete"
        ),
        "source_scientific_input_identity": package.manifest.get(
            "scientific_input_identity"
        ),
        "source_analysis_hash": package.manifest.get("analysis_hash"),
        "report_config": str(path),
        "report_config_sha256": config_hash,
        "number_source_ledger_entries_validated": len(ledger.entries),
        "phase_results": [
            {
                "metric_id": result.metric_id,
                "mode": result.mode,
                "status": result.status,
                "reason": result.reason,
                "source_table": result.source_table,
                "displayed_values": result.displayed_values,
                "panel_count": result.panel_count,
                "figures": [item.name for item in result.figures],
            }
            for result in phase_results
        ],
        "outputs": ["report.md", "report.tex", "report.pdf", "figures/"],
    }
    report_manifest = staging / "report_manifest.json"
    report_manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not ledger.entries:
        raise ValueError("report validation found no source-backed scientific numbers")

    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)
    return ReportResult(
        report_id=str(report["id"]),
        source_analysis=package.root,
        output_dir=output,
        markdown=output / markdown.name,
        latex=output / latex.name,
        pdf=output / pdf.name,
        manifest=output / report_manifest.name,
        figures=tuple(sorted((output / "figures").glob("*.png"))),
        provisional=not bool(package.validation.get("complete", False)),
    )


__all__ = ["ReportResult", "build_study_report"]
