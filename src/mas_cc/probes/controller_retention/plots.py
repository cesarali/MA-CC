"""One descriptive controller-retention figure per configured model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .analysis import CellEffect, effect_index, r_exposure
from .design import ONE_SLOT, TWO_SLOTS

DELTA_LIMITS = (-1.05, 1.05)


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def controller_retention_plot(
    effects: Sequence[CellEffect], model_label: str, output_dir: Path
) -> list[Path]:
    mine = [effect for effect in effects if effect.key[0] == model_label]
    if not mine:
        return []
    plt = _matplotlib()
    index = effect_index(mine)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), squeeze=False)
    for row, depth in enumerate((1, 2)):
        for column, target in enumerate(("truth", "false")):
            axis = axes[row][column]
            one = [index.get((model_label, depth, target, q, ONE_SLOT)) for q in (2, 3)]
            xs = [q for q, effect in zip((2, 3), one) if effect is not None]
            ys = [effect.delta_c for effect in one if effect is not None]
            if xs:
                axis.plot(xs, ys, "-o", label="one_slot")
            two = index.get((model_label, depth, target, 3, TWO_SLOTS))
            if two is not None:
                axis.scatter(
                    [3], [two.delta_c], marker="s", s=60, label="two_slots (probe-only)"
                )
                rescue = r_exposure(index, two.key)
                if rescue is not None:
                    axis.annotate(
                        f"rescue {rescue:+.3f}",
                        (3, two.delta_c),
                        xytext=(-70, 12),
                        textcoords="offset points",
                    )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_ylim(*DELTA_LIMITS)
            axis.set_xticks((2, 3))
            axis.set_xlabel("q (visible social slots)")
            axis.set_ylabel("Delta_C")
            axis.set_title(f"L={depth}, target={target}")
            axis.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    figure.suptitle(f"Controller retention — {model_label}")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"controller_retention_{_slug(model_label)}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path]


def render_all(
    effects: Sequence[CellEffect],
    model_labels: Sequence[str],
    output_dir: Path,
) -> dict[str, list[Path]]:
    return {
        label: controller_retention_plot(effects, label, output_dir)
        for label in model_labels
    }


__all__ = ["DELTA_LIMITS", "controller_retention_plot", "render_all"]
