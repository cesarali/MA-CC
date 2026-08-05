"""The answer key for the empowerment statistics — family 2.

The system computes two families of mutual information. Family 1
(within-episode ``I(A_i;A_j)``) has a per-config answer key on the game
object. Family 2 — ``I(C;O)`` and ``I(C;S_{t+h} | S_t)`` — is estimated by
`analysis/pipeline.py` and, until now, never checked against anything.

The stated reason it could not be checked was that empowerment is not a
property of a config in isolation but of which conditions you swept and how
many episodes you ran per cell — an experimental-design choice. Both halves of
that are true; the conclusion does not follow. **The sweep grid *is* the input
distribution ``p(c)`` of a channel**: equal repetitions per cell make it
uniform, unequal ones make it proportional to repetitions, and either way it is
a distribution we chose and therefore know exactly. Given it,

    I(C;O) = H(O) - sum_c p(c) H(O | c)

and ``p(o|c)`` is fixed by dynamics we wrote. "Depends on a design choice" and
"is not analytically computable" are different claims.

The real obstacle was architectural, not mathematical: a single game instance
does not know what it is being swept against, so family 2's truth cannot live
on it. It lives here, one level up, taking the resolved grid and the resolved
analysis settings as its inputs — never hand-written values, or the phantom
bug returns in a new place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from mas_cc.config import GameConfig

from . import exact
from .protocols import ExactDynamics, GroundTruth, GroundTruthQuantity, SyntheticGame


@dataclass(frozen=True, slots=True)
class SweepCell:
    """One grid cell: the config that ran, and how many episodes ran in it."""

    label: str
    config: GameConfig
    repetitions: int = 1

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("a sweep cell must run at least one episode")


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """The resolved grid — which is to say, the channel's input distribution."""

    condition_name: str
    cells: tuple[SweepCell, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        if not self.cells:
            raise ValueError("a sweep needs at least one cell")
        labels = [cell.label for cell in self.cells]
        if len(set(labels)) != len(labels):
            raise ValueError("sweep cell labels must be unique")

    @property
    def design_distribution(self) -> np.ndarray:
        """``p(c)``, proportional to episodes per cell.

        Equal repetitions give a uniform grid; unequal ones weight the channel
        input accordingly. Reading it off the repetitions rather than assuming
        uniformity is what keeps this honest for a resumed or partially failed
        grid, where the cells are genuinely unbalanced.
        """

        weights = np.array([cell.repetitions for cell in self.cells], dtype=float)
        return weights / weights.sum()

    @property
    def episode_count(self) -> int:
        return sum(cell.repetitions for cell in self.cells)


@dataclass(frozen=True, slots=True)
class AnalysisSpec:
    """The analysis settings, which must match what `pipeline.py` actually does.

    ``macrostate_bins`` is not a nicety. Left unbinned, the macrostate has N+1
    levels and the lagged CMI's degrees of freedom are ``|M|(|C|-1)(|M|-1)`` —
    at N=10 and two conditions that is 110, which at 100 episodes is about 0.79
    bits of pure plug-in bias, enough to swamp any real effect. Binning to 3-5
    levels is almost certainly not optional; `expected_plugin_bias_bits` below
    is what lets you see that before drawing a conclusion.
    """

    horizons: tuple[int, ...] = (1,)
    macrostate_bins: int | None = None
    window: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))
        if any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must be positive")
        if self.macrostate_bins is not None and self.macrostate_bins < 2:
            raise ValueError("macrostate_bins must be at least 2")

    def resolved_window(self, rounds: int, horizon: int) -> tuple[int, int]:
        """The rounds contributing (current, future) pairs at this horizon.

        A pair needs its future partner to exist, so the last usable current
        round is ``rounds - horizon`` — the same truncation `pipeline.py` gets
        from shifting and dropping the resulting NaNs.
        """

        if self.window is not None:
            first, last = self.window
        else:
            first, last = 1, rounds
        return max(first, 1), max(min(last, rounds - horizon), max(first, 1))


def binning_matrix(population_size: int, bins: int | None) -> np.ndarray:
    """Map the N+1 macrostate levels onto a common grid shared across conditions.

    This is the single most important guard in this module. If the swept
    condition is the population size and the macrostate is a raw action share,
    the share's support is ``{0, 1/N, ..., 1}`` — **a different grid for each
    N**. Observing a share of 0.15 then identifies N=20 with certainty because
    it is unattainable at N=10, the condition becomes near-perfectly
    recoverable from one observation, and the pipeline reports ``I(C;S) ~
    H(C)``: maximal, and entirely an artifact of alphabet support rather than
    of any dynamical influence.

    Binning the *share* onto equal-width bins on [0,1] gives every condition
    the same alphabet, which is what makes the comparison mean anything.
    """

    levels = population_size + 1
    if bins is None:
        return np.eye(levels)
    shares = np.arange(levels) / population_size
    assignment = np.minimum((shares * bins).astype(int), bins - 1)
    matrix = np.zeros((levels, bins), dtype=float)
    matrix[np.arange(levels), assignment] = 1.0
    return matrix


def _dynamics(game: SyntheticGame, sweep: SweepSpec) -> tuple[ExactDynamics, ...]:
    return tuple(game.exact_dynamics(cell.config) for cell in sweep.cells)


def sweep_ground_truth(
    game: SyntheticGame, sweep: SweepSpec, analysis: AnalysisSpec
) -> GroundTruth:
    """Exact family-2 quantities for a resolved sweep.

    ``sweep`` and ``analysis`` must be the objects that actually ran, never
    hand-written values — the whole reason ground truth is computed rather than
    recorded is that a config change must move the answer with it.
    """

    dynamics = _dynamics(game, sweep)
    design = sweep.design_distribution
    sizes = {item.population_size for item in dynamics}
    if len(sizes) > 1 and analysis.macrostate_bins is None:
        raise ValueError(
            "this sweep varies population_size but leaves the macrostate unbinned. "
            "The action share's support is {0, 1/N, ..., 1}, a different alphabet per "
            "condition, so the condition becomes recoverable from a single observation "
            "and I(C;S) collapses onto H(C) as a pure support artifact. Set "
            "AnalysisSpec.macrostate_bins to put every condition on a common grid."
        )

    quantities: list[GroundTruthQuantity] = []
    labels = [cell.label for cell in sweep.cells]

    # -- terminal ---------------------------------------------------------
    outcomes = np.stack([item.terminal_outcome_law() for item in dynamics])
    terminal = exact.design_mutual_information(design, outcomes)
    quantities.append(
        GroundTruthQuantity(
            name="terminal_mutual_information",
            value=terminal,
            definition=(
                "I(C; dominant action in the final round) under the design p(c). "
                "Ties resolve to the first declared action, matching reader.py."
            ),
        )
    )
    for index, label in enumerate(labels):
        quantities.append(
            GroundTruthQuantity(
                name="terminal_outcome_probability",
                value=float(outcomes[index, 0]),
                subject=(label,),
                units="probability",
                definition="P(terminal dominant action = the first declared action | c).",
            )
        )
    quantities.append(
        GroundTruthQuantity(
            name="terminal_plugin_bias",
            value=exact.plugin_bias_bits(
                (len(sweep.cells) - 1) * (outcomes.shape[1] - 1), sweep.episode_count
            ),
            definition=(
                "dof/(2 N_eff ln2) with N_eff the number of EPISODES, not rounds - "
                "within-episode pairs are correlated and buy no independent samples."
            ),
        )
    )

    # -- lagged conditional MI -------------------------------------------
    rounds = dynamics[0].rounds
    # One binner per condition, not one for the whole sweep. When the sweep
    # varies the population size each cell has a different raw macrostate
    # dimension, and it is exactly the binning that maps them onto the shared
    # alphabet the comparison needs - so the matrix has to be built per cell.
    binners = [
        binning_matrix(item.population_size, analysis.macrostate_bins) for item in dynamics
    ]
    macro_levels = binners[0].shape[1]
    if len({binner.shape[1] for binner in binners}) > 1:
        raise ValueError("every condition must bin onto the same number of macrostate levels")
    for horizon in analysis.horizons:
        window = analysis.resolved_window(rounds, horizon)
        # p(c, m, m') with axes (C, M, M') = (X, Z, Y): the conditional MI
        # I(C; M' | M) is exactly the CMI of that table.
        joint = np.zeros((len(sweep.cells), macro_levels, macro_levels), dtype=float)
        marginal = np.zeros((len(sweep.cells), macro_levels), dtype=float)
        for index, (item, binner) in enumerate(zip(dynamics, binners)):
            pair = item.pooled_macrostate_pair_law(horizon, window)
            joint[index] = binner.T @ pair @ binner
            marginal[index] = item.pooled_macrostate_law(window) @ binner
            joint[index] *= design[index] / max(joint[index].sum(), 1e-300)
        value = exact.conditional_mutual_information(joint)
        quantities.append(
            GroundTruthQuantity(
                name="lagged_conditional_mutual_information",
                value=value,
                subject=(str(horizon),),
                definition=(
                    "I(C; S_{t+h} | S_t), pooled over the analysed window before the "
                    "conditional MI is taken - matching a pooled-count estimator."
                ),
            )
        )
        quantities.append(
            GroundTruthQuantity(
                name="lagged_plugin_bias",
                value=exact.plugin_bias_bits(
                    macro_levels * (len(sweep.cells) - 1) * (macro_levels - 1),
                    sweep.episode_count,
                ),
                subject=(str(horizon),),
                definition="dof = |M|(|C|-1)(|M'|-1), over the number of episodes.",
            )
        )
    quantities.append(
        GroundTruthQuantity(
            name="condition_state_mutual_information",
            value=exact.design_mutual_information(design, marginal),
            definition=(
                "I(C;S) over the pooled macrostate. For a game with i.i.d. rounds the "
                "lagged CMI is exactly this minus I(S_t;S_{t+h}) and is flat in h."
            ),
        )
    )

    # -- bookkeeping the comparison needs to be honest --------------------
    quantities.extend(
        [
            GroundTruthQuantity(
                name="condition_alphabet_size", value=float(len(sweep.cells)),
                units="levels", definition="|C|, the number of grid cells.",
            ),
            GroundTruthQuantity(
                name="macrostate_alphabet_size", value=float(macro_levels),
                units="levels", definition="|M| after binning.",
            ),
            GroundTruthQuantity(
                name="effective_sample_size", value=float(sweep.episode_count),
                units="episodes",
                definition="Episodes, not rounds - the unit the bootstrap resamples over.",
            ),
            GroundTruthQuantity(
                name="macrostate_is_lumpable",
                value=1.0 if all(item.is_lumpable for item in dynamics) else 0.0,
                units="indicator",
                definition=(
                    "1 when every condition's macrostate chain is Markov. When 0, only "
                    "microstate propagation is correct - which is what this module does."
                ),
            ),
        ]
    )

    return GroundTruth(
        game_type=game.spec.game_type,
        parameters={
            "condition_name": sweep.condition_name,
            "cells": [
                {"label": cell.label, "repetitions": cell.repetitions}
                for cell in sweep.cells
            ],
            "design_distribution": design.tolist(),
            "horizons": list(analysis.horizons),
            "macrostate_bins": analysis.macrostate_bins,
            "window": list(analysis.resolved_window(rounds, analysis.horizons[0])),
        },
        quantities=tuple(quantities),
    )


def sweep_from_values(
    base: GameConfig, condition_path: str, values: Sequence[Any], *, repetitions: int = 1
) -> SweepSpec:
    """Build a sweep by varying one ``game.options`` key over a list of values.

    Mirrors how the grid runner sweeps a dotted config path, so the cells here
    are the same objects that would actually run rather than a parallel
    description of them.
    """

    from dataclasses import replace

    cells: list[SweepCell] = []
    for value in values:
        if condition_path == "population_size":
            config = replace(base, population_size=int(value))
        elif condition_path == "horizon":
            config = replace(base, horizon=int(value))
        else:
            options = dict(base.options)
            options[condition_path] = value
            # A per-agent override would silently win over the scalar being
            # swept, so it is dropped when the scalar is the swept axis.
            if condition_path == "epsilon":
                options.pop("epsilons", None)
            config = replace(base, options=options)
        cells.append(
            SweepCell(f"{condition_path}={value}", config, repetitions=repetitions)
        )
    return SweepSpec(condition_name=condition_path, cells=tuple(cells))


__all__ = [
    "AnalysisSpec",
    "SweepCell",
    "SweepSpec",
    "binning_matrix",
    "sweep_from_values",
    "sweep_ground_truth",
]
