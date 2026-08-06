# Master-only logging & the aggregate-metrics tier

*05.08.2026 — implementation spec*

## Goal

Grid sweeps (≤ 2 swept parameters, ~100 episodes/cell, ~1 day wall clock) need:

1. **Live monitoring** — is the job alive, how far along, what's the ETA. Checkable from a phone.
2. **Correct cross-episode aggregates** — computed once per cell, at cell completion.

Both are produced by the **master process only**. Workers never touch Comet.

---

## 1. Writer model

| Process | Writes |
|---|---|
| Worker | Episode parquet to the cluster filesystem. Nothing else. No Comet. |
| Master | All Comet logging: heartbeat, progress, grid image, cell aggregates, sweep metrics. |

Consequences (all intended):

- No race conditions on Comet step counters — one writer per experiment key.
- Comet is a **view**, not the store. If Comet fails, the run is unaffected; re-log later from parquet.
- Aggregates are **recomputable** — percentiles, relabeling and smoothing can be changed after the fact without re-running episodes.
- A job killed at 80% still yields complete, correct aggregates for every finished cell.

Per-episode curves are not lost: the master can log any finished episode's full round-by-round series from its parquet. Only *mid-episode* liveness is unavailable, which is acceptable at these episode durations.

---

## 2. Comet experiment layout

Two kinds of experiment. Each has exactly one meaning for `step` — do not mix.

### Sweep experiment (one per sweep) — `step = episodes_done`

The "is it running" dashboard.

- Params: full grid spec, repetitions/cell, seed, game name, analysis spec.
- Heartbeat metrics, logged on a **timer** (default 60 s) regardless of whether anything completed:
  `episodes_done`, `episodes_failed`, `cells_complete`, `elapsed_minutes`,
  `episodes_per_minute`, `eta_minutes`.
  *A flatlined heartbeat means dead, not slow — this is the whole point of the timer.*
- `grid_progress` image, logged via `log_image(fig, name="grid_progress", step=episodes_done)`.
  Comet's Image Panel exposes a stepper, so this scrubs as an animation of the grid filling in.
  Throttle to every N completions (default 25) to bound the asset list.
- Per-cell scalars appended as cells complete: `converged_fraction`, `median_consensus_round`,
  and the MI block (`estimate`, `null_p95`, `ground_truth`, `gap`).

### Cell experiment (one per grid cell, created at cell completion) — `step = round`

- Tagged with `sweep_id`; params = that cell's coordinates.
- The aggregate curves (§4). Because step is unambiguously the round index, Comet's line
  charts overlay cells against each other natively — which is the comparison we want.

Project view is then one row per cell plus one sweep row. Nothing else.

### Grid image

2-D `imshow` over the two swept parameters. Colour = `episodes_done / repetitions`.
Cells containing failures get a distinct colour (`NaN` + `cmap.set_bad("red")`), so **all-green
means all-healthy**, not merely all-finished. Annotate each cell with `done/total`.

---

## 3. The metrics tier that's missing

Existing tiers in `mas_cc/metrics/`:

| Tier | Scope | Exists |
|---|---|---|
| `StreamingMetric` | one round | yes |
| `FinalMetric` | one episode | yes |
| `AggregateMetric` | one **cell** (many episodes) | **new** |
| `SweepMetric` | the **grid** (many cells) | **new** |

```python
class AggregateMetric(ABC):
    """Cell-level. Consumes every completed episode in one grid cell."""
    @abstractmethod
    def compute(self, episodes: Sequence[EpisodeFrame]) -> AggregateResult: ...
    # AggregateResult: curves {name -> {round: (p10, p50, p90)}} + scalars {name -> float}

class SweepMetric(ABC):
    """Grid-level. Consumes the AggregateResult of every completed cell."""
    @abstractmethod
    def compute(self, cells: Mapping[CellId, AggregateResult]) -> SweepResult: ...
```

`AggregateMetric` runs once, on the cell-completion event. `SweepMetric` runs on every
cell completion (cheap, and lets partial MI estimates appear as the grid fills).

---

## 4. Aggregation rules

These are correctness requirements, not preferences. Do not "simplify" them.

### 4.1 Forward-fill absorbed episodes

Episodes end at different rounds (consensus is absorbing). Averaging at round $t$ over only
the episodes *still running* conditions on "hasn't converged yet" — a biased subset of slow,
near-50/50 runs — which produces a spurious dip in the middle of the curve.

Fill each finished episode forward with its terminal value to the length of the longest
episode in the cell. Legitimate because the value genuinely persists under an absorbing
end condition.

Always log `active_fraction(t) = n_active(t) / n_total` beside the curves. Without it, a
tight band late in the run may just mean "everything is padded."

Config: `forward_fill: absorbing | truncate | none`. Default `absorbing`.

### 4.2 Aggregate the dominant share, or relabel by winner

Averaging `population_action_share_per_option` across episodes washes out: roughly half the
episodes converge to Q and half to M, so the mean for option Q converges to 0.5 with growing
variance and looks like nothing happened.

Either aggregate `dominant_action_share` (converges to 1.0), or — preferred — **relabel each
episode so its eventual winner is always option 1**, then aggregate. Relabeling keeps the
loser's trajectory, so the asymmetry of the approach stays visible.

Sanity check: on synthetic Game 1 there is no winner, so relabelled curves must stay
symmetric. If they don't, the relabel is wrong.

Config: `relabel_by_winner: true`.

### 4.3 Percentile bands, not ±std

The share is bounded in [0,1] and is bimodal for much of a run. Mean ± std produces bands
outside [0,1] and misrepresents two humps as one wide one. Log `p10`/`p50`/`p90`.

State which question a band answers: dispersion across runs (percentiles/std) vs. uncertainty
in the mean (SEM). Default here is dispersion.

Config: `percentiles: [10, 50, 90]`.

### 4.4 Roll within, then aggregate

Rolling windows are applied **within** each episode before aggregating across episodes.
Aggregating first and rolling after gives a different and incorrect answer.

---

## 5. Config

```yaml
aggregation:
  forward_fill: absorbing        # absorbing | truncate | none
  relabel_by_winner: true
  percentiles: [10, 50, 90]
  rolling_window: 20             # applied within episode, before aggregation

  cell_metrics:                  # AggregateMetric, once per cell at completion
    - dominant_action_share      #   curve
    - action_share_relabelled    #   curve, winner-aligned
    - active_fraction            #   curve, always emitted
    - consensus_round            #   distribution -> median + IQR
    - converged_fraction         #   scalar

  sweep_metrics:                 # SweepMetric, on each cell completion
    - terminal_mi
    - lagged_cmi
    - mi_null_band
    - mi_ground_truth_gap        # requires sweep_ground_truth(); see empowerment doc

observability:
  comet:
    writer: master_only          # workers never log
    heartbeat_seconds: 60
    grid_image_every_n_episodes: 25
    sweep_experiment: true
    cell_experiments: true
```

---

## 6. Master control flow

```
on start:
    create sweep experiment, log params, start heartbeat timer

on heartbeat tick (every heartbeat_seconds):
    log progress scalars + ETA          # even if nothing changed

on episode completion:
    increment counters
    if episodes_done % grid_image_every_n_episodes == 0:
        render + log grid image at step=episodes_done

on cell completion:
    read that cell's episode parquet
    apply: relabel -> forward-fill -> rolling -> percentiles
    create cell experiment, log curves at step=round
    run SweepMetric over all completed cells
    log per-cell scalars + MI block to sweep experiment
    re-render grid image

on finish:
    final grid image, final SweepMetric pass, log results parquet as asset
```

---

## 7. Acceptance checks

1. Kill the master mid-run — finished cells still have complete, correct cell experiments.
2. Re-run aggregation from parquet only — reproduces identical curves without touching episodes.
3. Game 1 with `relabel_by_winner` — relabelled curves are symmetric (§4.2).
4. A cell where episodes end at wildly different rounds — the p50 curve is monotone, with
   `active_fraction` decaying alongside it (§4.1).
5. Heartbeat stops within ~2 intervals of the master being killed.
