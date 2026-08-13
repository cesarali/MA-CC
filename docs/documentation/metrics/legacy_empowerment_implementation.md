# The legacy `naming_game` empowerment pipeline: how it works

This is a technical description of **how** `src/naming_game/` implements
"committee empowerment" and what experiments it was built to run — not a
usage guide (that's [`docs/handoff/committee_empowerment_guide.md`](handoff/committee_empowerment_guide.md))
and not the general repo map (that's [`architecture_overview.md`](architecture_overview.md),
which deliberately excludes this legacy package). Read this if you want to
understand or reuse the actual estimator/experiment logic, line by line.

## 1. What "empowerment" means in this implementation

This pipeline does not implement the classical single-agent empowerment
definition (channel capacity between an actuator's own action sequence and
its later sensed state). Instead it measures something adjacent and simpler:
**how much does an experimentally assigned committee policy determine the
population's outcome** — mutual information between a *treatment label*
(`committee_policy`, e.g. `always_A` vs. `no_committee`) and either the
episode's terminal convention or a future population macrostate. It's an
intervention→outcome MI, estimated across many independent episodes, not a
single agent's actuator→sensor channel capacity within one trajectory. Keep
that distinction in mind — it's the reason the estimator (Section 5) counts
across *episodes*, not across time steps of one run.

## 2. The game layer: where the "committee" attaches

`naming_convention_game.py` implements the repeated coordination game itself
(adapted from `external/AI-norms`, Ashery et al.): each interaction samples
one ordered pair, both agents simultaneously pick an action from a shared
pool (`ConventionGameConfig.actions`, default `("Q", "M")`; the empowerment
experiment overrides this to `("A", "B")`), matching choices reward both
agents (`success_reward`, default `+100`), mismatches penalize both
(`failure_payoff`, default `-50`). Each `ConventionAgent` only ever sees its
own bounded history (`memory_size` entries) — never another agent's, never
global population state.

The committee mechanism is a `ConventionIntervention` protocol
(`naming_convention_game.py:37`) — just two methods,
`action_for(agent_id, interaction_index)` and `window_active(interaction_index)`.
`NamingConventionGame._request_decision` (`naming_convention_game.py:615`)
checks `intervention.action_for(...)` *before* ever calling the LLM: if it
returns an action, that agent's decision is `forced=True` and no API call is
made at all for that agent this interaction. This is the core design
guarantee from `docs/deliverables/committee_empowerment_probe.md`: committee
actions override only the *selected* decision, they never create a separate
memory type or alter how an ordinary agent's prompt is built — a committee
member reverts to ordinary LLM behavior the instant it's not forced.

`CommitteeSchedule` (`empowerment_experiment.py:186`), which implements
`ConventionIntervention`, is what the experiment layer actually constructs
per episode — it's a frozen dataclass holding which agent IDs are on the
committee, what policy they enact, what single action they force, and
(for pulse regimes) an `active_through_interaction` cutoff after which
`action_for` starts returning `None` again — i.e. the committee's pulse
naturally expires without any special-casing in the game engine.

## 3. Experiment design: `EmpowermentExperimentConfig` and the episode grid

`EmpowermentExperimentConfig` (`empowerment_experiment.py:81`) is the single
YAML-loaded configuration object (`load_experiment_config`) for the whole
experiment — population size (default 24), the two-name pool (hard-required
to be exactly `["A", "B"]` in this probe), memory length, episode horizon
(`max_population_rounds`), the committee sizes to sweep, and three
**regimes**:

| Regime | Committee policies swept | What it's testing |
| --- | --- | --- |
| `neutral` | `always_A`, `always_B`, `no_committee` | Baseline: does a permanently-forcing committee of size *k* pull the population toward its favored action, with no prior convention? |
| `consensus_attack` | `support_incumbent`, `promote_alternative`, `no_committee` | The population is pre-seeded (`game.seed_consensus_history`) at a stable `A` or `B` consensus; does a committee promoting the *other* action manage a takeover? |
| `pulse` | `alternative_pulse`, `no_pulse` | Same seeded-consensus setup, but the committee only forces its alternative action for a fixed number of population rounds (`pulse_rounds`), then stops — does the population recover the original convention, and how fast? |

`build_episode_specs` (`empowerment_experiment.py:253`) is the pure function
that expands `(regime × committee_size × stratum × policy × replicate)` into
one `EpisodeSpec` per cell — it's the actual combinatorial design matrix.
Each spec's `episode_id` is a truncated SHA-256 hash of every
identity-relevant field (regime, committee size, policy, seed, prompt
version, decision format, etc.) via `json.dumps(..., sort_keys=True)`
(`empowerment_experiment.py:289`) — this is what makes checkpointing
content-addressed rather than index-addressed (Section 4). Replication has
two modes (`ReplicationConfig.unit`): `per_policy` (N independent replicates
of *every* cell) or `per_stratum` (N total episodes round-robined across
policies within a stratum) — `_policy_replicates` (`empowerment_experiment.py:244`)
implements the split.

`_schedule_for` (`empowerment_experiment.py:314`) turns one `EpisodeSpec`
into the actual `CommitteeSchedule`: which agent IDs are drawn (seeded off
`spec.seed ^ 0xC01117EE` so committee membership is reproducible but
decorrelated from the game's own RNG stream), and which single action they
force, per the policy table above.

## 4. Running one episode, and what gets derived from it

`run_episode` (`empowerment_experiment.py:552`) builds the schedule, builds
the game, seeds a synthetic consensus history when the stratum calls for one
(`game.seed_consensus_history(spec.incumbent)`), runs
`config.max_interactions` pair interactions, and — through `derive_episode`
(`empowerment_experiment.py:426`) — computes, per interaction row, a
**rolling window** over the last `rolling_window` outputs (default
`3 * population_size`) and from it:

- `rolling_share_A` — the fraction of that window playing `A`;
- `macrostate_binary` — 1/0/carry-forward majority label (ties keep the
  previous value rather than flipping arbitrarily);
- `macrostate_three` — a coarser `A_dominant` / `mixed` / `B_dominant` label
  at 60/40 thresholds;
- `resolved_state` — `A`/`B`/`unresolved`, only once the window is full and
  the share crosses `resolution_threshold` (default 0.95) — this is the
  per-interaction quantity `estimate_lagged` (Section 5) actually conditions
  on.

From the full interaction sequence it then derives one **episode summary**
row: first-consensus timing, `takeover`/`terminal_takeover` (did the
committee's alternative ever appear / end up as the terminal state),
`incumbent_survives`, `recovery_time_*` and `recovery_censored` (for pulse
episodes — time from pulse removal back to the original incumbent, or
censored if it never recovers within the horizon), `peak_displacement` (the
highest the alternative's share ever got), and `post_consensus_persistence`
(fraction of population-round endpoints after first consensus that stayed on
that same label). These derived fields are exactly the columns the
`empowerment.py` estimator (Section 5) and `metrics.py` (Section 6) group and
count over — nothing in those modules re-derives them from raw actions.

## 5. Running the full grid: checkpointing and compaction

`run_experiment` (`empowerment_experiment.py:697`) is the orchestrator over
all specs from `build_episode_specs`. Concurrency is two-tier:
`episode_concurrency` episodes in flight at once (an `asyncio.Semaphore`),
each internally issuing up to `request_concurrency` LLM calls. Every episode
is checkpointed independently as
`.episode_shards/<experiment_fingerprint>/<episode_id>.{interactions,episode}.parquet`,
written via a temp-file-then-`replace` pattern so a killed process never
leaves a half-written shard; `resume=True` (the default) skips any
`episode_id` whose shard pair already exists on disk — this is why
`episode_id` is a content hash of the *data-generating* parameters
(Section 3) rather than a sequential index: the same config re-run after a
crash resumes exactly the episodes it's missing, and a config with an
expanded grid still recognizes the shards it already has. `_experiment_fingerprint`
(`empowerment_experiment.py:387`) hashes the config fields that actually
change the data (not concurrency/logging knobs), so unrelated re-runs don't
collide, but tuning purely operational settings doesn't invalidate existing
checkpoints either.

Once every spec's shard exists, all shards are concatenated and written as
two top-level artifacts — `interactions.parquet` (one row per pair
interaction) and `episodes.parquet` (one row per episode summary) — with a
row-count and `episode_id`-set consistency check against what was actually
requested before the temp files are atomically promoted
(`empowerment_experiment.py:847`). `clear_completed_shards` then optionally
removes the now-redundant per-episode shards. This Parquet pair is the *only*
input the analysis pipeline (Section 6) ever reads — it never touches the
game or provider layers again.

## 6. The estimator: `analysis/estimators.py`

Pure, dependency-free (beyond NumPy) direct-counting estimators over
complete contingency tables:

- `mutual_information_from_counts` — from a 2-D `(X, Y)` count table, returns
  an `Estimate` with three parallel readings from the *same* table: `jeffreys`
  (`+0.5` pseudocount smoothing over every possible cell —
  `complete_cells`), `unsmoothed`, and `miller_madow` (the standard bias
  correction `(k-1) / (2N ln 2)` added to each marginal/joint entropy term).
  `mutual_information` is the convenience wrapper that builds the count table
  from raw label sequences first.
- `conditional_mutual_information_from_counts` — same idea over a 3-D
  `(X, Z, Y)` table, computing `I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)`
  (`estimators.py:79`).

Reporting all three variants side by side (rather than picking one) is
deliberate — it's how the pipeline surfaces its own small-sample sensitivity
rather than hiding it behind a single number.

## 7. The estimation pipeline: `analysis/empowerment.py`

`analyze_histories` (`empowerment.py:322`) is the single entry point; it
loads `interactions.parquet`/`episodes.parquet`, then:

- **`estimate_terminal`** (`empowerment.py:118`) — for every stratum (regime,
  population size, committee size, initial condition, provider, model,
  prompt version, pulse duration, attack direction — `STRATA_COLUMNS`,
  everything in `metrics.GROUP_COLUMNS` except `committee_policy`), builds the
  `(committee_policy, final_convention)` contingency table **summed at the
  episode level** (one count per completed episode, not per interaction —
  episodes are the independent, exchangeable unit) and estimates
  `I(policy; terminal outcome)`. `resolved_only=True` re-runs it excluding
  episodes that never resolved, as a sensitivity check.
- **`estimate_lagged`** (`empowerment.py:185`) — for each configured horizon
  (in population rounds), builds `(committee_policy, macrostate_now, macrostate_future)`
  triples within each episode (`_lag_pairs`, shifting by `horizon * N`
  interactions) and estimates the conditional MI
  `I(policy; future macrostate | current macrostate)` — does knowing the
  policy still predict where the population ends up, *beyond* what the
  current macrostate already tells you.
- **Bootstrap confidence intervals** (`_bootstrap_interval`,
  `empowerment.py:97`) — resampled **with replacement over `episode_id`**,
  never over interaction rows, since interactions within an episode aren't
  independent.
- **`_estimation_status`** (`empowerment.py:67`) gates every estimate as
  `estimable` / `exploratory` (5–9 completed episodes for the thinnest
  policy in that stratum) / `non_estimable` (fewer than 5, or only one
  policy present) — a stratum that can't yet support an estimate reports
  `NaN` with a stated reason rather than a misleadingly precise number.
- `metrics.summarize_episode_metrics` + `add_efficiency` (Section 8) are
  folded in, and `efficiency = I(policy; outcome) / expected number of
  committee-forced actions` — bits of empowerment per intervention spent.

## 8. Null models and invariance checks: `analysis/surrogates.py`

Three surrogate transformations, all preserving each episode's internal
structure so the null distribution is comparable to the real one:

- **`shuffle_episode_labels`** — permutes `committee_policy` across episodes
  *within* strata defined by everything else (regime, committee size, pulse
  duration, provider, model, prompt version, initial condition) — this is
  the policy-label permutation null: what MI would you see if the policy
  had no real effect but the episode outcomes were otherwise unchanged.
- **`circular_shift_trajectories`** — rolls each episode's macrostate
  sequence by a random offset, which destroys the true policy→timing
  alignment while preserving the sequence's own autocorrelation structure —
  a temporal null specifically for `estimate_lagged`.
- **`swap_labels_half`** — relabels `A`↔`B` (and every derived field that
  encodes a direction: `resolved_state`, `attack_direction`,
  `macrostate_binary`, `rolling_share_A`, etc.) for a balanced random half of
  episodes per stratum — a real MI estimate must come out (nearly) identical
  before and after, since `A` and `B` are arbitrary labels; `analyze_histories`
  checks this explicitly (`invariant_within_tolerance`, 1e-9) and writes it to
  `label_swap_invariance.parquet`.

`estimate_nulls` (`empowerment.py:259`) re-runs `estimate_terminal` and
`estimate_lagged` (with resampling switched off — nulls need many cheap
draws, not per-draw confidence intervals) once per permutation over both the
label-shuffle and the circular-shift surrogate, `null_permutations` times,
and records every draw. `analyze_histories` also derives a **no-committee
baseline** — `committee_size == 0` episodes should show MI at or below the
95th percentile of their own shuffle-null distribution; if not, something
about the estimator or the episode design is leaking signal that has nothing
to do with the committee.

## 9. Complementary episode metrics: `analysis/metrics.py`

`summarize_episode_metrics` computes, per stratum, plain frequentist
readouts that stand next to the MI estimates rather than replacing them:
takeover/terminal-outcome/consensus/permanent-flip probabilities with Wilson
score intervals (`_wilson_interval`), and bootstrap-median recovery time
(`_bootstrap_median`) for pulse episodes only. `add_efficiency` is what joins
this table back onto the MI estimates to produce the bits-per-intervention
figure mentioned in Section 7.

## 10. Output artifacts and how to run it

`naming-game experiment --config <path> [--mock] [--no-resume] [--analyze]`
and `naming-game analyze-empowerment --history-dir <dir> [--horizons ...] [--bootstrap-resamples N] [--null-permutations N]`
(`cli.py:79`, `cli.py:92`) are the two commands; `run_experiment`/`analyze_histories`
above are what they call. A completed experiment directory holds
`interactions.parquet`, `episodes.parquet`, `experiment_config.json`, and
(until cleared) `.episode_shards/`. A completed analysis directory holds
`empowerment_estimates.parquet` (terminal + lagged, all three smoothing
variants, CIs, status), `episode_metrics.parquet`, `null_results.parquet`,
`label_swap_invariance.parquet`, `no_committee_baseline.parquet`,
`analysis_config.json`, a rendered `summary.md`, and `plots/`.

**The designed experiment grid**, per `docs/deliverables/committee_empowerment_probe.md`
(configs are no longer present in the working tree but are recoverable from
git history — `git show HEAD:configs/empowerment.yaml` etc.): a full run
(`configs/empowerment.yaml`) was specified at 17,500 episodes, ~12.6 million
pair interactions, up to 25.2 million model calls — explicitly *not* meant to
be launched without first costing it via a pilot
(`configs/empowerment_pilot.yaml`, `configs/empowerment_pilot_test.yaml`).
That full grid was never actually launched against a real provider (per the
same deliverable doc's "known boundaries" section) — only mock and small
live smoke tests were run end-to-end.

## 11. What this means for reuse

If a discrete-action MI metric is what you want for the *current* `mas_cc`
game (see `architecture_overview.md` Section 3.3), Sections 6–8 above are a
complete, tested recipe to port: contingency-table MI/CMI with the three
smoothing variants, episode-level bootstrap, and the three surrogate nulls.
The parts that are specific to *this* experiment's design rather than to the
estimator itself are Sections 2–4 (the committee-intervention mechanism and
the regime/policy grid) — those encode a particular causal question
("does a committee's forced policy move the outcome") that may or may not be
the question the new work is asking.
