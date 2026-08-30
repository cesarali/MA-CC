# Relational Population Study 08

Study 08 crosses receiver epistemic disposition with controller evidence
strategy under matched wrong/adversarial and truth-aligned control. Every cell
uses `recommendation_plus_fact`; recommendation-only is not a Study 08 cell.

## Design

- Receiver disposition: `naive`, `vigilant`
- Controller evidence strategy: `neutral`, `strategic`
- Derived labels: `naive_neutral`, `naive_strategic`, `vigilant_neutral`,
  `vigilant_strategic`
- Intervention budget: `4, 8, 12, 16, 20, 24`
- Controller semantics: wrong fixed option index `2`, or task truth via
  `target: correct`
- Tasks: `task_0001` through `task_0004`
- Repetitions: 10
- Fixed: `N=24`, `q=1`, `q_c=12`, `theta=0.5`, `beta=4`, 10 rounds

Arithmetic: `4 × 6 × 2 × 4 = 192` scientific cells and
`192 × 10 = 1,920` episodes.

Both semantic blocks use the same root seed and identical grid ordering.
`common_random_numbers_across_grid: true` gives every condition the same
repetition-index stream. Scientific cell identity remains independent of the
SLURM shard topology.

The study uses the generic automatic cell-array launcher. Results and SLURM
logs belong under `/work/ojedamarin/Projects/LanguageGames/MA-CC/results`, not
the home repository.

Neutral evidence is the first fact in frozen `fact_order`, independent of the
controller target. Strategic evidence maximizes positive cosine alignment
between the fact's frozen compass relation and the resolved controller target;
ties are broken by `fact_order`. A task/target with no positively aligned real
fact is rejected as strategically inadmissible. No selector creates or changes
fact text.

## Frozen-task evidence audit

The false target is the established option index `2`; truth uses `correct`.
All selected facts below are exact members of the frozen task.

| task | truth target | false target | neutral | strategic truth | strategic false | admissible |
|---|---|---|---|---|---|---|
| `task_0001` | `WEST` | `EAST` | `f1` | `f1` (`NORTHWEST`) | `f6` (`EAST`) | yes |
| `task_0002` | `SOUTHWEST` | `SOUTH` | `f1` | `f2` (`SOUTHWEST`) | `f1` (`SOUTH`) | yes |
| `task_0003` | `NORTH` | `EAST` | `f1` | `f4` (`NORTH`) | `f1` (`EAST`) | yes |
| `task_0004` | `SOUTHEAST` | `NORTHWEST` | `f1` | `f1` (`SOUTHEAST`) | `f6` (`NORTHWEST`) | yes |

## Preflight snapshot (updated 2026-08-29)

Both NeuralWatt configs passed offline-catalog experiment preflight independently: 96 cells,
960 episodes, 267,840 expected requests and 1,520,640 conservative requests per
config. Combined Study 08 demand is 192 cells, 1,920 episodes, 535,680 expected
requests and 3,041,280 conservative requests. The dated catalog estimate is
approximately 647.26 USD expected and 3,674.80 USD conservative.

Each decision permits five validation retries (six total attempts). The first
production launch demonstrated that the previous two-attempt setting could
terminate a whole episode after two malformed provider responses. At the
declared 5% invalid-response rate, six attempts reduce the expected number of
terminal malformed decisions across the complete study below 0.01. This is an
execution-integrity setting: it changes no prompt, seed, scientific coordinate,
or estimator, and the expected request/cost estimate remains unchanged.

Retryable HTTP transport faults have a separate nine-attempt bound with the
provider adapter's full-jitter exponential backoff. The shared provider
coordinator opens its global breaker at a 5% rolling retryable-failure ratio and
locally pauses a node after two failures. These execution-only settings were
added after a live synchronized HTTP 503 burst exhausted the former three
attempts while the older 25% breaker threshold was diluted by a preceding
healthy minute. The coordinator starts at two requests and waits five healthy
minutes before each additive increase toward five; this prevents the
30-second recovery oscillation observed during the same provider incident.

Automatic cell-array execution uses 192 shards, throttle five, one episode slot
and one local request permit per shard. The study-wide runtime coordinator caps
all Potsdam or NERSC workers at five in-flight NeuralWatt requests and applies a
500-RPM rolling gate. At the benchmark-informed 1.5-second planning latency this
is 200 RPM; the measured exact-prompt run reached approximately 211 RPM. The
NeuralWatt provider adapter supplies `response_format: json_object` by default;
the game and study configs contain no transport override, and the much slower
forced-tool protocol is not part of the corrected production run. The expected
provider time is roughly 42 hours and NERSC therefore resumes the same prepared
root across successive four-hour interactive allocations.

The checked-in result guard and scheduler fields remain Potsdam-safe under
`/work`. NERSC preparation explicitly stamps and rebinds generated artifacts
under `/pscratch`; it does not edit these source configs.

The deterministic preflight estimator reports 353/416 representative input
tokens for naive initialization/update prompts and 402/465 for vigilant ones.
The vigilant values equal the historical Study 06 strategic-warning estimates;
the naive prompt removes 49 tokens. The estimator's representative social
source already contains a fact in both modes, so it does not isolate a separate
recommendation-only versus plus-fact token delta.

Minimal cell coordinates are:

```yaml
game:
  options:
    receiver_epistemic_disposition: naive  # or vigilant
control:
  options:
    target: correct                        # or false target/index
    message_mode: recommendation_plus_fact
    controller_evidence_strategy: neutral # or strategic
```

The analysis recipe preserves the complete Study 06/07 estimator set and adds
factor/semantic comparison plots with shared color scales. Canonical
rounds retain pre-round target share, `phi`, `kappa`, time, truth/target state,
and support fields for state-local comparisons without another provider run.

Aggregation invokes the existing round-feedback estimator on supported
pre-round slices at resolutions `x`, `x × phi`, and `x × kappa`; it does not
introduce another MI/CMI implementation. It automatically renders matched
truth/false 2×2 families for `T_pi`, signed response (`chi`), and `eta_IR`, plus
`b=24` profiles and `phi(t)`, `kappa(t)`, truth-share, and target-share
evolution. Unsupported slices remain blank. The derived table also contains
descriptive vigilance, evidence-strategy, and truth-minus-false differences;
these are explicitly contrasts rather than new efficiency definitions.

## Single-affinity coordinates (corrected semantics)

The efficiency family is stated in the coordinates of the revised
single-affinity theory, so an empirical number and the exact theory number of
the same name are directly comparable. Three names changed meaning or arrived
new; everything they replaced is still in the output under its own name.

- **`round_target_susceptibility` is the canonical `chi`.** It is the
  state-matched difference of mean target-*fraction* changes,
  `E[dx | ADVOCATE, n] - E[dx | NO_OP, n]`, conditioned on
  `target_count_before`. `round_target_signed_actuation` measures the same
  difference in aligned-magnetization units and is therefore larger by
  `K/(K-1)`; it is retained as a diagnostic and is no longer what `eta_ir`
  consumes. On these three-option tasks the earlier `eta_ir` numerator was
  inflated by `(3/2)^2 = 2.25` for exactly that reason.
- **`eta_ir` is occupancy weighted.** It is a ratio of sums,
  `sum_n p(n) B_IR(n) / I(U; n' | n)`, not a mean of state-local ratios. The
  state-resolved surface the 2x2 heatmaps read is published separately as
  `eta_ir_state_local`.
- **`controlled_current` is not `cell_current`.** The thermodynamic current is
  `N sum_n p_k(n) a(n) chi(n)`, summed over the horizon. `cell_current` remains
  the terminal episode difference, a behavioural outcome that also contains the
  ordinary social dynamics.
- **`target_sensing_information_nats` is not `round_sensing_mi`.** The
  thermodynamic sensing term is the scalar channel `I(n_Z; Y_Z)` in nats, built
  from the empirical occupancy and the exact hypergeometric sensor kernel, one
  round at a time. `round_sensing_mi` remains the full K-option vector channel
  `I(N; Y)` in bits.
- **`eta_th = h*J_c / (h*J_c + I_sens)`** over the horizon, with `h` calibrated
  from controlled microscopic vote transitions using natural logs. Cases where
  the controller pushed against its own affinity are flagged
  (`eta_th_target_directed = false`), never clipped into `[0,1]`.

Both efficiencies carry percentile intervals from a whole-episode bootstrap
that recomputes every ingredient inside each replicate. Every derived row is
stamped `theory_semantics_version = single_affinity_v1`; rows without that stamp
predate the correction and must not be pooled with these.
`analysis/tables/single_affinity_theory_comparison.parquet` puts the empirical
and exact values of `chi, T_pi, eta_IR, J_c, I_sens, eta_th` side by side for
each calibrated cell. The matched q-voter theory remains a separate classical
reference and is never substituted into these formulas.
