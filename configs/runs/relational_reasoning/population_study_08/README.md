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

## Preflight snapshot (2026-08-27)

Both configs passed live-pricing experiment preflight independently: 96 cells,
960 episodes, 267,840 expected requests and 506,880 conservative requests per
config. Combined Study 08 demand is 192 cells, 1,920 episodes, 535,680 expected
requests and 1,013,760 conservative requests. Reported monetary cost is `0.00`
in the provider's `proxy_accounting_unit`; this is the provider's current zero
token-rate accounting result, not a currency-valued prediction.

The per-config serial-equivalent rough estimate is 100,440 seconds. Automatic
cell-array execution uses 192 shards, throttle 18, eight episode slots and eight
request permits per shard: 144 concurrent episode slots/request permits and an
864 RPM planning estimate beneath the declared 900 RPM target. The conservative
per-cell duration fits the four-hour SLURM limit. Actual wall time depends on
provider latency and adaptive load control.

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
