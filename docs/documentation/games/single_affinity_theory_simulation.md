# Single-affinity theory Monte Carlo simulation

This simulator is a Monte Carlo realization of the coarse single-affinity finite-compliance reference theory. Monte Carlo means repeated random trajectories used to estimate the probabilities defined exactly by the theory. It is not a q-voter agent simulation and is not a provider-free reimplementation of the full relational reasoning game.

## What it simulates

The complete state is `n`, the number of target supporters in a population of size `N`. Each feedback cycle is:

```text
n -> Y -> U -> n'
```

1. `Y` is sampled without replacement from the exact hypergeometric sensor law `S(Y|n)`.
2. `U` is the controller action. It is sampled from the policy using the realized `Y`, not directly from `n`.
3. If `U=0`, the count stays fixed. This is `Q0=I`, where `I` is the identity transition matrix.
4. If `U=1`, the simulator performs exactly `b` sequential draws from `K`. This is a trajectory-level realization of `Q1=K^b`.

The parameter `h` is directional affinity: it controls which direction a compliant revision favors. The parameter `gamma` is kinetic compliance: it controls the probability that one controlled opportunity changes an opinion. For finite `h` and positive `gamma`, movement in both directions remains possible.

The simulator has no language-model provider, prompts, named agents, ballots, facts, semantic relations, or ordinary q-voter updates.

## Initialization

Three initial count distributions are supported:

- `fixed_count`: every episode starts at one `n0`.
- `binomial`: `n0` follows the exact distribution returned by `binomial_ensemble(N, x0)`.
- `distribution`: `n0` follows an explicit probability vector of length `N+1`.

Each episode samples independently from the resolved initial distribution.

## Configuration

```yaml
simulation:
  type: single_affinity_theory
  seed: 20260828
  episodes: 100000
  rounds: 10
  validation_samples_per_state: 10000

theory:
  N: 24
  q_c: 12
  b: 18
  beta: 4.0
  theta: 0.5
  h: 2.0
  gamma: 0.35

initialization:
  type: binomial
  x0: 0.33

artifacts:
  record_cycles: false
  record_microsteps: false
```

`q` is intentionally not accepted. The revised controlled theory fixes it to `q=1`.

## Running

```text
mas-cc theory simulate --config <config.yaml> --output-dir <directory>
```

The command is provider-free. The same resolved configuration and seed reproduce the same trajectories and validation samples exactly.

An existing output directory is safe to reuse only when its `resolved_config.json` matches. A different resolved configuration is rejected so unrelated results cannot be mixed.

## Exact comparison

The simulator compares random estimates with the deterministic calculations in `theory_revised.py`:

- sensor probabilities `S`;
- policy probabilities conditional on `Y` and on `n`;
- one-opportunity kernel `K`;
- `Q0=I`;
- sequential `b`-step advocacy against `Q1=K^b`;
- the complete closed-loop kernel;
- susceptibility `chi`;
- state-local action information `T_pi` in bits;
- transient occupancy at every cycle;
- terminal-current mean and variance;
- sensing information, entropy production, and thermodynamic efficiency in nats.

The information-response efficiency and thermodynamic efficiency are ratios of accumulated numerators and denominators. Per-state or per-cycle ratios are not averaged.

## Outputs

- `resolved_config.json`: fully resolved parameters and initialization.
- `metadata.json`: theory version, seed, and model identity.
- `occupancy_by_round.csv`: Monte Carlo and exact nonstationary occupancies.
- `state_local_validation.csv`: exact and Monte Carlo `a`, `chi`, `T_pi`, and `eta_IR` by state.
- `horizon_summary.csv`: exact and Monte Carlo finite-horizon current, fluctuations, information, entropy production, and efficiencies.
- `thermodynamics_by_round.csv`: one row per finite cycle.
- `validation_summary.json`: maximum absolute differences from exact theory.
- `cycle_trajectories.csv`: written only when `record_cycles` is true.
- `controlled_microsteps.csv`: written only when `record_microsteps` is true.

Cycle storage is off by default. Aggregate counts are streamed while episodes run, so large runs do not need to retain all trajectories in memory.
