# HiddenBench imitation — round feedback

This is a separate scientific game from `hidden_bench_imitation`. It keeps the
same HiddenBench evidence, initialization, reasoning prompts, focal update,
population observables, and local trajectory schema, but uses two clocks:

- `round_index`: one slow sensing/action decision;
- `within_round_index`: exactly `N` focal-update opportunities per round.

`round_soft_target_budgeted` samples `q_c` opinions once, draws one
`NO_OP`/`ADVOCATE_Z` action, and—only for advocacy—samples exactly `b` update
positions uniformly without replacement before the round begins. A controlled
position replaces one of the `q` ordinary social slots; it never overwrites the
focal vote.

The reference classical kernel is a strict-unanimity q-voter update. The focal
copies an option only when all q effective influence slots agree on that
option; otherwise it stays put. In a controlled update the effective inputs
are the controller target plus q-1 ordinary peers, so only a non-target-to-
target switch is possible. For q=1, a controlled non-target focal always
switches to the target. The kernel performs no provider calls and adds no
spontaneous noise, soft response, anticonformity, or hidden control strength.

## Configuration

```yaml
game:
  type: hidden_bench_imitation_round_feedback
  population_size: 32
  horizon: 10                 # generic schema mirror; expressed in rounds
  options:
    rounds: 10
    dynamics_mode: classical # or reasoning
    social_group_size: 2     # q
    assignment_scheme: paraphrased_replication
    population_preparation:
      auto_build_missing: true
      paraphrase_annotations: data/hidden_bench/annotations/paraphrases.json
    classical:
      kernel: controlled_imitation_round_reference
control:
  mechanism: round_soft_target_budgeted
  options:
    target: correct
    sensor_sample_size: 8    # q_c
    threshold: 0.5
    beta: 6.0
    intervention_budget: 8   # b
    template_version: 3
```

The generic `game.horizon` field remains required by the repository schema;
this game uses `game.options.rounds` as the authoritative slow-clock horizon.
When automatic paraphrase preparation is enabled, an existing scaled task is
reused. A missing task is added deterministically from the annotation file;
the runtime never asks a model to create paraphrases.

## Persistence and analysis

Microscopic events remain in `trajectory.jsonl`. One independent
`imitation_round_feedback` record per round is written to
`round_trajectory.jsonl`, including the policy probability and schedule replay
metadata. Analyze a completed run or grid without provider calls:

```bash
python -m mas_cc.cli.main analysis hidden-bench-round-feedback \
  --run-dir results/<run>
```

Full-state direct-counting CMI is sparse for large `N` and `K`; use the emitted
support/overlap diagnostics and the target-count projection. Raw sensing MI is
also not directly comparable across different `q_c` alphabets without an
estimator/alphabet caveat. The reasoning process is reported as a first-order
round statistic, not assumed to be Markov.
