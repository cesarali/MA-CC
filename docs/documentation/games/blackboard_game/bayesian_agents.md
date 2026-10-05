# Exact Bayesian blackboard agents

Set `game.options.agent_decision_mode: bayesian` to replace population LLM
decisions with exact posterior decisions. The default is `llm`. This is an
optional policy within the existing `reasoning` dynamics, not the unimplemented
classical q-voter dynamics or a new controller policy.

Use a symbolic MuSR Team Allocation task containing
`<task_dataset_dir>/<task_id>/facts/all_true_facts.json`. Native tasks without
canonical predicates and spatial relational tasks are unsupported; the runtime
fails explicitly rather than falling back to an LLM. The existing task loader's
population and artifact checks still apply.

The relevant settings in an existing run config are:

```yaml
game:
  type: relational_imitation_round_feedback
  # Keep the population, task paths, rounds, and other settings of your run.
  options:
    dynamics_mode: reasoning
    agent_decision_mode: bayesian
    task_family: musr_team_allocation
    social_mode: board
    prompt_version: 5
    epistemic_persistence: 0.75
    initialization:
      mode: local_vote
    board:
      allow_no_post: true
      communication_profile: report_only
      report_citation_scope: active_or_observed
```

Keep the run's `prompt.prompt_version` consistent with the game version.
`full_communication` is also supported, but this first Bayesian policy only
posts `REPORT` or `NONE`; it does not issue requests or respond to persuasion.
For a fully provider-free run, use `control.mechanism: none` and a mock provider.
An enabled LLM controller continues to use the configured provider normally.

## Decisions and information limits

- **Vote:** condition the generator's exact independent uniform prior on the
  agent's active facts and grounded reports in its current sampled board view.
  The support is the 14,388 latent worlds with a unique winning allocation,
  out of the 19,683 possible vectors. Choose the maximum posterior allocation.
  On a tie, retain the previous vote if it is tied; otherwise choose with a
  seeded random stream. An empty inventory gives the symmetric prior.
- **Evidence:** use canonical predicates identified by fact IDs, not sentence
  embeddings. Votes, requests, directives, uncited prose, and unobserved facts
  do not enter the posterior. The interpretation dictionary is not conditioned
  on in full. The policy does not read the actual hidden scores or gold answer.
- **Post:** choose one legally citable fact. Prefer facts absent from the
  current sample, then facts least recently self-posted, with seeded random
  ties. Post the task's exact fact text. This simple sharing heuristic allows
  repetitions that can reactivate faded memories. Post `NONE` when no fact is
  citable. Novelty never consults unsampled board messages.
- **Memory:** use the existing acquisition, provenance, reactivation, and
  round-boundary forgetting rules unchanged. Historical but inactive facts are
  excluded until observed again. An observed fact affects the current vote
  even when `active_only` citation rules delay permission to forward it.

Ballots pass through the existing response contract, parser, action validator,
and transition logic. Each decision's action metadata records its evidence IDs,
three allocation probabilities, surviving-world count, and posting policy.
Trajectory and checkpoint formats remain usable; no fake provider attempt is
recorded. Agent request estimates are zero, including checkpoint ensembles.
Paired initializations and recovery reject mixing Bayesian and LLM decisions;
existing LLM artifact identities are preserved.

This is a model-aware reasoning reference: it knows the scoring rule and exact
meaning of every observed fact. It is not a model of language interpretation,
and its posting heuristic is not an optimal communication planner.
