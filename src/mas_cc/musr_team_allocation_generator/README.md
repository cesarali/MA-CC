# Native MuSR-style Team Allocation generator

This package creates frozen Team Allocation tasks inside MAS-CC. A **latent
problem** is the exact hidden table of skills and cooperation values. It is the
only source of the correct answer. A language model (LLM) writes indirect
natural-language evidence, but never chooses the answer.

The design is inspired by MuSR, “Testing the Limits of Chain-of-thought with
Multistep Soft Reasoning.” The pinned reference revision and license are in
[`attribution.md`](attribution.md).

## What generation does

For each world, the generator:

1. creates three people, two tasks, six skill values, and three cooperation
   values using seeded programmatic code;
2. enumerates the three possible allocations and rejects any world without one
   unique best score;
3. asks an existing MAS-CC provider for several independent evidence branches
   for each of the nine hidden facts;
4. rejects malformed output and direct answer leakage;
5. distributes complete evidence cards, rather than isolated sentences, over
   exactly the requested number of agents;
6. rejects a distribution if one agent can prove a unique answer from the
   hidden-fact provenance it holds;
7. asks the provider to solve the task from all visible evidence and requires a
   configurable majority of correct attempts;
8. writes self-contained JSON tasks and a hash-checked manifest.

Task generation is offline. The population experiment reads frozen files and
makes no generation calls.

## Command

Use an existing MAS-CC provider component. For example, to generate one small
QA dataset with the University provider and `microsoft/gpt-5.6-terra`:

    python -m mas_cc.musr_team_allocation_generator.cli generate \
      --provider configs/components/llm_providers/university.yaml \
      --model microsoft/gpt-5.6-terra \
      --num-tasks 1 \
      --population-size 24 \
      --branches-per-latent-fact 3 \
      --tree-depth 2 \
      --seed 0 \
      --output results/local/musr_team_allocation_smoke

This normally makes nine evidence-generation calls and three validation calls
per accepted world: one batched evidence call per hidden fact, followed by a
three-attempt full-information check. Invalid language or failed semantic QA
can add retry calls.

Validate files and manifest hashes without provider calls:

    python -m mas_cc.musr_team_allocation_generator.cli validate \
      results/local/musr_team_allocation_smoke

`--skip-full-information-validation` exists for local development only. A task
created with that switch records that the check was skipped and must not be
used as a scientifically validated dataset until separate QA is completed.

## Frozen task fields

Each task contains:

- `scenario`, `options`, `gold_index`, and `gold_answer`;
- stable evidence-card IDs and `agent_evidence_ids`;
- exact matrices, candidate scores, winner, and margin under `latent`;
- hidden facts and full entailment trees under `reasoning_provenance`;
- seed, provider, model, prompt version, generation settings, and pinned MuSR
  revision under `generation`;
- exact, structural, leakage, and full-information results under `validation`;
- a content fingerprint.

The older nested adapter, which required a pre-generated MuSR dataset, was
removed. Existing `datasets/` files are legacy frozen outputs only. They are
not inputs to this generator.
