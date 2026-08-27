# Study 08 local epistemic-prompt examples

This is a zero-network, zero-cost inspection variant of Study 08. It keeps the
Study 08 reasoning game and prompt-class axis, while fixing all other axes to
one small condition:

- four cells, one for each `epistemic_prompt_class`;
- six agents;
- three population rounds per episode;
- three independent episodes per cell (12 episodes total);
- one task, one controller condition, and no repeated grid values;
- deterministic local mock provider;
- two retained prompt examples per cell.

Run from the repository root:

```bash
/home/cesarali/miniconda3/envs/MA-CC/bin/mas-cc experiment preflight \
  --config configs/runs/relational_reasoning/population_study_08/local_prompt_examples/study08_local_epistemic_prompt_examples.yaml \
  --output-dir results/inspection/study08_local_epistemic_prompt_examples_preflight

/home/cesarali/miniconda3/envs/MA-CC/bin/mas-cc experiment run \
  --config configs/runs/relational_reasoning/population_study_08/local_prompt_examples/study08_local_epistemic_prompt_examples.yaml \
  --output-dir results/local/study08_epistemic_prompt_examples
```

Each cell writes `prompt_examples.md`. Use its `overrides.json` to identify the
prompt class assigned to that cell.
