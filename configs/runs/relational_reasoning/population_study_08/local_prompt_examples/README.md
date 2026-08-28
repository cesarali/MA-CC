# Study 08 local epistemic-prompt examples

## Minimal one-episode prompt inspection

For the smallest locally launched real-model prompt-class grid (four cells,
one repetition per cell, six agents, four rounds, no metrics, and one retained
prompt per cell), ensure
`POTSDAM_API_KEY` and `BASE_POTSDAM_LLM_URL` are set, then run:

```bash
/home/cesarali/miniconda3/envs/MA-CC/bin/mas-cc experiment run \
  --config configs/runs/relational_reasoning/population_study_08/local_prompt_examples/study08_minimal_local_prompt_inspection.yaml \
  --output-dir results/local/study08_minimal_real_prompt_grid
```

Each cell sends 30 expected real provider calls: six initial ballots plus six
agent updates in each of four rounds. Across the four prompt-class cells, that
is four episodes and 120 expected calls. The grid is:

- `naive_neutral`
- `naive_strategic`
- `vigilant_neutral`
- `vigilant_strategic`

The new result path deliberately differs from the earlier mock run, so its
first execution cannot resume the mock episode. On later executions, the
default resume behavior avoids accidentally paying for completed cells again.
Add `--no-resume` only when you deliberately want to repeat all 120 calls.

The compact run uses `artifact_profile: results_only`; each cell has its own
human-readable `prompt_examples.md` and `overrides.json`. The overrides file
identifies that cell's prompt class. Find and print all four prompts with:

```bash
find results/local/study08_minimal_real_prompt_grid -name prompt_examples.md -print -exec sed -n '1,240p' {} \;
```

Inspect the rest of the compact result tree with:

```bash
find results/local/study08_minimal_real_prompt_grid -maxdepth 6 -type f | sort
```

## Four-class prompt examples

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
