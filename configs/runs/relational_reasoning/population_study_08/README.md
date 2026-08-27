# Relational Population Study 08

Study 08 compares four categorical epistemic prompt classes under matched
wrong/adversarial and truth-aligned recommendation-only control.

## Design

- Prompt class: `naive`, `distributed_information`,
  `strategic_uncertainty`, `evidence_calibrated`
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

The analysis recipe preserves the complete Study 06/07 estimator set and adds
prompt-class/semantic comparison plots with shared color scales. Canonical
rounds retain pre-round target share, `phi`, `kappa`, time, truth/target state,
and support fields for state-local comparisons without another provider run.
