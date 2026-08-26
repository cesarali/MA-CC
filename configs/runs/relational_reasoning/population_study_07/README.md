# Population Study 07 — fine policy gain and truth-aligned control

Study 07 inherits the actual Study 06 main configuration and execution setup.
It contains two matched GPT-OSS blocks at `N=24`, `q=1`, `q_c=12`, ten rounds,
four tasks, and ten repetitions per cell.

## Scientific blocks

- `study07_fine_beta_atlas.yaml`: wrong/adversarial Study 06 target, fixed
  `theta=0.5`, with `b=[4,8,12,16,20,24]` and
  `beta=[0.5,1,2,4,8,16]` (144 cells; 1,440 episodes including tasks).
- `study07_truth_aligned_b_theta.yaml`: exact Study 06 main `(b, theta)` grid,
  changing only the controller target from fixed wrong index `2` to the
  existing supported `correct` target (120 cells; 1,200 episodes).

Both retain seed `20260822`, the Study 06 non-CRN grid seeding behavior,
`results_only` retention, GPT-OSS provider settings, recommendation-only soft
control, and the same canonical aggregation and estimator engines.

## Execution and results

Submission uses the same generic automatic cell-array launcher as Study 06:
264 cell shards, throttle 18, eight episode/request slots per active shard,
and an estimated ceiling of 864 RPM. Results and SLURM logs are rooted under
`/work`, never the home source checkout.

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main study submit \
  --config-dir configs/runs/relational_reasoning/population_study_07
```

After every cell seals, aggregate strictly:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/relational_population_study_07
```

See `PREFLIGHT.md` for provider demand, token bounds, and runtime assumptions.

