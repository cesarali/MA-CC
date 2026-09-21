# Notation to configuration

| Symbol | Meaning | Configuration / artifact field |
|---|---|---|
| `K` | parents per prepared setting | `ensemble.parent_count` |
| `L` | preparation rounds | `ensemble.preparation_rounds` |
| `M` | continuation rounds | `ensemble.continuation_rounds` |
| `B` | continuation copies | `ensemble.continuation_copies` |
| `N` | population size | `game.population_size` |
| `q` | board sample size | `game.options.social_group_size` |
| `rho` | epistemic persistence | `game.options.epistemic_persistence` |
| `b` | posting budget | `ensemble.posting_budgets` (applied to `control.options.intervention_budget`) |
| `s` | controller sensor size | `control.options.sensor_sample_size` |
| `theta` | sensing threshold | `control.options.threshold` |
| `beta` | sensing slope | `control.options.beta` |
| `h` | post-checkpoint horizon | `post_branch_horizon` |
| `r` | absolute population round | `absolute_round` |
| `A` | assigned branch label | derived from `branch_policy` and `posting_budget` |
| `n_0` | checkpoint target count | shared `h=0` checkpoint observation |
| `m_h` | target count at horizon `h` | semantic option count in the branch endpoint |

The parent—not a branch round, continuation, or scheduler shard—is the
independent resampling and fold-assignment unit.
