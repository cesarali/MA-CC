# Relational Population Study 09a

Study 09a preserves the four epistemic conditions from Study 08 while keeping
controller target semantics as a separate axis.

## Epistemic conditions

The four labels are derived from two independent settings:

| Receiver disposition | Evidence strategy | Derived condition |
|---|---|---|
| `naive` | `neutral` | `naive_neutral` |
| `naive` | `strategic` | `naive_strategic` |
| `vigilant` | `neutral` | `vigilant_neutral` |
| `vigilant` | `strategic` | `vigilant_strategic` |

Every condition uses `message_mode: recommendation_plus_fact`.

Controller target semantics is separate:

- `truth`: `target: correct`
- `false`: fixed option index `target: 2`, which is wrong for every selected task

Neutral evidence uses the existing deterministic, target-independent first
fact in frozen `fact_order`. Strategic evidence uses the existing target-aware
selector. It may select only a true fact from the frozen task. If no true fact
supports the target, the run is rejected rather than inventing evidence.

## Design

- Fixed intervention budget: `b=12`
- Tasks: `task_0001` and `task_0002`
- Repetitions: 10
- Fixed: `N=24`, `q=1`, `q_c=12`, `theta=0.5`, `beta=4`, 10 rounds
- Shared seed: `20260828`

Arithmetic: `4 epistemic conditions × 2 targets × 2 tasks = 16`
scientific cells. With 10 repetitions, the study contains 160 episodes.

Both target blocks have identical grid order and the same root seed. This
pairs their repetition-index random streams. It does not force later model
responses to be identical.

The study uses the generic automatic cell-array launcher. It adds no
study-specific SLURM job and no replacement information estimator.

## Files

- `study09a_false_epistemic.yaml`: false controller target
- `study09a_truth_epistemic.yaml`: truth controller target
- `analysis.yaml`: reused epistemic and single-affinity analysis family
- `study.yaml`: stable config order and cluster execution policy

Both configs passed live-pricing preflight on 2026-08-28. See `PREFLIGHT.md`
for measured request, token, price, runtime, and scheduler-plan estimates.