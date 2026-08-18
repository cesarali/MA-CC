# Population study 01 — overnight reduction

The same four scientific arms as [the full study](../README.md), sized to
finish in one overnight cluster run.

**Derived from the full configs, not rewritten.** Each file here is its parent
with exactly three reductions applied, plus a renamed run identity so results
do not collide. Verify that claim yourself:

```bash
cd configs/runs/relational_reasoning/population_study_01
for slug in a_no_control b_social_control c_epistemic_control d_adversarial_diagnostic; do
  diff <(grep -v '^\s*#' relational_population_study01_${slug}.yaml            | grep -v '^\s*$') \
       <(grep -v '^\s*#' overnight/relational_population_study01_${slug}_overnight.yaml | grep -v '^\s*$')
done
```

---

## What was reduced

| | full | overnight |
|---|---|---|
| worlds | 10 (`task_0001`…`task_0010`) | **5** (`task_0001`…`task_0005`) |
| repetitions | 3 | **2** |
| `b` levels (arms B, C) | `[6, 18, 24]` | **`[6, 24]`** — `b/N = 0.25, 1.0` |

Plus bookkeeping: `experiment.name` gains `-overnight`, tags gain `overnight`,
`metadata.study` becomes `population_study_01_overnight`.

## What was **not** touched

`N = 24`, `L = 2`, `q = 1`, `rounds = 10`, `r = [1, 3, 6, 12]`, the matched
datasets, semantic votes and the per-call option shuffle, `local_vote`
initialization, `social_distrust: true`, `advocacy_schedule: always`, controller
targets and message modes, the fact selector, provider, prompts, budget,
storage, observables, `analysis.enabled: false`.

Each episode is still full-resolution: 24 agents × 10 rounds. This is *fewer
episodes*, not smaller ones.

---

## Size

| Arm | Config | Cells | Episodes | Calls |
|---|---|---|---|---|
| **A** no control | `..._a_no_control_overnight.yaml` | 20 | 40 | 10,560 |
| **B** pure social | `..._b_social_control_overnight.yaml` | 40 | 80 | 21,120 |
| **C** epistemic | `..._c_epistemic_control_overnight.yaml` | 40 | 80 | 21,120 |
| **D** adversarial | `..._d_adversarial_diagnostic_overnight.yaml` | 20 | 40 | 10,560 |
| | **total** | **120** | **240** | **63,360** |

264 calls per episode = 24 local initial votes + 24 agents × 10 rounds.

Cell counts: A and D are `4 r × 5 worlds`; B and C are `4 r × 5 worlds × 2 b`.

---

## The arms, unchanged

| Arm | Control | Target | Message | `b` |
|---|---|---|---|---|
| A | `none` | — | — | — |
| B | `relational_round_budgeted` | `correct` | `recommendation_only` | 6, 24 |
| C | `relational_round_budgeted` | `correct` | `recommendation_plus_fact` (selector `supporting`) | 6, 24 |
| D | `relational_round_budgeted` | `random_incorrect` | `recommendation_only` | 24 |

All controlled arms keep `advocacy_schedule: always`. B versus C is still the
headline contrast — identical but for one injected supporting fact. D still
separates "the population reasoned" from "the population followed".

Dropping the middle `b = 18` rung leaves the two ends of the actuation ladder,
so `b/N` is a two-point contrast rather than a three-point curve in this run.
That is the one place the reduction costs resolution rather than just power.

---

## Launch

Preflight first — no model inference, but `pricing.mode: live` fetches a price
list, so run it where the provider credentials are:

```bash
cd /path/to/MA-CC
for arm in a_no_control b_social_control c_epistemic_control d_adversarial_diagnostic; do
  conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
    --config configs/runs/relational_reasoning/population_study_01/overnight/relational_population_study01_${arm}_overnight.yaml \
    --output-dir inspection/relational_study01_${arm}_overnight_preflight
done
```

Then the four runs. They are independent — run them sequentially overnight, or
in parallel if the provider tolerates 4 × `request_concurrency: 8`:

```bash
mkdir -p logs
for arm in a_no_control b_social_control c_epistemic_control d_adversarial_diagnostic; do
  conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
    --config configs/runs/relational_reasoning/population_study_01/overnight/relational_population_study01_${arm}_overnight.yaml \
    2>&1 | tee logs/relational_study01_${arm}_overnight.log
done
```

`storage.overwrite: false` and `checkpoint_mode: episode` mean an interrupted
run resumes where it stopped: re-issue the same command.

If you only have time for part of it, **A and D first** (80 episodes, 21,120
calls). They bracket the effect — unaided collective reasoning, and maximal
social pressure against the truth — and tell you whether B and C are worth
their 42k.

---

## Reading the results

Unchanged from the full study: per round, off `round_trajectory.jsonl`,

```text
κ_t = mean_supporting_fact_coverage
φ_t = full_proof_agent_share
p_truth(t) = truth_vote_share
```

plus peer/controller exposures and acquisitions, `controller_target` (the
realized wrong option in arm D), and the full microscopic trajectory in
`trajectory.jsonl`.
