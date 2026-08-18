# Relational reasoning — population study 01

The first population experiment on `relational_imitation_round_feedback`.
Four arms, one config each, launchable independently.

**Nothing here has been run against a live model.** The datasets are generated
and validated, the configs resolve and expand, and one cell of arm C was
executed end to end on the mock provider to confirm the observables. Preflight
and launch are yours.

---

## Shared design

| | |
|---|---|
| population | `N = 24` |
| reasoning depth | `L = 2` supporting facts |
| social slots | `q = 1` |
| answers | `K = 3` |
| distractors | 4 |
| rounds | 10 population rounds (= 240 microscopic updates) |
| initialization | `local_vote` — each agent decides once from `K_i(0)` alone |
| worlds | 10 |
| repetitions | 3 per cell |
| hidden profile | `no_single_agent_solution = true` at **every** `r` |

Votes are **semantic** (compass relations). Each LLM call gets its own seeded
`A`/`B`/`C` shuffle and the returned letter is resolved back to the relation
immediately, so no globally shared letter can become a population attractor.

---

## The `r` axis

`r` is how many of the 24 agents initially hold each supporting fact.

| `r` | holders per supporting fact | agents starting with no facts |
|---|---|---|
| 1 | 1 / 24 | 18 |
| 3 | 3 / 24 | 15 |
| 6 | 6 / 24 | 11 |
| 12 | 12 / 24 | 0 |

`r = 12` is the tightest feasible point of the generator's
`L·r ≤ N·(L−1)` constraint for `N = 24, L = 2` — each fact is held by half the
population and still nobody can answer alone.

**Matched by construction.** For a given world, the world, question, supporting
facts, distractors, correct relation and answer set are byte-identical across
all four `r` values; only `agents[*].fact_ids` and
`generation.support_redundancy` differ. Enforced by
`tests/mas_cc/test_relational_task_data.py::test_changing_r_changes_only_the_information_allocation`.

Datasets:

```text
src/mas_cc/relational_task_generator/relational_task_generator/datasets/
├── pop24_L2_r01/   ├── pop24_L2_r03/   ├── pop24_L2_r06/   └── pop24_L2_r12/
```

Each holds `task_0001.json` … `task_0010.json`, a `manifest.json` and an
`INSPECTION.md`, and passes the generator's own `validate_dataset.py`
(including its seed-regeneration check).

---

## The arms

| Arm | Config | Control | Target | Message | `b` | Cells |
|---|---|---|---|---|---|---|
| **A** no control | `..._a_no_control.yaml` | `none` | — | — | — | 40 |
| **B** pure social | `..._b_social_control.yaml` | `relational_round_budgeted` | `correct` | `recommendation_only` | 6, 18, 24 | 120 |
| **C** epistemic | `..._c_epistemic_control.yaml` | `relational_round_budgeted` | `correct` | `recommendation_plus_fact`, selector `supporting` | 6, 18, 24 | 120 |
| **D** adversarial | `..._d_adversarial_diagnostic.yaml` | `relational_round_budgeted` | `random_incorrect` | `recommendation_only` | 24 | 40 |

All controlled arms use `advocacy_schedule: always` — the controller advocates
every round rather than reacting to what it sensed. Under the closed soft loop
the actuation a population receives is a function of that population's own
state, so a cell where the population converged early is a cell where the
controller mostly stopped, and *"did control move it"* and *"did it need
moving"* are confounded. Open loop fixes the intervention and lets the
population vary. Sensing still runs and is still logged.

**B versus C is the headline contrast**: identical in every respect except that
C attaches one real supporting fact to the same recommendation. The difference
is the value of *evidence* over *advocacy*.

**D is the diagnostic that makes the others readable.** Under a wrong target
`m_ctrl` and `m_truth` move in opposite directions, so "the population reasoned
its way to the answer" and "the population followed whoever spoke" stop being
the same observation.

---

## Cost

| Arm | Cells | Episodes | Provider calls |
|---|---|---|---|
| A | 40 | 120 | 31,680 |
| B | 120 | 360 | 95,040 |
| C | 120 | 360 | 95,040 |
| D | 40 | 120 | 31,680 |
| **total** | **320** | **960** | **253,440** |

264 calls per episode = 24 local initial votes + 24 agents × 10 rounds.

That is a large first study. Two obvious ways to cut it before launching:

* **run A and D first** (80 cells, 63,360 calls). They bracket the effect —
  unaided collective reasoning, and maximal social pressure against the truth —
  and tell you whether B and C are worth their 190k calls.
* **drop to 5 worlds or 2 repetitions**, which halves or two-thirds the whole
  study while keeping every arm.

---

## Observables

Read straight off `round_trajectory.jsonl`, per round:

```text
κ_t = mean_supporting_fact_coverage   (1/N) Σ_i |K_i(t) ∩ S| / |S|
φ_t = full_proof_agent_share          (1/N) Σ_i 1[S ⊆ K_i(t)]
p_truth(t) = truth_vote_share
```

alongside `peer_fact_exposures`, `controller_fact_exposures`, `new_peer_facts`,
`new_controller_facts`, `controller_target` (the realized wrong option in arm
D), `controller_fact_id`, and the `m_truth` / `m_ctrl` / `m_order` / `H_vote`
family. The microscopic trajectory in `trajectory.jsonl` carries per-update
`focal_known_fact_ids_before` / `_after`, `peer_exposed_fact_ids`,
`new_peer_fact_ids`, `new_controller_fact_ids`, `vote_before`, `vote_after`.

`analysis.enabled: false` in every arm — no MI/CMI/TE yet, by design.

---

## Commands

Preflight first; it makes no model calls. (`pricing.mode: live` does fetch a
price list, so it needs the provider credentials.)

```bash
for arm in a_no_control b_social_control c_epistemic_control d_adversarial_diagnostic; do
  conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
    --config configs/runs/relational_reasoning/population_study_01/relational_population_study01_${arm}.yaml \
    --output-dir inspection/relational_study01_${arm}_preflight
done
```

Revalidate the datasets at any time (deterministic, no model calls):

```bash
cd src/mas_cc/relational_task_generator/relational_task_generator
for r in 01 03 06 12; do python validate_dataset.py datasets/pop24_L2_r$r; done
```

Check the matched-design and position-bias invariants:

```bash
conda run -n MA-CC --no-capture-output python -m pytest \
  tests/mas_cc/test_relational_task_data.py \
  tests/mas_cc/test_relational_imitation_round_feedback.py -q
```

Launch one arm:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/population_study_01/relational_population_study01_a_no_control.yaml
```

Regenerate a dataset byte-identically:

```bash
cd src/mas_cc/relational_task_generator/relational_task_generator
python generate_dataset.py --num-tasks 10 --population-size 24 --reasoning-depth 2 \
  --support-redundancy 6 --distractors 4 --distractor-redundancy 1 --num-options 3 \
  --seed 20260818 --no-single-agent-solution --output datasets/pop24_L2_r06
```
