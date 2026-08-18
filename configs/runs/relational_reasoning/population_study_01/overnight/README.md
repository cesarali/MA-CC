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
| `storage.artifact_profile` | `full` | **`results_only`** — see [Artifacts](#artifacts--results_only) |

Plus bookkeeping: `experiment.name` gains `-overnight`, tags gain `overnight`,
`metadata.study` becomes `population_study_01_overnight`.

## What was **not** touched

`N = 24`, `L = 2`, `q = 1`, `rounds = 10`, `r = [1, 3, 6, 12]`, the matched
datasets, semantic votes and the per-call option shuffle, `local_vote`
initialization, `social_distrust: true`, `advocacy_schedule: always`, controller
targets and message modes, the fact selector, seeds, provider, prompts, budget,
`analysis.enabled: false`.

**No dynamics, task, controller or seed changed.** The artifact profile decides
what is written to disk afterwards, not what happens during an episode — the
same trajectory is produced either way.

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

## Artifacts — `results_only`

The full study writes `artifact_profile: full`. The overnight run writes
**`results_only`**, the repository's existing compact profile. One completed
cell keeps:

```text
<run>/
├── manifest.json  resolved_base_config.yaml  grid_summary.csv
├── budget_state.json  comet_summary.json  timing_study.md  grid_progress.png
├── scientific_events.parquet                  # merged across cells
└── cells/cell-NNNN/
    ├── resolved_config.yaml  overrides.json   # exact cell identity
    ├── cell_summary.json  aggregate.json  cell_complete.json
    ├── prompt_examples.md                     # one example, reproducibility
    ├── metrics/plots/*.png
    ├── scientific_events.parquet
    └── round_records/<episode-id>/
        ├── round_trajectory.jsonl             # <- the scientific channel
        └── micro_slot_trajectory.jsonl        # trimmed per-update slots
```

Deleted at cell completion: `trajectory.jsonl` (the full microscopic
trajectory), `events.jsonl`, `experiment.log`, `audit_traces.jsonl`,
`streaming.csv`, `local_metrics.csv`, `api_call_status.jsonl`,
`usage_cost.jsonl`, per-episode manifests and the `prompts/*.md` tree. A
2-episode probe came to **528 KB** total.

### The round record is self-contained

Because the micro trajectory does not survive, every quantity the r-scan and
the control comparison need is written **per round**, not derived afterwards.
`round_trajectory.jsonl` carries one row per population round with 95 fields,
including:

```text
truth_vote_share                    p_truth(t)
mean_supporting_fact_coverage       κ_t
full_proof_agent_share              φ_t
vote_entropy  vote_entropy_before

knowledge_share_k0 / _k1 / _k2      fraction of the population knowing exactly k
truth_share_k0 / _k1 / _k2          fraction OF THAT STRATUM voting correctly
                                    (None when the stratum is empty — "nobody is
                                    here" stays separable from "everybody here
                                    is wrong")
knowledge_stratum_counts            raw counts behind both
truth_counts_by_stratum
… and the same six with a _before suffix

supporting_fact_reach               holders of each supporting fact
peer_fact_exposures                 controller_fact_exposures
new_peer_facts                      new_controller_facts
controller_target_share             (and _before)
controlled_update_count             controlled_off_target_count
controlled_adoption_count
controlled_target_adoption_rate     adoptions / off-target controlled updates,
                                    None when no controlled update had an
                                    off-target focal

m_truth_* m_ctrl_* m_order_* H_vote_* delta_*  occupation_counts_before/after
population_state_before/after  controller_action/target/fact_id/fact_text
controlled_positions + seed + hash  sensor_* ...
```

`truth_share_k*` is the one that makes the r-scan readable: a population-wide
`p_truth` cannot separate *"more agents hold the proof"* from *"agents who hold
the proof use it"*, and sweeping `r` is precisely a sweep over the first while
asking about the second.

Every one of these is checked against the population state itself in
`tests/mas_cc/test_relational_imitation_round_feedback.py` (see
`test_the_summary_agrees_with_the_final_population_state`).
