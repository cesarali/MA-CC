# theory_claude/ — Claude's theory work: models of the reading reversal and persistence, tested on two tasks

This folder holds the theory: small models that try to explain two effects seen in the LLM-free simulator
(Simulation 2), and the reports that judge them. The data the models are judged against live with each task:

| task | what it is | where its data, setups and scorecards live |
|---|---|---|
| **task003** | team allocation: 3 people, 2 jobs, 3 candidate allocations (the original task) | [`../task003_llm_free/`](../task003_llm_free/), study folder [`simulation_2/key_phenomena_hypothesis/`](../task003_llm_free/simulation_2/key_phenomena_hypothesis/) |
| **Object Placements** | MuSR's "where would person P look for item I" stories (a second task, used as an out-of-sample test) | [`../object_placements_llm_free/`](../object_placements_llm_free/) |

**The two effects.**
- *Reading reversal:* reading more posts (larger q) helps a false controller when agents forget fast, and protects
  the swarm when agents remember (memory ρ close to 1).
- *Persistence:* the controller's effect stays long after it stops posting.

**Start here:** [`reports/2026-10-22_theory_review/theory_review_report.md`](reports/2026-10-22_theory_review/theory_review_report.md),
all findings in one place, written for a reviewer: why M1/M2 fail, M3 and M4 on task003 (K = 1, 2, 4) and on Object
Placements.
The same findings as a paper, with numbered equations for M1–M4 and nine figures:
[`reports/2026-10-23_theory_report/theory_report.pdf`](reports/2026-10-23_theory_report/theory_report.pdf)
(source `theory_report.tex`; figures rebuilt by `make_figures.py`).

## The models

All four use the simulator's own rules for one agent (exact Bayesian posterior, vote, choice of what to post; imported
from the simulator code, so they cannot drift from it). They differ in how agents interact.

| model | one-line description | verdict |
|---|---|---|
| M1 | deterministic mean field: the board is an average; each agent's memory is redrawn fact by fact | fails (held-out signs at chance) |
| M2 | as M1, but each agent keeps its exact memory | fails, same reason: no fact can go extinct |
| M3 | M2 + whole-number holder counts per fact in a 24-agent swarm; a fact with no holder is gone for good (extinction) | partial: signs mostly right, effects ~60% |
| M4 | M3 + one shared, realised board (a finite list of posts that all agents read) | working theory; also passes the pre-registered Object Placements test |

"Mean field" = each agent interacts with the average of all others instead of with specific agents.

## What is here

| path | what |
|---|---|
| [`00_microscopic_rules.md`](00_microscopic_rules.md) | the exact simulator rules every model uses |
| [`M1_M2_mean_field.md`](M1_M2_mean_field.md), [`M3_extinction.md`](M3_extinction.md) | write-ups of M1/M2 (and the diagnostics of why they fail) and of M3; M4 is described in the report, §1.4 |
| [`models/mean_field.py`](models/mean_field.py) | the four models (`Spec(closure="M1".."M4")`) |
| [`models/run_models.py`](models/run_models.py) | runs a model over a set of conditions: `--set dev|all|post_stop|k2|k4|op|check` |
| `models/diag_*.py` | one-off diagnostics behind `M1_M2_mean_field.md` (finite board, replicated agents, pinned facts) |
| [`reports/`](reports/) | dated reports; never replaced, new ones go in new dated folders |
| `results/theory_runs/` | model outputs, `<model>_<set>.parquet` (not in git; rebuild with `run_models.py`) |

## Where each result is scored

| result | scored by | scorecard |
|---|---|---|
| task003, development/held-out split | [`held_out_validation/score.py`](../task003_llm_free/simulation_2/key_phenomena_hypothesis/held_out_validation/score.py) | `held_out_validation/scorecard_M{1..4}*.csv` |
| task003, after the controller stops | [`held_out_validation/score_post_stop.py`](../task003_llm_free/simulation_2/key_phenomena_hypothesis/held_out_validation/score_post_stop.py) | `held_out_validation/scorecard_M{3,4}_post_stop*.csv` |
| task003, K = 1, 2, 4 projects | [`finite_size/score_models_K.py`](../task003_llm_free/simulation_2/key_phenomena_hypothesis/finite_size/score_models_K.py) | `tables/scaling_K_models.csv` |
| Object Placements (pre-registered) | [`op_predictions_and_score.py`](../object_placements_llm_free/simulation_2/analysis/op_predictions_and_score.py) | `preregistered_M{3,4}.csv`, `scorecard_op_M{3,4}*.csv` |

The pre-registered split (which task003 data may be used to choose a model) is
[`calibration/holdout_preregistration.md`](../task003_llm_free/simulation_2/key_phenomena_hypothesis/calibration/holdout_preregistration.md).
The plan behind this work and the one-line-per-decision log are in the task003 study folder:
[`00_theory_plan_2026-10-21.md`](../task003_llm_free/simulation_2/key_phenomena_hypothesis/00_theory_plan_2026-10-21.md),
[`progress.md`](../task003_llm_free/simulation_2/key_phenomena_hypothesis/progress.md).

## Example

From the repository root (`caffeinate -i` keeps the Mac awake):

```
caffeinate -i .venv/bin/python analysis/theory_claude/models/run_models.py --closure M4 --set op --seeds 20 --replicas 20 --workers 7
```

writes `analysis/theory_claude/results/theory_runs/M4_op.parquet`: one row per (setup, ρ, q, arm, realisation, time) with the
vote shares and the controller's messages.

## History

Moved here on 23 October 2026 from `task003_llm_free/simulation_2/key_phenomena_hypothesis/theory/` (with `git mv`,
so `git log --follow` shows the earlier history), because the models are now tested on two tasks. The code was changed
only in its paths; reruns from here reproduce the stored M4 outputs exactly. The pre-registration document still links to
the old `../theory/` location; it is left unedited because it is the record of what was fixed in advance.

**Renamed on 27 October 2026** from `analysis/theory/` to `analysis/theory_claude/`, because a second theory agent
(ChatGPT) now works in `analysis/theory_chatgpt/`, and the two agents exchange dated notes in
`analysis/theory_blackboard/`. Code paths and the living READMEs were updated. Dated reports, pre-registrations and the
progress log keep the old path `analysis/theory/` as written; read it as `analysis/theory_claude/`.
