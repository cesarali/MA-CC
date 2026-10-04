# task_004 handover

Status as of 2026-09-25. Nothing in this work is committed; `git status` shows
the full set.

## What task_004 is

A harder variant of `task_003`: the 6 decisive facts are removed, so no fact
settles the answer and the group must combine weak evidence. 15 agents (was
24), q=7, 30 rounds, pooled P(truth)=0.737. Ground truth `ALLOCATION_0`,
task-declared false target `ALLOCATION_2`. Lives at
`results/studies/musr_truthful_selective_task_calibration_01/tasks/task_004`.

## Done: no-control arms (SLURM 305, 307 on Cesar, both COMPLETED)

20 episodes each at rho=1.0 and rho=0.75, gridded over communication profile.

|                    | rho=1.0        | rho=0.75       |
|--------------------|----------------|----------------|
| report_only        | 88.7%, 9/10    | 89.3%, 9/10    |
| full_communication | 81.3%, 9/10    | 44.7%, 4/10    |

(mean final truth vote share, episodes where truth wins)

`report_only` is flat in rho; `full_communication` collapses. Mechanism:
`full_communication` lets an agent post without citing a fact, so evidence
circulation drops (36-246 cited posts/episode vs 430-450 under `report_only`).
Within rho=0.75 full_communication, wins average 170 cited posts vs 72 for
losses; r=+0.72 between cited posts in rounds 1-5 and the final truth share
(n=10, suggestive not conclusive - early citing is entangled with early truth
share).

The current comparison figures are in `results/local/task_004_panels/`, grouped
by rho. Its `src/data/` contains compact plot-ready trajectories for the
no-control, episode-b90, and round-b3 conditions; `src/manifest.json` records
their Cesar result roots. Regenerate all figures with
`.venv/bin/python results/local/task_004_panels/src/render.py`.

## Built, not yet run: false-control arms

`configs/runs/relational_reasoning/blackboard_game/iclr_experiments/25-09-2026-task004-false-rho{1,075}/`

Both remain preflight-clean in local deployment-adjusted copies. Key settings:

    target: ALLOCATION_2          # task-declared false target, NOT ALLOCATION_1
    intervention_budget: 30       # whole game, not per round
    controller_budget_scope: episode
    sensor_sample_size: 100      # full previous-day board (maximum 39 messages)
    controller_max_posts_per_round: omitted  # no daily cap
    advocacy_schedule: always     # actuation gate off; controller decides
    controller_report_cooldown_rounds: 0
    controller_report_max_posts_per_fact: 30   # = budget, cannot bind
    controller_authoring: llm_authored
    parallelism: 10, request_concurrency: 200

Grid is `[report_only]` only: `full_communication` + episode budget raises by
design (no full policy lets the controller choose its own message count).
Initialisations point at the no-control arms' artifacts so the pairing holds.

## Code changes (uncommitted, ~3 source files)

New whole-game budget mode. The codebase encoded "b is a per-round quota" in
five independent places; each is now scoped, not deleted, so per-round configs
and v0/v1 semantics are untouched:

1. `budget <= controller-reportable pool (24)` - per-round only
2. `budget <= N agents` - per-round and slot-scheduled modes only
3. exact-fill policy selection - `report_only`+`episode` now derives
   `llm_authored_report_only_v1` at `runtime.py:~1130`
4. "exactly b reports" post-check - per-round only
5. per-round offer capped at `min(remaining, pool, max_posts_per_round)`

Also: new `HOLD` communication mode (post nothing, keep the allowance); the
failure fallback now holds instead of posting `eligible_facts[0]`; controller
prompt rewritten (scenario/question/options/game/role + markdown tables
replacing the JSON payload). The Task 004 revision explains the 30-day board
cycle and makes spending optional before asking for fact IDs. The prompt now
shows human day numbers rather than zero-based runtime round indices. The old
`controller_prompt_v2_verbatim.md` is an illustrative draft, not a live trace.

Current local check: the focused prompt and blackboard tests pass. The broader
MuSR blackboard file still has missing `task_001` and Cesar-only `task_003`
fixtures in this checkout, so use the failure names rather than the raw count
when comparing suites.

## Open issues

### Optional controller public memory (local, not yet synced)

`control.options.controller_memory_mode: public_ledger` is an opt-in mode for
the board-sensed, LLM-authored controller. Omitting it preserves the current
prompt. The runtime records only message IDs the controller sampled at each
dawn, including days when the binary policy declines to post, and carries this
observation list across round-boundary continuations. The prompt keeps
yesterday's board verbatim and adds compact day, participant, and fact-citation
tables. Votes in those tables are last *posted* votes of distinct observed
participant authors, never a population census; controller posts are separate.
This first version requires one-day board message lifetime.

The matched one-episode config is
`configs/runs/smoke/task004_public_ledger_smoke/`. Do not sync it to Cesar while
the current worker is active; `scripts/Cesar/sync.sh` rejects that for good
reason. Compare against the memory-free smoke before promoting this option to
the false-control production configs.

1. **The controller never held in the earlier smoke.** It spent its full
   per-round allowance (5/5 with cap 2; 3-6/round uncapped). The cap-2 run
   exhausted the 30 messages by day 16. The new prompt asks it to weigh posting
   against waiting; the cap is now removed. Re-run a smoke before drawing a
   behavioral conclusion.
2. The controller now sees the entire *previous day's board* in the Task 004
   false-control configs. The sensor still accepts a finite message count, set
   to 100; at most 15 agent posts plus 24 distinct controller reports can be
   created in one day. This is a deliberate full-information pilot. A later
   partial-board study can restore a smaller `sensor_sample_size` without
   changing agent board sampling. The controller call remains fresh each day:
   it receives yesterday's board and its own posting history, but no earlier
   board or vote time series.
3. Smoke config: `configs/runs/smoke/task004_hold_smoke/`. Use a new output
   location or clear the old smoke output before re-running, because
   `checkpoint_mode: episode` can resume the prior design.
4. An episode is ~480 *sequential* LLM calls (agents cascade within a round and
   cannot be parallelised). ~55 min/episode. Parallelism is across episodes
   only. Note `request_concurrency` feeds the cluster planner: at 200 it
   collapses `rpm_bound` to 1, harmless at 1 shard but would serialise shards
   if a second cell is ever restored.
5. Controller prompts are no longer byte-identical to the v0/v1 archives
   (scenario preamble + markdown tables). Accepted: task_004 is a new baseline.
6. v0/v1 controllers never knew what `ALLOCATION_n` meant - they were told to
   push a label they could not interpret. Worth stating as a limitation of
   those archives; may bear on the "truth controller hurts" result.

7. The eight-spent/two-posted discrepancy in
   `controller_prompt_v2_verbatim.md` came from its manually assembled example.
   In the retained smoke trajectory, every controller call has equal counts for
   spent messages, posting-history rows, and per-fact prior posts. The example
   is not a valid runtime snapshot.

## Next steps

1. Run one fresh HOLD smoke with the full-board, no-cap prompt.
2. Inspect its actual daily posting choices and rendered prompts.
3. Prepare and submit both false-control studies to Cesar if the smoke is sound.
4. Plot and compare against the no-control baselines above.
