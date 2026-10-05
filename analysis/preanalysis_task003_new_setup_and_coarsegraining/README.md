# preanalysis_task003_new_setup_and_coarsegraining — fixing the first batch of simulations and estimators

> ## Building the next experiment? Go straight here.
>
> **What are we running?** →
> [`50_next_experiment/EXPERIMENTS.md`](50_next_experiment/EXPERIMENTS.md)
>
> **Building the configs?** →
> [`50_next_experiment/designs/AGENT_RUNBOOK.md`](50_next_experiment/designs/AGENT_RUNBOOK.md)
>
> **Every property and fact list?** →
> [`50_next_experiment/designs/`](50_next_experiment/designs/)
>
> **The two setups are frozen (5 October 2026)**: `task003-symmetric` (agents
> can prove the answer; two controller pools, one per target) and
> `task003-nosolution` (agents cannot; one shared pool). Neither has been built
> into a task directory yet.
>
> The rest of this directory is the **analysis those designs came out of**. You
> do not need it to build them, but §10 of that README says which part answers
> what.

**Start here.** One goal, three strands: understand what the task actually is,
fix the controller so the two steering directions are comparable, and find a
state representation in which information quantities can be estimated at all.

Everything is offline. No simulations were run and no provider requests made.
Inputs: the archived `21-09-2026-full-vs-report-v1` study under
`/Users/rsanchez/Projects/agents_control/new_rnd_init_experiment` (read-only),
and the generator source in `src/mas_cc/musr_team_allocation_generator`.

## Read in this order

| # | folder | question it answers |
|---|---|---|
| **10** | [`10_task_and_facts/`](10_task_and_facts/) | What is task_003? What are the 49 facts, who holds them, and who can prove the answer? |
| **20** | [`20_controller_redesign/`](20_controller_redesign/) | Why is the truth-vs-decoy comparison unfair, and what replaces it? |
| **30** | [`30_coarse_graining/`](30_coarse_graining/) | Can we compress the population state enough to estimate anything? Is the result Markovian? |
| **40** | [`40_information_estimates/`](40_information_estimates/) | The actual path divergences, mutual informations and efficiencies |
| **50** | [`50_next_experiment/`](50_next_experiment/) | The next study: the two frozen setups (`task003-symmetric`, `task003-nosolution`), how to build them, and the run plan |
| **60** | [`60_exact_simulation/`](60_exact_simulation/) | An LLM-free mirror of the game (agents compute exact posteriors). Its task003-symmetric runs used the superseded single-pool design |
| **70** | [`70_codex_simulations/`](70_codex_simulations/) | A larger LLM-free study and mean-field test. Same caveat as 60 for task003-symmetric |

Shared: [`scripts/`](scripts/) reproduces 30 and 40; [`results/`](results/) holds the
per-round state files and run logs (gitignored).

The whole thing as one 8-page document with the equations typeset:
[`preanalysis_task003_report.pdf`](preanalysis_task003_report.pdf) (source in
[`report/main.tex`](report/main.tex), rebuild with `scripts/build_report.py`).

## The three findings that matter

**The controller is not what it claims.** Fact admission is evaluated once for
the *decoy* target and reused for both configurations, so both controllers quote
the same 24 decoy-favouring facts. The truth-targeting arm recommends
`ALLOCATION_0` while citing evidence selected to undermine it, and differs mainly
in firing 32% of rounds against 54%. Measured target information is **0.022 bits**
— essentially zero.

**The population holds the only proof, and barely.** All six decisive facts sit in
agents' starting packets and none in the controller's pool. Of 7,694 minimal
proofs, agents can assemble **4**; the controller can assemble **0**.

**Estimation needs a coarse-graining, and it only fully works at ρ = 1.** The
325-state vote vector defeated the earlier critic-based estimators. Binning the
mean agent posterior into 4 states makes path KL exactly computable — but four
independent tests agree the chain is Markovian **only in the silent arm at
persistence 1**. At ρ = 0.75 it is an approximation carrying ~0.015 nats/step.

## Standing caveats

- **The A0 arm is confounded** and must not be read as evidence about truth-directed
  steering. Its recorded dynamics are sound and are used where only dynamics matter.
- **One task, one model, 60 initializations.** Everything is task_003 with
  `gpt-oss-120b`.
- **One task, one model.** See above.
- This directory *is* versioned on `dev/rsanchez`, unlike the rest of
  `results/`: the root `.gitignore` excludes `/results/*` and `data/`, and both
  are overridden for this subtree. Run logs, `__pycache__/` and LaTeX
  intermediates stay ignored (see `.gitignore` here).

## Authorship note

`10_task_and_facts/task_003_redesign.md` and `audit_target_pools.py` were written
in a separate session on 1 October and are preserved unchanged. They cover the
pool correction from a different angle and reach compatible conclusions.
