# experimental_setup — the two frozen setups

**Status: both setups are frozen (5 October 2026).** Nothing has been built into
a task directory or run with language models yet.

## The world they share

Both setups use the same world, `task_003`: three people, two jobs, three
candidate allocations. The hidden numbers are `(3,1,1,2,2,1,1,1,2)` and the
allocation scores 8 / 4 / 6, so **`ALLOCATION_0` (A0) is correct** and
`ALLOCATION_2` (A2) is the wrong answer a false controller steers toward.
There are 49 true facts; every fact either agents or controllers ever hold is one
of them. What each fact says and how strong it is:
[`../task_and_facts/task_003_facts.csv`](../task_and_facts/task_003_facts.csv).

In both setups: **24 agents, each holding exactly one fact**, and a controller
that posts true facts from its **pool** to steer the agents.

## The two setups

| | [`task003_symmetric/`](task003_symmetric/) | [`task003_nosolution/`](task003_nosolution/) |
|---|---|---|
| agents | hold all 6 **decisive** facts, so together they **can prove A0** | hold **no** decisive fact; together they sit at exactly 50/50 between A0 and A2 and **cannot prove anything** |
| controller pools | **two**, one per target: `controller_pool_A0.json` when steering to A0, `controller_pool_A2.json` when steering to A2 | **one**, `controller_pool.json`, for both targets: 4 facts favouring each of A0, A1, A2 |
| what makes it fair | both pools bring the agents 10 new facts of matched strength | one unbiased menu for both targets, and no controller can ever complete a proof |
| question it answers | Can the swarm assemble a proof it owns, and can a controller help or stop it? | Can a controller tip a swarm that is genuinely undecided? |

Each folder holds everything about its setup: a `README.md` (what, why, how it
was built), the agents, the pool or pools, and the `build.py` that made them.

"Decisive" means one of the 6 facts that, together, prove A0. Definitions of
this and of "eligible" are in
[`../task_and_facts/task_003_analysis.md`](../task_and_facts/task_003_analysis.md).

## How the files are written

Every file lists **fact IDs** such as `cf_x00_eq_3`, plus, next to each ID, what
the fact says and how strong it is, so a person can read it without the CSV.

**`agents.json`**

| field | what it is |
|---|---|
| `agent_assignments` | **the part code uses**: `{"agent_001": ["cf_x00_ge_x04"], ...}`, 24 entries, one fact each |
| `facts` | each distinct fact: its text, `dP` (how much it raises each allocation's probability above ⅓, read alone), which allocation it `favours`, its `agent_group`, how many agents hold it |
| `properties` | checks: joint posterior, decisive facts held, proofs available, and more |

**`controller_pool*.json`**

| field | what it is |
|---|---|
| `used_when_controller_targets` | which target this pool serves |
| `fact_ids` | **the part code uses**: the pool, in order |
| `facts` | each fact's text and strength; in the symmetric pools also its `partner_in_other_pool` |
| `properties` | checks: whole-pool posterior, whether it proves anything, overlap with the agents |

Example, one entry of `task003_symmetric/controller_pool_A0.json` → `facts`:

```json
{"fact_id": "cf_x00_eq_3",
 "text": "Alice's skill for build the data pipeline is strong.",
 "dP": {"ALLOCATION_0": 0.252, "ALLOCATION_1": -0.126, "ALLOCATION_2": -0.126},
 "favours": ["ALLOCATION_0"],
 "eligible_for_one_agent": false,
 "decisive": false,
 "partner_in_other_pool": "cf_x05_eq_1",
 "held_by_agents": 0}
```

## Rebuilding

Each `build.py` writes its folder's JSON files and gives the same result every
run. Run from the repository root with `.venv/bin/python`.

## Other folders

| folder | what it is |
|---|---|
| [`archived_reference/`](archived_reference/) | the setup actually run in September (study `21-09-2026-full-vs-report-v1`), with César's original 24-fact pool. A reference only; never build it |
| [`llm_runs/`](llm_runs/) | the plan for running both setups with language-model agents: arms, phases, cell counts ([`EXPERIMENTS.md`](llm_runs/EXPERIMENTS.md)), the step-by-step build runbook ([`AGENT_RUNBOOK.md`](llm_runs/AGENT_RUNBOOK.md)) and config templates |

## Open issues

- **No simulation has run the frozen setups yet.** The new LLM-free simulator
  starts from [`../simulator/runtime_rules.md`](../simulator/runtime_rules.md).
- **Nothing logs the proof-assembly rate** (per round, the fraction of agents
  whose active memory holds a complete proof). That is what task003-symmetric
  exists to measure.
- **One world.** Every number is tuned to `task_003`; balance in one world does
  not show results generalise.
