# 15_agent_simulation: the clean 15-agent campaign

**Start here for new simulations (from 29 October 2026).** This directory holds the setups, and later the runs and
analyses, of a clean rerun of the two main phenomena with **15 agents**:
- the **reading reversal**: reading more helps a covert controller at short memory and protects the swarm at long
  memory;
- **persistence**: the controller's effect outlasts its posting.

## Why 15 agents

The LLM version of these experiments (the swarm-control repository, Design R) can afford only 15 agents and up to 100
paths per cell. This campaign gives the matched Bayesian-reasoner baseline (Simulation 2 mechanics, 1,000 paths per
cell). It is also an occasion to rerun everything in one clean, documented place.

## Layout

```
15_agent_simulation/
├── README.md              this file
├── build_setups.py        builds all four setups (deterministic; rebuild rather than edit)
├── make_readmes.py        writes each setup's README from its JSON files
├── build_summary.json     key properties of the four setups
├── task003/
│   ├── README.md
│   ├── symmetric/         agents.json, controller_pool_A0.json, controller_pool_A2.json, README.md
│   └── no_solution/       agents.json, controller_pool.json, README.md
└── object_placement/
    ├── README.md
    ├── symmetric/         (same files as task003/symmetric)
    └── no_solution/       (same files as task003/no_solution)
```

## The four setups (all: 15 agents, one distinct fact each)

| setup | agents' facts together | controller pools |
|---|---|---|
| `*/symmetric` | prove A0 (P(A0) = 1); no single fact or pair does | two, unchanged from the 24-agent setups: A0 pool (truth arm), A2 pool (false arm) |
| `*/no_solution` | approximately uniform (1/3, 1/3, 1/3) | one 12-fact pool for both targets, also approximately uniform; agents and pool together prove nothing |

Each setup's `README.md` lists its exact properties. The symmetric agents are a 15-fact subset of the original setups'
18 distinct agent facts. The no-solution sets are new: the old 24-agent no-proof setups combined to (0.5, 0, 0.5), not
uniform.

## The campaign (decided 29 October 2026; pre-registered in `00_preregistration.md`)

- **Arms:** false controller (target A2) and its paired silent arm; no truth arm.
- **Agents:** 15, W = 30 (keeps board turnover N/W = 0.5, as with 24 agents and W = 48).
- **q:** 1, 2, 3, 4, 6, 8, 12, 16 (log scale).
- **ρ:** 0.018, 0.135, 0.37, 0.61, 0.78, 0.88, 0.94, 1. These are spaced evenly in mean memory lifetime
  τ = −1/ln ρ ≈ 0.25, 0.5, 1, 2, 4, 8, 16, ∞, because the crossover moves to short memory at large q and sits at long
  memory in the no-proof setups.
- **Reading:** standard (`posts`) and deduplicated (`distinct_facts`; stage 2 of `../theory_claude/dedup_defence/`).
- **Controller:** rate λ_c = 1, q_c = 8, budget 30, vote gate 0.75, proof stop; it stops at t = 30.
- **Run length and size:** to t = 130; 1,000 paths per cell; 512 cells per task, 1,024 in all.

**How to run** (from the repository root; about 5–8 hours on 12 workers, task003 first):
`bash analysis/15_agent_simulation/run_overnight.sh`. Outputs go to `results/<date>_sim2_15agents_<task>/`
(gitignored, about 25 GB per task). Logs go to `logs/`. Reports go to `reports/YYYY-MM-DD_<topic>/`.

**Files:**
- `configs/`: the two study configs (one per world).
- `setups.json`: maps setup names (`task003_symmetric_15`, `task003_nosolution_uniform_15`, `op_symmetric_15`,
  `op_nosolution_uniform_15`) to the folders here.

**Code added for this campaign (29 Oct 2026; backward-compatible; all 77 simulator tests pass):**
- `llmfree_core/setups.py` reads `setups.json` and takes the pool layout from the files present;
- `sim2/checks.py` checks the four new setups, with uniform no-solution agents;
- `sim2/simulation_2.py` has the `read_mode` option;
- the runner has `read_mode` as a grid factor and `record.agent_snapshot_times`.

## Older simulations stay where they are

Earlier work is organised by project, for historical reasons. It is **not** being reorganised, to avoid breaking paths,
manifests and provenance:
- `task003_llm_free/`: task003, 24 agents. Setups in `experimental_setup/`, simulators in `simulation_1/` and
  `simulation_2/`, results in `results/`.
- `object_placements_llm_free/`: Object Placements, 24 agents.
- `theory_claude/`, `theory_chatgpt/`, `theory_blackboard/`: the theory work (closed 28 October 2026; the deduplicated
  reading defence continues in `theory_claude/dedup_defence/`).

The frozen 24-agent setups are never edited. This directory never writes into them.
