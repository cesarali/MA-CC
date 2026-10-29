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

## Planned campaign (decided 29 October 2026; to be pre-registered before running)

- W = 30 (keeps board turnover N/W = 0.5, as with 24 agents and W = 48).
- q ∈ {3, 6, 9, 12}; ρ ∈ {0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1}.
- Continuation to t = 130 after the controller stops at t = 30; 1,000 paths per cell.
- Deduplicated reading as an extra factor: stage 2 of `../theory_claude/dedup_defence/`.
- The LLM's 100-path runs on a subset of cells.

**Code note.** The existing loader (`task003_llm_free/llmfree_core/setups.py`) recognises setups by their old folder
names (`task003_symmetric`, `op_nosolution`, ...), and Simulation 2 uses the setup name for its proof-stop shortcut.
The campaign runner must therefore map these folders to the right pool layout, with a small backward-compatible loader
extension, before any run. Older setups must keep loading exactly as before.

## Older simulations stay where they are

Earlier work is organised by project, for historical reasons. It is **not** being reorganised, to avoid breaking paths,
manifests and provenance:
- `task003_llm_free/`: task003, 24 agents. Setups in `experimental_setup/`, simulators in `simulation_1/` and
  `simulation_2/`, results in `results/`.
- `object_placements_llm_free/`: Object Placements, 24 agents.
- `theory_claude/`, `theory_chatgpt/`, `theory_blackboard/`: the theory work (closed 28 October 2026; the deduplicated
  reading defence continues in `theory_claude/dedup_defence/`).

The frozen 24-agent setups are never edited. This directory never writes into them.
