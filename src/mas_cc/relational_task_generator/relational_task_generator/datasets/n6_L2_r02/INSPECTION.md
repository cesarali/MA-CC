# Human-readable example inspection

These are deterministic renderings of the symbolic JSON tasks. Exact coordinates
remain available in each JSON file for auditing but are not repeated here; a
downstream game should normally expose agents only to the facts listed in their
`fact_ids`.

## task_0001

### Task

Seed: `874272090446871458`

### Supporting facts

- `f1` — Zora is south of Maro.
- `f2` — Maro is west of Feni.

### Distractors

- `f3` — Garo is north of Belo.

### Distribution of facts across agents

- `agent_001`: f1
- `agent_002`: f2
- `agent_003`: f2
- `agent_004`: f3
- `agent_005`: f1
- `agent_006`: —

### Question

Where is Zora relative to Feni?

### Reasoning chain

1. Zora is south of Maro.
2. Maro is west of Feni.

### Correct answer

`SOUTHWEST` (option `A`)

---
