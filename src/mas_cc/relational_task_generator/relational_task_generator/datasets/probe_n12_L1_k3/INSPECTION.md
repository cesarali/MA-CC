# Human-readable example inspection

These are deterministic renderings of the symbolic JSON tasks. Exact coordinates
remain available in each JSON file for auditing but are not repeated here; a
downstream game should normally expose agents only to the facts listed in their
`fact_ids`.

## task_0001

### Task

Seed: `7473446316198562528`

### Supporting facts

- `f1` — Nira is north of Faro.

### Distractors

- `f2` — Lumo is southeast of Daro.
- `f3` — Daro is east of Mira.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: —
- `agent_003`: —
- `agent_004`: —
- `agent_005`: f3
- `agent_006`: f1
- `agent_007`: f1
- `agent_008`: —
- `agent_009`: f1
- `agent_010`: f1
- `agent_011`: f2
- `agent_012`: —

### Question

Where is Nira relative to Faro?

### Reasoning chain

1. Nira is north of Faro.

### Correct answer

`NORTH` (option `A`)

---

## task_0002

### Task

Seed: `3070273083776054309`

### Supporting facts

- `f1` — Gavi is southeast of Wali.

### Distractors

- `f2` — Jeni is northeast of Naro.
- `f3` — Naro is west of Maro.

### Distribution of facts across agents

- `agent_001`: f3
- `agent_002`: f1
- `agent_003`: f1
- `agent_004`: —
- `agent_005`: f2
- `agent_006`: f1
- `agent_007`: —
- `agent_008`: —
- `agent_009`: —
- `agent_010`: —
- `agent_011`: f1
- `agent_012`: —

### Question

Where is Gavi relative to Wali?

### Reasoning chain

1. Gavi is southeast of Wali.

### Correct answer

`SOUTHEAST` (option `C`)

---

## task_0003

### Task

Seed: `3145825638826882744`

### Supporting facts

- `f1` — Zani is northwest of Wali.

### Distractors

- `f2` — Yaro is north of Pelo.
- `f3` — Pelo is west of Lira.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: —
- `agent_003`: —
- `agent_004`: f1
- `agent_005`: f1, f2
- `agent_006`: f1
- `agent_007`: f3
- `agent_008`: —
- `agent_009`: —
- `agent_010`: —
- `agent_011`: f1
- `agent_012`: —

### Question

Where is Zani relative to Wali?

### Reasoning chain

1. Zani is northwest of Wali.

### Correct answer

`NORTHWEST` (option `C`)

---
