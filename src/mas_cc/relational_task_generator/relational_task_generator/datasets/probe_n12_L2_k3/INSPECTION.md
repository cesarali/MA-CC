# Human-readable example inspection

These are deterministic renderings of the symbolic JSON tasks. Exact coordinates
remain available in each JSON file for auditing but are not repeated here; a
downstream game should normally expose agents only to the facts listed in their
`fact_ids`.

## task_0001

### Task

Seed: `7473446316198562528`

### Supporting facts

- `f1` — Nira is north of Belo.
- `f2` — Belo is southeast of Daro.

### Distractors

- `f3` — Lumo is northwest of Navi.
- `f4` — Navi is west of Tero.

### Distribution of facts across agents

- `agent_001`: f1
- `agent_002`: —
- `agent_003`: f2
- `agent_004`: f3
- `agent_005`: f1, f4
- `agent_006`: f1, f2
- `agent_007`: —
- `agent_008`: —
- `agent_009`: f2
- `agent_010`: f1, f2
- `agent_011`: —
- `agent_012`: —

### Question

Where is Nira relative to Daro?

### Reasoning chain

1. Nira is north of Belo.
2. Belo is southeast of Daro.

### Correct answer

`EAST` (option `A`)

---

## task_0002

### Task

Seed: `3070273083776054309`

### Supporting facts

- `f1` — Gavi is northwest of Feni.
- `f2` — Feni is west of Meno.

### Distractors

- `f3` — Nori is southeast of Peni.
- `f4` — Peni is northeast of Kelo.

### Distribution of facts across agents

- `agent_001`: f2
- `agent_002`: f1
- `agent_003`: f1, f2
- `agent_004`: —
- `agent_005`: f2
- `agent_006`: f1
- `agent_007`: f4
- `agent_008`: f1
- `agent_009`: —
- `agent_010`: f2
- `agent_011`: f3
- `agent_012`: —

### Question

Where is Gavi relative to Meno?

### Reasoning chain

1. Gavi is northwest of Feni.
2. Feni is west of Meno.

### Correct answer

`NORTHWEST` (option `B`)

---

## task_0003

### Task

Seed: `3145825638826882744`

### Supporting facts

- `f1` — Zani is west of Jeni.
- `f2` — Jeni is southeast of Seni.

### Distractors

- `f3` — Meno is north of Savi.
- `f4` — Savi is west of Bavi.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: —
- `agent_003`: f2
- `agent_004`: f1
- `agent_005`: f1, f2
- `agent_006`: f1
- `agent_007`: f2
- `agent_008`: —
- `agent_009`: —
- `agent_010`: f4
- `agent_011`: f1, f2
- `agent_012`: f3

### Question

Where is Zani relative to Seni?

### Reasoning chain

1. Zani is west of Jeni.
2. Jeni is southeast of Seni.

### Correct answer

`SOUTH` (option `C`)

---
