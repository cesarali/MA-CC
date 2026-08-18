# Human-readable example inspection

These are deterministic renderings of the symbolic JSON tasks. Exact coordinates
remain available in each JSON file for auditing but are not repeated here; a
downstream game should normally expose agents only to the facts listed in their
`fact_ids`.

## task_0001

### Task

Seed: `2215413865579214693`

### Supporting facts

- `f1` — Bavi is northeast of Zora.
- `f2` — Zora is northwest of Ralo.

### Distractors

- `f3` — Javi is south of Vela.
- `f4` — Vela is northeast of Cali.
- `f5` — Cali is east of Savi.
- `f6` — Savi is northwest of Maro.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: —
- `agent_003`: f1
- `agent_004`: f1
- `agent_005`: f2, f4
- `agent_006`: f2
- `agent_007`: —
- `agent_008`: —
- `agent_009`: f1
- `agent_010`: f3
- `agent_011`: —
- `agent_012`: —
- `agent_013`: f1
- `agent_014`: f2
- `agent_015`: f2
- `agent_016`: —
- `agent_017`: —
- `agent_018`: f2
- `agent_019`: f2
- `agent_020`: f1
- `agent_021`: f1, f5
- `agent_022`: f6
- `agent_023`: —
- `agent_024`: —

### Question

Where is Bavi relative to Ralo?

### Reasoning chain

1. Bavi is northeast of Zora.
2. Zora is northwest of Ralo.

### Correct answer

`NORTH` (option `C`)

---

## task_0002

### Task

Seed: `3331710926435733375`

### Supporting facts

- `f1` — Fira is west of Kelo.
- `f2` — Kelo is southwest of Selo.

### Distractors

- `f3` — Tavi is northeast of Teni.
- `f4` — Teni is southeast of Gavi.
- `f5` — Gavi is north of Faro.
- `f6` — Faro is southwest of Yaro.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: f2
- `agent_003`: —
- `agent_004`: —
- `agent_005`: f6
- `agent_006`: f1
- `agent_007`: —
- `agent_008`: f1, f5
- `agent_009`: f2, f3, f4
- `agent_010`: —
- `agent_011`: f1
- `agent_012`: f2
- `agent_013`: f1
- `agent_014`: f1
- `agent_015`: —
- `agent_016`: f1
- `agent_017`: f2
- `agent_018`: —
- `agent_019`: —
- `agent_020`: f2
- `agent_021`: —
- `agent_022`: —
- `agent_023`: —
- `agent_024`: f2

### Question

Where is Fira relative to Selo?

### Reasoning chain

1. Fira is west of Kelo.
2. Kelo is southwest of Selo.

### Correct answer

`SOUTHWEST` (option `C`)

---

## task_0003

### Task

Seed: `9006565506066611614`

### Supporting facts

- `f1` — Viko is southeast of Lumo.
- `f2` — Lumo is southeast of Tavi.

### Distractors

- `f3` — Naro is east of Pira.
- `f4` — Pira is south of Kavi.
- `f5` — Kavi is east of Ceno.
- `f6` — Ceno is east of Jora.

### Distribution of facts across agents

- `agent_001`: f2
- `agent_002`: f2
- `agent_003`: f2
- `agent_004`: f1
- `agent_005`: f1
- `agent_006`: —
- `agent_007`: f2
- `agent_008`: —
- `agent_009`: —
- `agent_010`: —
- `agent_011`: f1
- `agent_012`: —
- `agent_013`: —
- `agent_014`: f3, f4
- `agent_015`: —
- `agent_016`: f1
- `agent_017`: f1
- `agent_018`: —
- `agent_019`: f6
- `agent_020`: —
- `agent_021`: f1
- `agent_022`: f2, f5
- `agent_023`: —
- `agent_024`: f2

### Question

Where is Viko relative to Tavi?

### Reasoning chain

1. Viko is southeast of Lumo.
2. Lumo is southeast of Tavi.

### Correct answer

`SOUTHEAST` (option `C`)

---
