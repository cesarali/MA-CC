# Human-readable example inspection

These are deterministic renderings of the symbolic JSON tasks. Exact coordinates
remain available in each JSON file for auditing but are not repeated here; a
downstream game should normally expose agents only to the facts listed in their
`fact_ids`.

## task_0001

### Task

Seed: `9173884889605768521`

### Supporting facts

- `f1` — Teni is south of Meno.
- `f2` — Meno is east of Seni.
- `f3` — Seni is south of Renu.

### Distractors

- `f4` — Vani is southeast of Mira.
- `f5` — Mira is north of Zavi.

### Distribution of facts across agents

- `agent_001`: f2
- `agent_002`: f3
- `agent_003`: f1
- `agent_004`: f2
- `agent_005`: f3, f5
- `agent_006`: f3, f4
- `agent_007`: f1
- `agent_008`: —
- `agent_009`: —
- `agent_010`: f1
- `agent_011`: f2
- `agent_012`: —

### Question

Where is Teni relative to Renu?

### Reasoning chain

1. Teni is south of Meno.
2. Meno is east of Seni.
3. Seni is south of Renu.

### Correct answer

`SOUTHEAST` (option `B`)

---

## task_0002

### Task

Seed: `11511988360119458157`

### Supporting facts

- `f1` — Vero is northwest of Garo.
- `f2` — Garo is north of Peni.
- `f3` — Peni is east of Zani.

### Distractors

- `f4` — Mira is east of Davi.
- `f5` — Davi is northwest of Pelo.

### Distribution of facts across agents

- `agent_001`: f1
- `agent_002`: f3
- `agent_003`: —
- `agent_004`: —
- `agent_005`: f4
- `agent_006`: f1
- `agent_007`: f2
- `agent_008`: f1
- `agent_009`: f3
- `agent_010`: f2
- `agent_011`: f3
- `agent_012`: f2, f5

### Question

Where is Vero relative to Zani?

### Reasoning chain

1. Vero is northwest of Garo.
2. Garo is north of Peni.
3. Peni is east of Zani.

### Correct answer

`NORTH` (option `A`)

---
