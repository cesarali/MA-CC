# Human-readable example inspection

These are deterministic renderings of the symbolic JSON tasks. Exact coordinates
remain available in each JSON file for auditing but are not repeated here; a
downstream game should normally expose agents only to the facts listed in their
`fact_ids`.

## task_0001

### Task

Seed: `18025959922836359799`

### Supporting facts

- `f1` — Davi is south of Lira.
- `f2` — Lira is northwest of Hira.

### Distractors

- `f3` — Sora is southwest of Vero.
- `f4` — Vero is east of Faro.

### Distribution of facts across agents

- `agent_001`: f1
- `agent_002`: f4
- `agent_003`: f1
- `agent_004`: f2
- `agent_005`: f2
- `agent_006`: f2
- `agent_007`: f3
- `agent_008`: f1
- `agent_009`: f1
- `agent_010`: —
- `agent_011`: f2
- `agent_012`: —

### Question

Where is Davi relative to Hira?

### Reasoning chain

1. Davi is south of Lira.
2. Lira is northwest of Hira.

### Correct answer

`WEST` (option `C`)

---

## task_0002

### Task

Seed: `13295414370049309223`

### Supporting facts

- `f1` — Jeni is northeast of Zani.
- `f2` — Zani is north of Pelo.

### Distractors

- `f3` — Meno is south of Ceno.
- `f4` — Ceno is southwest of Viko.

### Distribution of facts across agents

- `agent_001`: f2
- `agent_002`: f2
- `agent_003`: f1
- `agent_004`: f3, f4
- `agent_005`: f2
- `agent_006`: —
- `agent_007`: f2
- `agent_008`: —
- `agent_009`: f1
- `agent_010`: f1
- `agent_011`: f1
- `agent_012`: —

### Question

Where is Jeni relative to Pelo?

### Reasoning chain

1. Jeni is northeast of Zani.
2. Zani is north of Pelo.

### Correct answer

`NORTHEAST` (option `B`)

---

## task_0003

### Task

Seed: `9801262203822032224`

### Supporting facts

- `f1` — Kelo is south of Daro.
- `f2` — Daro is east of Maro.

### Distractors

- `f3` — Demi is northwest of Meno.
- `f4` — Meno is northwest of Yaro.

### Distribution of facts across agents

- `agent_001`: f1
- `agent_002`: —
- `agent_003`: —
- `agent_004`: f2
- `agent_005`: f2, f4
- `agent_006`: —
- `agent_007`: f2
- `agent_008`: f1
- `agent_009`: f2
- `agent_010`: f1
- `agent_011`: f1
- `agent_012`: f3

### Question

Where is Kelo relative to Maro?

### Reasoning chain

1. Kelo is south of Daro.
2. Daro is east of Maro.

### Correct answer

`SOUTHEAST` (option `C`)

---
