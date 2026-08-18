# Human-readable example inspection

These are deterministic renderings of the symbolic JSON tasks. Exact coordinates
remain available in each JSON file for auditing but are not repeated here; a
downstream game should normally expose agents only to the facts listed in their
`fact_ids`.

## task_0001

### Task

Seed: `18025959922836359799`

### Supporting facts

- `f1` — Davi is northwest of Lira.
- `f2` — Lira is southwest of Hira.

### Distractors

- `f3` — Sora is southwest of Vero.
- `f4` — Vero is north of Faro.
- `f5` — Faro is northwest of Zavi.
- `f6` — Zavi is east of Havi.

### Distribution of facts across agents

- `agent_001`: f2, f5
- `agent_002`: f4
- `agent_003`: —
- `agent_004`: f1
- `agent_005`: —
- `agent_006`: f1
- `agent_007`: f1
- `agent_008`: f2
- `agent_009`: f1
- `agent_010`: f1, f3
- `agent_011`: f2
- `agent_012`: f1, f6
- `agent_013`: —
- `agent_014`: —
- `agent_015`: f2
- `agent_016`: f2
- `agent_017`: —
- `agent_018`: —
- `agent_019`: —
- `agent_020`: —
- `agent_021`: f2
- `agent_022`: —
- `agent_023`: —
- `agent_024`: —

### Question

Where is Davi relative to Hira?

### Reasoning chain

1. Davi is northwest of Lira.
2. Lira is southwest of Hira.

### Correct answer

`WEST` (option `B`)

---

## task_0002

### Task

Seed: `13295414370049309223`

### Supporting facts

- `f1` — Jeni is south of Zani.
- `f2` — Zani is southwest of Pelo.

### Distractors

- `f3` — Meno is north of Ceno.
- `f4` — Ceno is east of Viko.
- `f5` — Viko is north of Mira.
- `f6` — Mira is southeast of Tero.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: f1
- `agent_003`: —
- `agent_004`: —
- `agent_005`: —
- `agent_006`: —
- `agent_007`: —
- `agent_008`: f2
- `agent_009`: f6
- `agent_010`: f2
- `agent_011`: f2, f5
- `agent_012`: f2
- `agent_013`: —
- `agent_014`: —
- `agent_015`: f1, f3
- `agent_016`: f1
- `agent_017`: f2
- `agent_018`: f1
- `agent_019`: f1
- `agent_020`: —
- `agent_021`: f2
- `agent_022`: —
- `agent_023`: f1
- `agent_024`: f4

### Question

Where is Jeni relative to Pelo?

### Reasoning chain

1. Jeni is south of Zani.
2. Zani is southwest of Pelo.

### Correct answer

`SOUTHWEST` (option `B`)

---

## task_0003

### Task

Seed: `9801262203822032224`

### Supporting facts

- `f1` — Kelo is east of Daro.
- `f2` — Daro is northwest of Maro.

### Distractors

- `f3` — Demi is northwest of Meno.
- `f4` — Meno is north of Yaro.
- `f5` — Yaro is north of Wira.
- `f6` — Wira is northeast of Savi.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: f2
- `agent_003`: f1
- `agent_004`: —
- `agent_005`: f1
- `agent_006`: —
- `agent_007`: f1, f3
- `agent_008`: f2
- `agent_009`: —
- `agent_010`: f2
- `agent_011`: f5
- `agent_012`: —
- `agent_013`: f1
- `agent_014`: —
- `agent_015`: —
- `agent_016`: f2, f4
- `agent_017`: f1
- `agent_018`: f1
- `agent_019`: —
- `agent_020`: f6
- `agent_021`: f2
- `agent_022`: —
- `agent_023`: f2
- `agent_024`: —

### Question

Where is Kelo relative to Maro?

### Reasoning chain

1. Kelo is east of Daro.
2. Daro is northwest of Maro.

### Correct answer

`NORTH` (option `A`)

---
