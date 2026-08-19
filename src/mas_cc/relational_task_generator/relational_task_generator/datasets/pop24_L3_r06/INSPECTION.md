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
- `f3` — Hira is east of Sora.

### Distractors

- `f4` — Vero is south of Faro.
- `f5` — Faro is west of Zavi.
- `f6` — Zavi is south of Havi.
- `f7` — Havi is southwest of Belo.

### Distribution of facts across agents

- `agent_001`: f6
- `agent_002`: f1, f7
- `agent_003`: —
- `agent_004`: f1
- `agent_005`: f3
- `agent_006`: f3
- `agent_007`: f2
- `agent_008`: f1
- `agent_009`: f2
- `agent_010`: f2
- `agent_011`: f1, f5
- `agent_012`: f1
- `agent_013`: f3, f4
- `agent_014`: f2
- `agent_015`: —
- `agent_016`: f3
- `agent_017`: —
- `agent_018`: f2
- `agent_019`: —
- `agent_020`: f3
- `agent_021`: f3
- `agent_022`: —
- `agent_023`: f2
- `agent_024`: f1

### Question

Where is Davi relative to Sora?

### Reasoning chain

1. Davi is northwest of Lira.
2. Lira is southwest of Hira.
3. Hira is east of Sora.

### Correct answer

`WEST` (option `B`)

---

## task_0002

### Task

Seed: `13295414370049309223`

### Supporting facts

- `f1` — Jeni is south of Zani.
- `f2` — Zani is southeast of Pelo.
- `f3` — Pelo is south of Meno.

### Distractors

- `f4` — Ceno is east of Viko.
- `f5` — Viko is northwest of Mira.
- `f6` — Mira is southwest of Tero.
- `f7` — Tero is south of Wali.

### Distribution of facts across agents

- `agent_001`: f1
- `agent_002`: f2
- `agent_003`: f1
- `agent_004`: f3, f4, f6
- `agent_005`: f2
- `agent_006`: f2
- `agent_007`: —
- `agent_008`: f3
- `agent_009`: f3
- `agent_010`: f2
- `agent_011`: —
- `agent_012`: f3
- `agent_013`: f1
- `agent_014`: f1, f7
- `agent_015`: f1
- `agent_016`: —
- `agent_017`: f3
- `agent_018`: f2, f5
- `agent_019`: —
- `agent_020`: f2
- `agent_021`: —
- `agent_022`: f1
- `agent_023`: —
- `agent_024`: f3

### Question

Where is Jeni relative to Meno?

### Reasoning chain

1. Jeni is south of Zani.
2. Zani is southeast of Pelo.
3. Pelo is south of Meno.

### Correct answer

`SOUTHEAST` (option `B`)

---

## task_0003

### Task

Seed: `9801262203822032224`

### Supporting facts

- `f1` — Kelo is northwest of Daro.
- `f2` — Daro is northwest of Maro.
- `f3` — Maro is north of Demi.

### Distractors

- `f4` — Meno is north of Yaro.
- `f5` — Yaro is northeast of Wira.
- `f6` — Wira is east of Savi.
- `f7` — Savi is south of Renu.

### Distribution of facts across agents

- `agent_001`: —
- `agent_002`: f1, f4, f6
- `agent_003`: f2
- `agent_004`: f1, f7
- `agent_005`: f1
- `agent_006`: f3
- `agent_007`: f3
- `agent_008`: f2
- `agent_009`: f3
- `agent_010`: f3
- `agent_011`: —
- `agent_012`: f1
- `agent_013`: f2
- `agent_014`: —
- `agent_015`: —
- `agent_016`: f3
- `agent_017`: f2
- `agent_018`: f1
- `agent_019`: f1
- `agent_020`: —
- `agent_021`: f3
- `agent_022`: f2
- `agent_023`: f2, f5
- `agent_024`: —

### Question

Where is Kelo relative to Demi?

### Reasoning chain

1. Kelo is northwest of Daro.
2. Daro is northwest of Maro.
3. Maro is north of Demi.

### Correct answer

`NORTHWEST` (option `A`)

---
