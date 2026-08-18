# Relational Task Generator

A small, standalone, standard-library-only generator for frozen synthetic **multi-agent relational reasoning tasks**.

This folder is intentionally **not a Python package**. There is no `src/`, no `pyproject.toml`, no installation step, and no dependency on any existing repository.

It also **does not implement a multi-agent game**. It does not call LLMs, create prompts, run agents, implement voting dynamics, controllers, information-theoretic metrics, or interact with external APIs. Its only purpose is to generate and validate task instances that another system can consume later.

## Scientific purpose

The generator creates controlled tasks in which:

1. the symbolic ground-truth world is known exactly;
2. answering the query requires composing a known chain of facts;
3. the supporting facts can be distributed across multiple agents;
4. the initial fact allocation is explicitly recorded;
5. the union of the population's information contains the full supporting set;
6. optionally, no single agent initially possesses the complete supporting set.

The first task family is **2-D spatial relational reasoning**.

---

## Folder contents

```text
relational_task_generator/
├── README.md
├── generate_dataset.py
├── validate_dataset.py
├── generator.py
├── distribution.py
├── rendering.py
├── validation.py
├── examples/
│   ├── manifest.json
│   ├── INSPECTION.md
│   ├── task_0001.json
│   ├── task_0002.json
│   └── ...
└── tests/
    └── test_generator.py
```

All code uses only the Python standard library.

---

## 1. Symbolic spatial world

The internal world uses integer coordinates and the eight compass relations:

```text
NORTH      = ( 0,  1)
NORTHEAST  = ( 1,  1)
EAST       = ( 1,  0)
SOUTHEAST  = ( 1, -1)
SOUTH      = ( 0, -1)
SOUTHWEST  = (-1, -1)
WEST       = (-1,  0)
NORTHWEST  = (-1,  1)
```

A symbolic fact

```json
{
  "id": "f1",
  "subject": "Lumo",
  "relation": "NORTH",
  "object": "Kavi",
  "role": "supporting"
}
```

means exactly

```text
position(Lumo) - position(Kavi) = (0, 1)
```

The stored coordinates are therefore an auditable ground truth, not a probabilistic interpretation.

The answer to a query such as

```text
Where is Lumo relative to Tero?
```

is obtained from the exact net displacement between the two queried entities. A non-zero displacement is mapped to one of the eight compass sectors by the signs of its `x` and `y` components.

Generated tasks never use coincident query endpoints.

---

## 2. Reasoning chains and reasoning depth `L`

`reasoning_depth` is the number of supporting facts in the required graph path between the query subject and reference.

Version 1 supports:

```text
L = 1, 2, 3, 4
```

with `L = 2` as the default.

For `L = 2`, a task may contain:

```text
f1: Lumo is north of Kavi.
f2: Kavi is east of Tero.
```

The query is `Lumo` relative to `Tero`, and the supporting set is explicitly stored:

```json
"supporting_fact_ids": ["f1", "f2"]
```

The generator constructs the supporting component as a self-avoiding chain. Validation checks that the shortest supporting-fact path between the query entities has length exactly `L` and that the designated supporting facts imply the stored answer.

### Important edge case

`no_single_agent_solution=true` is impossible for `L=1`: if an agent receives the sole required fact, that agent necessarily has the complete supporting set. The generator therefore rejects that configuration with a clear error.

---

## 3. Supporting facts versus distractors

Each fact is marked as either:

```json
"role": "supporting"
```

or

```json
"role": "distractor"
```

In version 1, distractors are generated in a **separate disconnected spatial component**. This is intentionally conservative: it makes their irrelevance to the query chain exact and easy to audit.

Distractors are still internally consistent spatial facts and are assigned to agents, but they cannot create a shortcut, alternative proof, or contradiction involving the queried entities.

The number of distractor facts is configured with `--distractors`.

---

## 4. Multiple-choice answers

The default is:

```text
K = 3 options
```

The correct compass relation is stored independently from its randomized presentation position:

```json
"answer": {
  "correct_relation": "NORTHEAST",
  "options": [
    {"label": "A", "relation": "WEST"},
    {"label": "B", "relation": "NORTHEAST"},
    {"label": "C", "relation": "SOUTH"}
  ],
  "correct_option": "B"
}
```

`num_options` may be any value from 2 through 8. All options are distinct and exactly one equals `correct_relation`.

---

## 5. Distribution across agents

Every task stores the initial information allocation explicitly:

```json
"agents": {
  "agent_001": {
    "fact_ids": ["f1", "f4"]
  },
  "agent_002": {
    "fact_ids": ["f2"]
  }
}
```

### `support_redundancy`

`support_redundancy = r` means **each supporting fact is assigned to exactly `r` distinct agents**.

For example:

```text
population_size = 24
support_redundancy = 6
```

means every required supporting fact is initially known by exactly six agents.

### `distractor_redundancy`

Distractors are also assigned to the population. `distractor_redundancy` controls how many agents receive each distractor and defaults to 1.

### `no_single_agent_solution`

When enabled, the assignment algorithm enforces:

```text
for every agent a:
    supporting_fact_ids is NOT a subset of facts_known_by(a)
```

while still guaranteeing that the population union contains the complete supporting set.

For a task with `L` supporting facts, each agent is allowed to receive at most `L-1` of them. A requested configuration is rejected if it is mathematically impossible. The main feasibility condition is:

```text
L * support_redundancy <= N * (L - 1)
```

with `L >= 2` and `support_redundancy <= N`.

The assignment is deterministic given the task seed.

---

## 6. Symbolic data and deterministic language rendering

The symbolic representation is primary. Natural language is derived deterministically from it.

For example:

```json
{
  "subject": "Lumo",
  "relation": "NORTH",
  "object": "Kavi"
}
```

is rendered as:

```text
Lumo is north of Kavi.
```

The relation-to-phrase mapping is isolated in `rendering.py`. Version 1 has one canonical template per relation. Alternative deterministic paraphrase sets can be added later without changing the task schema or symbolic world generator.

No LLM is used for rendering or paraphrasing.

---

## 7. JSON schema used by generated tasks

Each task is self-contained. A typical file has the following structure:

```json
{
  "schema_version": "spatial_relational_task_v1",
  "task_id": "task_0001",
  "seed": 123,
  "generation": {
    "dataset_seed": 42,
    "task_index": 1,
    "population_size": 24,
    "reasoning_depth": 2,
    "support_redundancy": 6,
    "distractors": 4,
    "distractor_redundancy": 1,
    "num_options": 3,
    "no_single_agent_solution": true
  },
  "world": {
    "coordinate_convention": "...",
    "entities": [
      {
        "name": "Lumo",
        "coordinates": {"x": 1, "y": 1}
      }
    ],
    "facts": [
      {
        "id": "f1",
        "subject": "Lumo",
        "relation": "NORTH",
        "object": "Kavi",
        "role": "supporting"
      }
    ]
  },
  "query": {
    "subject": "Lumo",
    "reference": "Tero",
    "reasoning_depth": 2,
    "supporting_fact_ids": ["f1", "f2"]
  },
  "answer": {
    "correct_relation": "NORTHEAST",
    "options": [
      {"label": "A", "relation": "WEST"},
      {"label": "B", "relation": "NORTHEAST"},
      {"label": "C", "relation": "SOUTH"}
    ],
    "correct_option": "B"
  },
  "agents": {
    "agent_001": {
      "fact_ids": ["f1", "f4"]
    }
  },
  "rendered": {
    "question": "Where is Lumo relative to Tero?",
    "facts": {
      "f1": "Lumo is north of Kavi."
    },
    "reasoning_chain": [
      "Lumo is north of Kavi.",
      "Kavi is east of Tero."
    ]
  }
}
```

The exact generated examples in `examples/` should be treated as the reference for the concrete v1 schema.

---

## 8. Dataset manifest and reproducibility

Every generated dataset folder also contains `manifest.json`, recording:

- the dataset seed;
- all generation parameters;
- the ordered list of task files;
- a SHA-256 fingerprint for each canonical task JSON object;
- a SHA-256 fingerprint for the full ordered dataset.

Each task receives a deterministic child seed derived from:

```text
(dataset_seed, task_index)
```

Generation uses only local deterministic pseudorandomness. No wall-clock timestamps or external data are inserted into task files or the manifest, so the same command with the same seed and parameters produces byte-identical JSON and inspection files under the same Python implementation.

`validate_dataset.py` goes further: by default it **regenerates every task from the seed and configuration stored in that task and checks exact canonical equality**.

---

## 9. Automatic validation guarantees

Every candidate task is validated before it is accepted. Invalid candidates are discarded and regenerated rather than written as valid data.

The validator checks at least the following:

1. **World consistency**: every symbolic spatial constraint agrees with the stored exact coordinates, and the constraint graph is independently checked for contradictory cycles.
2. **Correct answer from the world**: the stored `correct_relation` matches the exact query displacement.
3. **Supporting facts imply the answer**: solving from only `supporting_fact_ids` yields the stored answer.
4. **Exact reasoning depth**: the shortest support-only graph path has length `L`.
5. **Distractor irrelevance**: v1 distractors use entities disjoint from the query-support component.
6. **Distinct options**: no repeated compass relation appears among answer options.
7. **Exactly one correct option**: one and only one option equals `correct_relation`.
8. **Population coverage**: every supporting fact is possessed by the population.
9. **Support redundancy**: every supporting fact has exactly the requested recipient count.
10. **No individual solution when requested**: no agent possesses all supporting facts.
11. **Agent references are valid**: every fact ID assigned to an agent exists.
12. **Distractor assignment**: every distractor is distributed with the requested redundancy.
13. **Deterministic rendering**: stored natural-language strings exactly match the canonical renderer.
14. **Seed reproducibility**: dataset validation regenerates tasks and compares canonical JSON exactly.
15. **Manifest integrity**: per-task and whole-dataset SHA-256 fingerprints are checked.

---

## 10. Generate a dataset

From inside this folder:

```bash
python generate_dataset.py \
    --num-tasks 100 \
    --population-size 24 \
    --reasoning-depth 2 \
    --support-redundancy 6 \
    --distractors 4 \
    --num-options 3 \
    --seed 42 \
    --no-single-agent-solution \
    --output generated/
```

Optional additional parameters include:

```text
--distractor-redundancy INT
--allow-single-agent-solution
--overwrite
```

Run:

```bash
python generate_dataset.py --help
```

for the complete CLI.

---

## 11. Validate an existing dataset

```bash
python validate_dataset.py generated/
```

A successful run prints:

```text
VALID: generated
```

By default, this includes the exact seed-regeneration check. For a faster structural/integrity check only:

```bash
python validate_dataset.py generated/ --skip-reproducibility-check
```

---

## 12. Run tests

From inside the folder:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- `L = 1, 2, 3, 4` generation;
- hidden-profile-like no-single-agent assignments;
- rejection of impossible `L=1` no-single-agent configurations;
- deterministic repeated generation;
- complete written-dataset reproducibility;
- independent dataset validation;
- detection of invalid agent fact references.

---

## 13. Included example dataset

The checked-in `examples/` folder contains approximately the requested inspection configuration:

```text
num_tasks = 20
N = 24
L = 2
K = 3
support_redundancy = 6
distractors = 4
distractor_redundancy = 1
no_single_agent_solution = true
seed = 42
```

`examples/INSPECTION.md` renders several tasks in a compact human-readable form with:

- task ID and seed;
- supporting facts;
- distractors;
- complete agent-to-fact distribution;
- question;
- reasoning chain;
- correct answer.

The JSON files remain the authoritative symbolic records.

---

## 14. Explicit non-goals

This folder does **not** contain or assume:

- LLM calls;
- external API calls;
- multi-agent interaction loops;
- q-voter or other opinion dynamics;
- controllers;
- agent prompts;
- mutual information;
- transfer entropy;
- any existing repository layout;
- third-party frameworks.

It is deliberately a transparent data generator and validator only.
