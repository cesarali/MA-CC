# HiddenBench paraphrase and factorization pipeline

## Purpose and scope

This document explains how the repository prepared alternative HiddenBench
private-information assignments for populations larger than the original three
or four participants. It describes the behavior of the actual scripts, the
meaning of the generated files, and the status of the existing annotation run.

The two transformations are:

| Method | What changes | Why use it? |
| --- | --- | --- |
| `paraphrased_replication` | Different agents receive differently worded versions of the same complete hidden fact. | Increase population size without every holder using identical text. |
| `factorized_evidence` | One hidden fact is divided into 2–4 smaller semantic components and the components are distributed across agents. | Study whether information can be reconstructed when it is more finely distributed. |

Only `hidden_information` is transformed. The scenario, shared information,
answer options, and correct-answer label are retained from the canonical task.
The correct answer is available to the annotation verifier for leakage checks,
but generated private observations are forbidden from stating or hinting at it.

This work is implemented by:

- `scripts/generate_semantic_annotations.py`, which generates and verifies the
  reusable annotation pools; and
- `scripts/prepare_hiddenbench.py`, which assigns accepted annotations to a
  chosen number of agents and writes a scaled dataset.

All paths in the commands below are relative to
`scripts/local_llms/hiddenbench_population_pipeline/`.

## The common generation process

For each task, the program processes every original hidden-information item as
a separate **evidence type**. Its list index is the evidence-type ID.

```text
canonical hidden fact
        |
        v
LLM generator proposes candidates
        |
        v
LLM verifier audits candidates
        |
        v
accepted annotation checkpoint
        |
        v
population builder assigns annotations to agents
```

The generator receives the task description, shared facts, all hidden facts,
answer options, and correct answer. Supplying the whole context lets it avoid
accidentally importing another participant's fact or leaking the answer.

The verifier receives the same context, the target source fact, and the
generated candidates. It returns structured JSON verdicts. Only candidates
marked `acceptable: true` enter the usable pool.

Generation proceeds one task at a time and writes an evidence-level checkpoint
after each unit. `--resume` keeps accepted paraphrases and completed
factorizations instead of generating them again. Every API call and its token,
model, response, and verification metadata is also recorded in
`annotations/generation_audit.jsonl` and inside the annotation JSON.

After a complete traversal, the script adds `status: "frozen"`. The population
builder deliberately refuses annotation files without this marker. This avoids
silently constructing an experimental condition from a half-finished pool.

## Paraphrasing

### What a paraphrase means here

A paraphrase is a standalone restatement of one **complete hidden fact**. It is
not a shortened clue and it is not a piece of the fact. The meaning relevant to
the decision should be unchanged.

For example, an original hidden fact in task 1 is:

> The supply truck headed to the village from East Town was stuck in the tunnel.

One accepted variant in the existing pool is:

> The supply truck traveling from East Town to the village was stuck in the tunnel.

Both observations convey the same origin, destination, vehicle, and obstruction.

### Generator requirements

For every evidence type, the generator is asked for distinct variants using
lexical, syntactic, referential, or discourse changes. Each candidate must:

1. preserve every answer-relevant proposition in the source;
2. add no entity, relation, time, quantity, certainty, cause, or consequence;
3. neither state nor hint at the correct answer;
4. import nothing from another hidden fact;
5. differ enough in wording or syntax to add linguistic diversity; and
6. make sense on its own.

The current configuration targets 10 accepted variants per evidence type. A
generation batch requests 12 candidates by default, and an evidence type may
use up to 10 generation-and-verification rounds. Candidates are normalized and
deduplicated against the original text and already accepted variants before
verification.

### Verifier requirements

The verifier tests every candidate for:

- entailment by the source;
- reverse entailment, meaning the candidate has not dropped source content;
- absence of added answer-relevant information;
- absence of correct-answer leakage; and
- absence of overlap with other hidden evidence types.

A candidate is acceptable only when both entailment directions are true and
all three contamination flags are false. Accepted variants receive stable IDs
such as `0-000`, where the first number is the evidence type.

### Building a paraphrased population

The population builder first balances evidence types across agents. It then
shuffles each type's accepted paraphrase pool with a deterministic seed and
assigns one full variant to each agent of that type.

By default, a variant cannot be reused within a task population. If a type has
10 variants, it can therefore support at most 10 holders without reuse. A task
with four balanced evidence types can support roughly 40 agents, while a task
with three types supports roughly 30. The `--allow-paraphrase-reuse` option can
go beyond this limit, but repeated wording then reappears in the population.

Each generated agent record retains:

- the evidence type;
- the chosen variant ID and paraphrased private text;
- the original hidden-information index and source text; and
- `transformation: "validated_paraphrase"`.

## Factorization

### What factorization means here

Factorization is **semantic decomposition**, not automatic sentence splitting.
One hidden fact is divided into 2–4 propositions that are jointly sufficient to
recover the complete fact but individually insufficient.

For the same task-1 truck fact, the selected existing factorization contains:

1. “A supply truck existed.”
2. “It was destined for the village.”
3. “It had departed from East Town.”
4. “It became stuck in a tunnel.”

The components separate the entity, destination, origin, and incident. They
only reproduce the original information after they are combined.

Components may be labeled with the roles `entity`, `relation`, `constraint`,
`observation`, or `bridge`. These labels describe the component's semantic
function; they do not affect voting directly.

### Generator requirements

The generator proposes up to four alternative factorizations for each evidence
type, with 2–4 components per alternative. A valid proposal must:

1. jointly reconstruct the complete source fact;
2. ensure no single component is equivalent to the complete source;
3. add nothing not entailed by the source or scenario;
4. avoid revealing the correct answer;
5. represent a meaningful semantic or inferential split; and
6. mark the fact non-factorizable when no defensible split exists.

Each proposal also includes a plain-language reconstruction rule and an
explanation of why the split is meaningful.

### Verifier and selection requirements

The verifier scores each candidate on whether:

- its components jointly reconstruct the source;
- every component is supported by the source;
- each component is individually insufficient;
- there is no answer leakage;
- no other hidden fact was imported; and
- the split is meaningful rather than arbitrary.

Any failed requirement makes the proposal unacceptable. Among acceptable
alternatives, the program selects the one with the highest verifier-provided
`quality_score`. Component IDs encode the evidence type, alternative, and
component, for example `0-2-3`.

There is one implementation caveat: when the **generator itself** says a fact
is non-factorizable, or returns no structurally valid 2–4 component candidate,
the current script records that result as complete without a separate verifier
decision. Such a fact causes its entire task to be excluded later from the
factorized experimental condition.

### Building a factorized population

The selected components from all evidence types are pooled and assigned with a
deterministic seeded allocator:

- every component is assigned at least once;
- when agents are fewer than components, some agents receive multiple pieces;
- the allocator tries not to give one agent two components of the same evidence
  type;
- when agents outnumber components, components are replicated so no agent is
  empty; and
- diagnostics identify any agent who nevertheless receives every component of
  one evidence type.

An agent can therefore hold components from several original evidence types.
Its record contains `evidence_types`, component IDs, component texts, source
indices, and `transformation: "factor_components"`.

The allocation diagnostic is important: if one agent receives every component
of an evidence type, that agent has effectively reconstructed the original
hidden fact without discussion. The resulting dataset warns about this and
should undergo the information-sufficiency audit before use.

## Difference from exact replication

| Property | Exact replication | Paraphrased replication | Factorized evidence |
| --- | --- | --- | --- |
| Meaning held by one agent | One complete original hidden fact | One complete hidden fact in new wording | One or more partial components |
| Text identical across holders | Yes | Normally no | Components may repeat only when population exceeds component count |
| Requires LLM annotations | No | Yes | Yes |
| Requires a frozen pool | No | Yes | Yes |
| Can increase informational fragmentation | No | No | Yes |

Exact replication is the completed `N_32.json` currently available. It repeats
the original hidden facts across agents but does not modify their text.

## Existing files and current status

| File | Current state |
| --- | --- |
| `annotations/paraphrases.json` | 42 tasks complete; task 43 partially complete; tasks 44–65 absent; not frozen |
| `annotations/factorizations.json` | Tasks 1–42 complete; tasks 43–65 absent; not frozen |
| `annotations/generation_audit.jsonl` | Detailed generation and verification audit trail |
| `annotations/gpt5mini_full_run.log` | Console log from the interrupted run |
| `data/hiddenbench/scaled/exact_replication/N_32.json` | Complete 65-task, 32-agent exact-replication dataset |
| `data/hiddenbench/scaled/paraphrased_replication/N_32.json` | Not built |
| `data/hiddenbench/scaled/factorized_evidence/N_32.json` | Not built |

The existing annotation files record `microsoft/gpt-5-mini` as both generator
and verifier. Although the pipeline supports separately configured generator
and verifier models, this particular partial run used the same model for both.
It should therefore be described as a two-pass self-verification process, not
independent cross-model verification.

The run stopped on task 43, evidence type 1, after 10 paraphrase rounds produced
zero accepted variants for that evidence item. Because the run did not traverse
all selected tasks, neither annotation file received `status: "frozen"`, and
the population builder correctly refuses to use them.

These data and annotation directories are Git-ignored. They exist in the local
workspace but are not normal tracked repository files.

## Commands used by the workflow

Generate or resume both annotation types:

```bash
python scripts/generate_semantic_annotations.py \
  --input data/hiddenbench/canonical/tasks.json \
  --output-dir annotations \
  --mode both \
  --paraphrases-per-type 10 \
  --factorization-alternatives 4 \
  --max-components 4 \
  --resume \
  --require-complete
```

After—and only after—the annotation files are complete and frozen, build the
two 32-agent population datasets:

```bash
python scripts/prepare_hiddenbench.py \
  --agents 32 \
  --method paraphrased_replication \
  --annotations annotations/paraphrases.json \
  --data-root data/hiddenbench

python scripts/prepare_hiddenbench.py \
  --agents 32 \
  --method factorized_evidence \
  --annotations annotations/factorizations.json \
  --data-root data/hiddenbench
```

Finally, the transformed datasets should be checked with
`scripts/run_information_sufficiency_audit.py`. Generation-time verification
checks semantic preservation and leakage, while the later sufficiency audit
tests the experimental property that partial individual views remain difficult
and the pooled view remains solvable. These are different checks, and both are
needed before treating a transformed population as validated research data.
