# Handoff: Native MuSR-Style Team Allocation Generator for MAS-CC

## Status / decision

**Stop the previous adapter-based implementation.**

The earlier `musr_team_allocation_generator` that expects an already-generated MuSR `team_allocation.json` should be treated as obsolete for this direction. It can be deleted or ignored.

The new goal is:

> Port/adapt the minimum useful parts of MuSR Team Allocation generation into MAS-CC, replace MuSR's model wrapper with the existing MAS-CC provider abstraction, generate richer MuSR-style evidence directly inside our codebase, distribute that evidence over a population of agents, validate the distributed information structure, and save frozen MAS-CC task JSON files.

The runtime experiment must **not** depend on MuSR or perform task-generation calls.

---

## 1. Scientific objective

We want a reasoning task richer than the current spatial relational task while preserving the existing population/control experiment.

The task should have:

- exact programmatic ground truth;
- a small discrete answer space, initially `K = 3`;
- natural-language evidence requiring nontrivial inference;
- partial but meaningful information for each agent;
- no single agent initially able to determine the answer;
- enough independently generated evidence for populations such as `N = 24`;
- frozen task files so all experimental conditions can reuse exactly the same worlds.

The intended generation pipeline is:

```text
exact latent Team Allocation problem
        ↓
latent skill/cooperation facts
        ↓
MuSR-style reasoning-tree expansion via MAS-CC LLM provider
        ↓
multiple independent natural-language evidence branches
        ↓
coherent evidence bundles
        ↓
distribution over N agents
        ↓
structural + exact validation
        ↓
frozen MAS-CC JSON dataset
```

The LLM is used only to produce the linguistic reasoning/evidence layer.

**The LLM must never determine the gold answer.**

---

## 2. Upstream MuSR basis

Use the public MuSR repository as the basis:

- Repository: `Zayne-sprague/MuSR`
- Pin the upstream commit used.
- MuSR is MIT licensed.
- Preserve the upstream MIT license and clear attribution for copied/adapted files.

Useful upstream components include:

```text
src/logic_tree/tree.py
src/dataset_builder.py
src/dataset_types/team_allocation.py
musr_dataset_scripts/create_team_allocation.py
```

Do **not** vendor the whole repository.

Only port/adapt the minimum code needed for:

1. Team Allocation latent problem construction;
2. latent fact creation;
3. reasoning-tree representation;
4. recursive entailment-tree expansion;
5. relevant validators/prompts.

Where copying code is unnecessary, reimplement the small interface cleanly while preserving attribution to the MuSR design.

---


## 2.1 Upstream checkout and preferred generation model

The implementation agent should explicitly obtain MuSR rather than assuming it is already present.

Use a temporary/external checkout for reference, then copy/adapt only the required components into MAS-CC:

```bash
git clone https://github.com/Zayne-sprague/MuSR.git
cd MuSR

# Record the exact upstream revision used.
git rev-parse HEAD
```

Prefer pinning the checkout to the exact commit recorded in MAS-CC attribution/generation metadata before adapting code.

For the natural-language reasoning-tree/evidence generation calls, it is acceptable to use:

```text
gpt-5.6-terra
```

through the existing MAS-CC provider abstraction.

Do not add a direct OpenAI/MuSR model dependency merely for this generator. The point is to reuse the MuSR generation logic while routing all inference through MAS-CC's normal provider stack.

This generation process is expected to be relatively modest in provider usage. For an initial development/smoke dataset, think in terms of **tens of LLM calls rather than thousands**. Do not prematurely optimize the implementation around massive batching. First make the generation/validation pipeline correct and auditable.

Because multiple branches may be generated per latent fact, actual call count depends on the implementation strategy. Where safe, generate several independent branches in a single structured LLM request rather than issuing one request per individual evidence sentence.


## 3. Proposed package layout

Create:

```text
src/mas_cc/musr_team_allocation_generator/
├── __init__.py
├── README.md
├── attribution.md
├── schemas.py
├── latent_problem.py
├── reasoning_tree.py
├── provider_adapter.py
├── prompts.py
├── evidence_generation.py
├── distribute.py
├── validate.py
├── generate.py
└── cli.py
```

Optionally keep copied/adapted upstream code isolated:

```text
src/mas_cc/musr_team_allocation_generator/vendor/musr/
```

if that makes attribution and future maintenance cleaner.

Do not modify `relational_task_generator`.

---

## 4. Exact latent Team Allocation problem

Reuse/adapt MuSR's Team Allocation idea:

- 3 people;
- 2 tasks / skills;
- one person assigned to task 1;
- the remaining two assigned jointly to task 2;
- individual skill values;
- pairwise cooperation values;
- score of each candidate allocation;
- exactly one best allocation.

Conceptually:

```text
score(allocation)
    = skill contribution
    + skill contribution
    + teamwork contribution
```

The exact data structure should retain:

```python
people
tasks
skills
skill_matrix
cooperation_matrix
candidate_allocations
candidate_scores
gold_index
gold_allocation
margin_to_second_best
```

All answer correctness comes from this symbolic/numeric layer.

Reject any generated world without a unique optimum.

---

## 5. Important extension beyond vanilla MuSR

The released MuSR Team Allocation instances contain only a limited number of underlying facts / reasoning branches. That is too small for a population such as `N = 24` if we merely distribute existing leaves.

Therefore **do not simply reproduce one reasoning tree per latent fact**.

Add a configurable parameter such as:

```yaml
branches_per_latent_fact: 3
tree_depth: 2
```

For every important latent fact, generate several **independent semantic evidence branches**.

Example hidden fact:

```text
Alice has high programming skill.
```

Possible independently generated branches:

```text
Branch A
- Alice built several software tools in a previous project.
- Her team relied on her to resolve difficult implementation bugs.

Branch B
- Alice recently entered a programming competition.
- She reached the final round.

Branch C
- Alice maintained a Python data-processing pipeline.
- Colleagues repeatedly asked her to review technical changes.
```

The exact latent fact is hidden from the experimental agents.

The branches should support inference toward that latent fact without literally stating it.

Do the same for cooperation facts.

For `N = 24`, target enough material for roughly **2–4 coherent evidence bundles per agent**, without requiring extreme replication.

A reasonable initial target is around `48–96` evidence snippets/cards per world, but make this configurable.

---

## 6. Use MAS-CC providers, not MuSR's OpenAI wrapper

MuSR's original code calls its own `OpenAIModel`.

Replace that with a small adapter around the existing MAS-CC provider abstraction.

Conceptually:

```python
class MuSRGenerationModel:
    def __init__(self, provider):
        self.provider = provider

    def inference(self, prompt: str, **kwargs):
        # translate to the normal MAS-CC provider call
        ...
        return text
```

Requirements:

- use the same provider/configuration system already used elsewhere in MAS-CC;
- do not add a separate OpenAI-specific dependency;
- allow University/OpenAI/local providers wherever the normal MAS-CC provider interface supports them;
- generation seed, model, temperature, and prompt version must be recorded in output metadata.

The CLI should look conceptually like:

```bash
python -m mas_cc.musr_team_allocation_generator.cli generate \
  --provider <existing-provider-config> \
  --model <model> \
  --num-tasks 100 \
  --population-size 24 \
  --branches-per-latent-fact 3 \
  --tree-depth 2 \
  --seed 0 \
  --output <path>
```

Adapt this to the repository's existing configuration conventions instead of inventing an incompatible CLI if an established pattern already exists.

---

## 7. Evidence generation

MuSR's useful idea is:

```text
latent exact fact
    ↓
entailment / reasoning tree
    ↓
explicit natural-language leaf evidence
```

Preserve this idea.

However, for MAS-CC we do **not** need to synthesize one giant final story containing every fact.

Instead produce:

1. a short **global scenario introduction** visible to all agents;
2. a collection of independently addressable **evidence cards/bundles**.

Example:

```json
{
  "evidence_id": "e_017",
  "latent_fact_id": "skill_alice_programming",
  "branch_id": "skill_alice_programming_b2",
  "text": [
    "Alice maintained the laboratory's Python analysis pipeline last year.",
    "When difficult bugs appeared, colleagues usually asked Alice to diagnose them."
  ]
}
```

This is better for the MAS experiment because evidence can be distributed and tracked exactly.

The generator should retain the full reasoning-tree provenance for analysis/debugging, but agents should only see the intended natural-language evidence.

---

## 8. Evidence leakage constraints

Generated evidence must not simply state the hidden fact or answer.

For example, reject phrases such as:

```text
Alice is good at programming.
Alice should be assigned to programming.
Allocation B is the best choice.
```

Adapt MuSR-style forbidden-text / structure validation where useful.

At minimum validate:

- task names / skill labels are not used in a way that directly leaks the latent conclusion when prohibited;
- candidate answer strings are not copied into evidence;
- gold option is never explicitly named as correct;
- each branch contains the requested number/shape of evidence statements;
- evidence is non-empty and parseable.

Retain failed generations only in logs, not in final datasets.

---

## 9. Distribute coherent evidence bundles, not individual random leaves

Do **not** blindly scatter individual sentences across agents.

The atomic distributed unit should be a coherent branch / evidence card.

For `N` agents construct:

```python
agent_evidence_ids: dict[agent_id, list[evidence_id]]
```

Desired properties:

- every agent gets useful evidence;
- evidence load is reasonably balanced;
- no agent receives the complete task;
- different agents cover different latent facts/branches;
- some controlled redundancy is allowed and configurable;
- the union of population evidence contains enough information to recover the correct allocation.

Keep a configurable redundancy parameter if useful, but do not make redundancy the only way to scale to `N=24`: the multiple-branches-per-fact generation above is the primary solution.

---

## 10. Validation

Validation is central. Do not accept generated worlds merely because the LLM output parses.

### A. Exact global answer

From the latent matrices:

```text
exactly one candidate allocation is optimal
```

and record its margin over the runner-up.

### B. Full population completeness

The union of all assigned evidence must cover the intended latent facts/branches sufficiently to represent the complete problem.

### C. No-single-agent-solution

No individual agent's initial information should structurally certify a unique global allocation.

Because natural-language evidence is probabilistic/semantic, implement the strongest exact structural check available using provenance:

```text
evidence card
    → latent fact / branch supported
```

An agent should not possess a provenance set sufficient to reconstruct every decisive latent quantity needed to uniquely identify the gold allocation.

### D. Avoid useless partial information

Also reject extremely weak allocations where most agents receive essentially irrelevant information.

Each agent should normally receive at least one evidence bundle that bears on one or more candidate allocations.

### E. Full-information solvability test

Add an explicit LLM validation mode that gives one validation agent **all natural-language evidence for a task** and asks it to solve the same `K`-way allocation problem that experimental agents will see.

This checks the crucial semantic property:

```text
all generated natural-language evidence
        ↓
capable LLM
        ↓
gold allocation
```

The exact latent solver still defines correctness; this LLM test only verifies that the generated language actually communicates enough information for the intended reasoning task.

For development, run this with `gpt-5.6-terra` (through the MAS-CC provider layer) on every generated candidate task or on a configurable validation subset.

Recommended acceptance logic for the first dataset:

- full-information answer must equal `gold_index`;
- parsing must succeed;
- retain the model's rationale/output for debugging;
- reject/regenerate tasks that repeatedly fail full-information validation.

Because LLM inference is stochastic, allow a configurable number of validation attempts rather than rejecting on a single miss. For example, require a strong majority such as `2/3` successful full-information solves.

### F. Partial- and zero-information validation

Also prepare a separate validation study analogous to the relational task validation:

```text
full evidence accuracy      → high
partial evidence accuracy   → above chance but below full
zero evidence accuracy      → approximately chance
```

This is especially important before launching the main population study. Full-information solvability should be checked during dataset QA; the broader full/partial/zero calibration can remain a separate validation experiment.

Neither empirical test defines the gold answer. The exact latent allocation solver remains the source of ground truth.

---

## 11. Frozen output schema

Each generated world should be self-contained and immutable for downstream experiments.

Suggested structure:

```json
{
  "task_id": "...",
  "task_family": "musr_team_allocation",
  "scenario": "...",

  "people": [...],
  "tasks": [...],
  "skills": [...],

  "options": [...],
  "gold_index": 1,

  "evidence": [...],
  "agent_evidence_ids": {...},

  "latent": {
    "skill_matrix": {...},
    "cooperation_matrix": {...},
    "candidate_scores": [...],
    "gold_allocation": ...,
    "margin": ...
  },

  "reasoning_provenance": {...},

  "generation": {
    "seed": 0,
    "provider": "...",
    "model": "...",
    "temperature": ...,
    "tree_depth": 2,
    "branches_per_latent_fact": 3,
    "prompt_version": "...",
    "musr_repo": "Zayne-sprague/MuSR",
    "musr_commit": "..."
  }
}
```

If the existing relational task schema provides suitable field conventions, follow those conventions rather than introducing gratuitously different names.

---

## 12. Relationship to the existing game

This handoff is primarily about **dataset generation**.

Do not build a separate population game.

The intended downstream design remains:

```text
games/relational_reasoning/imitation_round_feedback
```

with a task-family switch such as:

```yaml
task_family: spatial_relational
```

or:

```yaml
task_family: musr_team_allocation
```

The runtime should continue to own:

- population state;
- current discrete vote;
- microscopic updates;
- rounds;
- controller;
- sensing;
- actuation;
- metrics;
- analysis.

The task adapter should expose:

```python
scenario
options
gold_answer
evidence_by_id
initial_evidence_by_agent
```

and any task-specific epistemic/provenance utilities needed by analysis.

Do not duplicate the controller/runtime inside the generator package.

---

## 13. Future q-message / blackboard compatibility

Design the evidence objects so they can later be shared through a persistent message board.

The planned future microscopic interaction is:

```text
private evidence
+ q sampled persistent messages
        ↓
LLM update
        ↓
new discrete vote
+ new public message
```

Therefore every initial evidence item must have a stable ID.

Later, a public message can reference:

```python
shared_evidence_ids
author_id
vote
natural_language_reason
reply_to
timestamp/update_index
```

Do not implement the blackboard as part of this generator unless it is trivial and explicitly requested. The generator only needs to make future provenance possible.

---

## 14. Tests

Add tests for at least:

1. latent allocation generator always has a unique optimum;
2. deterministic reproduction under fixed seed for the purely programmatic layer;
3. provider adapter works with a mocked MAS-CC provider;
4. reasoning-tree/evidence serialization round-trips;
5. forbidden-answer leakage is rejected;
6. evidence bundles are distributed to exactly `N` agents;
7. every evidence ID referenced by an agent exists;
8. no-single-agent structural validation works;
9. complete population evidence passes completeness validation;
10. one small mocked end-to-end generation produces a valid frozen JSON world without making real provider calls.

Real-provider generation should be an integration test / manual command, not required for normal unit tests.

---

## 15. Acceptance criteria

The implementation is complete when:

- MAS-CC can generate new Team Allocation worlds without requiring a pre-generated MuSR dataset;
- generation calls go through the existing MAS-CC provider layer;
- the gold answer is computed exactly without an LLM;
- generated tasks pass a full-information LLM solvability check against that exact gold answer;
- multiple independent MuSR-style reasoning branches are generated per latent fact;
- the output contains enough evidence for meaningful distribution over `N=24`;
- evidence is distributed as coherent bundles;
- no single agent initially has a structurally complete solution;
- output is frozen JSON compatible with a thin task adapter in the existing game;
- generation metadata, upstream MuSR commit, and attribution are recorded;
- no task-generation LLM calls occur during the population experiment.

---

## 16. Explicit non-goals for this implementation

Do not add:

- semantic retrieval;
- embeddings;
- a blackboard runtime;
- new controller theory;
- new thermodynamic metrics;
- a second population game;
- tool-use agents;
- dynamic task generation during experiments.

Keep this implementation focused on the new frozen task generator.

---

## 17. Migration from the previous handoff

The previous adapter/converter implementation should **not constrain this work**.

Specifically:

```text
OLD:
download/run MuSR dataset
→ convert released MuSR examples
→ distribute existing leaves
```

is replaced by:

```text
NEW:
reuse/adapt MuSR generation code
→ call LLM through MAS-CC provider
→ generate many evidence branches ourselves
→ distribute
→ validate
→ freeze
```

Delete or archive the previous `musr_team_allocation_generator` adapter if it would create ambiguity.

Preserve only pieces that are genuinely reusable, such as schema helpers, exact distribution validation, or attribution text.

---

## Upstream attribution note

This implementation is an extension/adaptation of the Team Allocation and reasoning-tree generation ideas/code from:

**MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning**  
Zayne Sprague et al.

Upstream repository: `https://github.com/Zayne-sprague/MuSR`

MuSR is released under the MIT License. Preserve the required copyright/license notice for copied or substantially adapted source files.
