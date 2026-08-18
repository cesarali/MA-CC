# Relational support benchmark

**Question.** Before relational tasks carry a multi-agent experiment: does
possessing the designated supporting facts actually determine whether an LLM
solves the task?

**Quantity.** Correctness is scored on the **semantic relation** the model
names, never on the letter it was printed under. For a task of reasoning depth
`L`,

```text
A_k = P(correct | k of the L supporting facts are shown),   k = 0 .. L
```

and for `L = 2` the contrast the experiment exists to report:

```text
accuracy_full  >>  accuracy_partial      (and accuracy_zero at the chance floor)
```

There is no agent, no vote, no peer, no controller and no round anywhere in this
package. One task plus one evidence condition equals one prompt equals one
request equals one row.

---

## Answer-position control

The first run reproduced the reasoning separation (full 0.99, partial 0.33) and
also showed the model answering `A` in **89%** of zero-evidence items. Under the
generator's frozen letter assignment that makes `accuracy_zero` a measurement of
how often `A` happened to be correct, not of chance.

So the semantic answer is decoupled from the displayed letter. The generated
worlds, facts, chains, distractors and correct relations are **untouched**; only
the relation-to-letter assignment is rebuilt at prompt time. Every task and
evidence condition is presented **three times** — correct relation at `A`, at
`B`, at `C` — so position is balanced inside every analysis cell by
construction rather than by luck. `FACTS` and `QUESTION` are byte-identical
across those three; only the option block moves.

A model with a pure letter habit therefore scores exactly `1/K` no matter how
strong the habit is, which is what makes `accuracy_zero` interpretable.

`presentation.mode: frozen` reproduces the pre-control assignment, so the two
can be compared rather than argued about.

---

## The elimination confound (reported, not repaired)

Position is not the only way an item can be answerable without the evidence.
A partial condition leaves the query *displacement* undetermined — the validator
enforces that — but the three *displayed* relations are not all necessarily
reachable. Each omitted supporting fact is one unit step, so the reachable set
is the image of the shown constraints under every assignment of directions to
the missing links, and a displayed relation outside that image can be eliminated
with no further evidence.

Measured over the L=2 grid: **32%** of partial conditions leave only one feasible
option, 53% leave two, and just 15% are a genuine 3-way choice. A perfect
reasoner holding one fact would therefore score **0.634**, not 0.333.

Consequences, all of them reported rather than patched — this is a property of
the frozen world and its option set, and the benchmark may not edit either:

* `num_feasible_options` is recorded on every row;
* accuracy is reported split by it, and `num_feasible_options == 3` is the honest
  partial-evidence subset;
* `validation_report.json` carries a pooled `diagnostics` block with the collapse
  share and the eliminator ceiling.

It is deliberately **not** a pass/fail check. Tasks where every partial condition
collapses exist (`task_0003` in the example dataset), and aborting a run over a
legitimate frozen task would be wrong.

**So `P(correct | partial) ≈ P(correct | zero)` should not be expected to hold**,
and its failure is not evidence against the task family. The comparison that
survives this is `accuracy_full` against the eliminator ceiling, and the
`num_feasible_options == 3` subset against chance.

---

## What the conditions are

Given a task with supporting set `S` (`|S| = L`) and its distractors `D`:

| condition | shows | count |
|---|---|---|
| `zero` | `D` only | 1 |
| `partial` | `D` + a proper subset of `S` | `C(L,k)` per `k`, capped |
| `full` | `D` + all of `S` | 1 |

`D`, the question and the option block are **identical** across every condition
of a task. Facts are shuffled once per task and every condition renders its
subset in that same order, so conditions differ by deletion and by nothing else.

Fact identifiers are not shown. In the game an agent needs them to cite one;
here a list reading `f1, f3, f4` would announce that `f2` was withheld, which is
exactly the manipulation under test. The identifiers live in the results table.

Subsets for `0 < k < L` are **enumerated**, not prefixed: in a chain `A→B→C` the
two links are different evidence and may differ in how much they narrow the
answer, so they are reported separately before pooling. `max_subsets_per_k`
caps the `C(L,k)` blow-up at `L = 4` with a seeded sample.

---

## What is checked before anything is sent

`preflight` generates the tasks, renders all the prompts, runs all fourteen
checks and opens no provider. A `run` repeats the whole thing and **refuses to
send a single request** if any check fails.

| check | what it rules out |
|---|---|
| `reasoning_depth_matches_support_count` | a task whose declared `L` is not its chain length |
| `shown_support_matches_condition` | the renderer and the condition disagreeing |
| `full_condition_shows_every_supporting_fact` | a "full" prompt that is quietly partial |
| `distractor_condition_constant_across_conditions` | distractor load co-varying with evidence |
| `omitted_support_text_absent` | the withheld sentence appearing verbatim |
| `omitted_support_pair_not_re_derivable` | another shown fact relating the same entity pair |
| `full_condition_implies_stored_answer` | a task whose support does not prove its answer |
| `partial_conditions_underdetermined` | **a partial condition that still determines the answer** |
| `options_block_identical_across_conditions_within_permutation` | option text or order moving between evidence conditions |
| `facts_block_byte_identical_across_permutations` | a permutation changing anything but the options |
| `option_relation_set_invariant_across_permutations` | a permutation swapping which relations are offered |
| `correct_relation_sits_at_its_declared_position` | a mislabelled placement, i.e. wrong scoring |
| `correct_position_balanced_across_labels` | an unbalanced position design |
| `option_relations_distinct_within_prompt` | two options carrying the same relation |
| `question_identical_across_conditions` | the question moving between conditions |
| `no_task_record_fields_in_prompt` | a serialised task record reaching the model |
| `correct_answer_only_present_as_an_option` | the answer marked or repeated |
| `no_coordinates_in_facts_block` | ground-truth coordinates leaking |
| `facts_block_is_exactly_the_shown_facts` | any text in the fact list that is not a shown fact |

`partial_conditions_underdetermined` is the load-bearing one, and it is not a
string search. It re-solves the shown constraints with `geometry.py` — an
*independent* re-implementation of the v1 spatial semantics, typed out from the
generator README rather than imported — and asserts the query endpoints are not
connected. A leak arriving through a distractor's geometry rather than through
its words would pass every text check and fail this one.

That independence is not decorative: the first version of `geometry.py` had the
displacement sign inverted, and `full_condition_implies_stored_answer` caught it
on the first preflight.

Reproducibility is *executed*, not asserted: every dataset is regenerated from
the seed stored inside each task and compared as canonical JSON, by the
generator's own `validate_dataset.py`.

---

## Parameters

Swept, because they change the item a single model sees:

* `reasoning_depth` (L) — 1..4, generator-supported
* `distractors` — irrelevant facts sharing the page
* `num_options` — the chance floor

Held fixed, because they only decide *who is told what* and this benchmark tells
one model everything its condition allows: `population_size`,
`support_redundancy`, `distractor_redundancy`, `no_single_agent_solution`. They
are echoed into every output row so a later run cannot silently disagree.

`no_single_agent_solution` is dropped automatically at `L = 1` — one fact handed
to one agent *is* the whole proof, and the generator rejects that combination.

---

## Commands

```bash
# free: generate, render, validate, send nothing
python -m mas_cc.cli.main benchmark relational-support preflight \
  --config configs/benchmarks/relational_support/L2_validation.yaml

# one request per item
python -m mas_cc.cli.main benchmark relational-support run \
  --config configs/benchmarks/relational_support/L2_validation.yaml

# the tables
python -m mas_cc.cli.main benchmark relational-support summarize \
  --input-dir results/benchmarks/relational_support/relational-support-L2-validation
```

## Output

```text
<output-dir>/
├── datasets/<L?_D?_O?>/        the generated tasks + generator manifest
├── prompts/*.md                complete condition sets, diffable side by side
├── plan.json                   config echo, fingerprints, reproducibility results
├── validation_report.json      every check, per grid cell
├── rows.jsonl / rows.csv       one row per item
├── rows.partial.jsonl          crash-safe journal, removed on clean completion
└── summary/
    ├── accuracy_by_k.csv                 A_k per parameter condition
    ├── accuracy_by_condition_id.csv      per-subset, before pooling
    ├── headline_l2.csv                   full / partial / zero / gap
    ├── l2_single_fact_conditions.csv     the two singletons, separately
    ├── headline_overall.csv              pooled over distractor settings
    ├── accuracy_by_correct_position.csv  accuracy by displayed position
    ├── predicted_position_distribution.csv   which letter the model picked
    ├── predicted_relation_distribution.csv   which relation the model picked
    ├── accuracy_by_feasible_options.csv  accuracy by surviving menu size
    ├── summary.json
    └── summary.md
```

## Progress and crash safety

A run prints one line per completed item to stderr (running accuracy, error
count, ETA) and appends each row to `rows.partial.jsonl` the moment it returns.
A run that dies at item 900 of 960 keeps the 900 already paid for. The ordered
`rows.jsonl` and `rows.csv` are written at the end and the journal removed.

## A caveat worth stating

In a partial or zero condition, one or both queried entities may not appear in
any shown fact at all. That is unavoidable — it is what withholding a chain link
*means* — but it makes "evidence is missing" detectable from the prompt. `A_k`
therefore measures whether the model *can* derive the answer, not whether it
believes it can. A model that guesses when it notices missing evidence and one
that confabulates confidently will both show a low `A_k`; the `zero` condition is
what separates "insufficient evidence" from "guessable from the question alone".
