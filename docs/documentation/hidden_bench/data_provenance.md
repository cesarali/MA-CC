# HiddenBench data provenance

What the `hidden_bench` games read, where it came from, and which parts of the
upstream pipeline are finished. Required by §0 of
[`docs/tdd/architecture/05082026_HIDDENBENCH_BRIEF.md`](../tdd/architecture/05082026_HIDDENBENCH_BRIEF.md).

**Nothing in this document was regenerated.** The corpus and its expansions are
prior work by the repository owner; the game code reads what the pipeline
produced and never rebuilds it.

---

## 1. Where everything actually is

The corpus is the brief's §2 location, `data/hidden_bench/`. It previously sat
under `scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/`,
which buried a shared run-time input inside a tool directory; the pipeline now
writes straight to the shared location, so producer and consumers name one path
and no copy exists. The root is:

```
data/hidden_bench/
├── source/
│   ├── benchmark.json          65 tasks, byte-preserved from Hugging Face
│   └── source_metadata.json    repo id, commit SHA, sha256
├── canonical/
│   └── tasks.json              population-neutral normalization  ← the games read this
└── scaled/
    └── exact_replication/
        └── N_32.json           the only prebuilt population that exists
```

`data.py::DEFAULT_CORPUS_ROOT` points there; `game.options.corpus_root`
overrides it per run.

### Upstream identity

| Field | Value |
| --- | --- |
| Hugging Face repo | `YuxuanLi1225/HiddenBench` |
| Resolved commit | `1e3c25b1fd798c6717f4df0463edd3825c8e37f9` |
| File | `benchmark.json`, 65 tasks |
| SHA-256 | `2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3` |

Task shape: 58 tasks have 4 hidden items, 7 have 3. Every task has exactly 3
possible answers.

## 2. The scripts

| Role | File |
| --- | --- |
| Download + canonicalize + scale | `scripts/.../scripts/prepare_hiddenbench.py` |
| Shared primitives (seeding, allocation, vote normalization) | `scripts/.../scripts/hiddenbench_common.py` |
| Paper-protocol runner and **prompt templates** | `scripts/.../scripts/hiddenbench_evaluation.py` |
| LLM annotation generation | `scripts/.../scripts/generate_semantic_annotations.py` |
| Information-sufficiency audit | `scripts/.../scripts/run_information_sufficiency_audit.py` |

`scripts/` is not an importable package, so `data.py` and `vanilla/prompts.py`
**reimplement** the few primitives they need rather than importing them. Both
reimplementations are pinned by test, not by hope — see §5.

## 3. Scheme names: the brief guessed, the pipeline decides

The brief's §4 explicitly labelled its taxonomy a prior guess. It was wrong.
The authoritative names are `prepare_hiddenbench.py --method` values:

| Brief's guess | Actual | Notes |
| --- | --- | --- |
| `bijective` | *(no pipeline equivalent)* | Kept in `data.py` as an explicit alias for the paper's N == C baseline; refuses any other N. |
| `redundant` | **`exact_replication`** | Evidence types dealt round-robin over N agents, then shuffled. At N > C some types are held by several agents — which is what "redundant" was gesturing at. |
| `factorized` | **`factorized_evidence`** | A clue is split into verified semantic components. Needs LLM annotations. |
| — | **`paraphrased_replication`** | No brief equivalent. A clue is restated so N agents can hold distinct wordings of one fact. Needs LLM annotations. |
| `padded` | `padded` | **mas_cc-local.** Not a pipeline method; implemented in `data.py` because it is a genuine group-size control the pipeline does not cover. |
| `decoy` | `decoy` | **mas_cc-local**, same status. |

`data.py::PIPELINE_SCHEMES` and `LOCAL_SCHEMES` keep the two groups apart so
nobody mistakes a locally-derived population for audited pipeline output.

## 4. Field-name reconciliation (§3.1)

The brief guessed a flat record. The corpus is richer, and `schemas.py` keeps
the corpus's names:

| Brief §3.1 | Corpus | Note |
| --- | --- | --- |
| `name` | `name` | same |
| `description` | `source_description` **and** `scenario_description` | Two of them. The first says "the other three community leaders" and is only correct at N = 4; the second is the population-neutral rewrite. `HiddenProfileTask.description_for(n)` picks. |
| `hidden_information: tuple[str, ...]` | `list[{evidence_type, source_text}]` | Structured. The list index **is** the evidence type, and the expansion script relies on that; `_canonical_task` asserts it. |
| `n_agents_native` | `source_base_agent_count` | same meaning |
| `source` | *(absent)* | Synthesized: `"canonical"` or `"scaled:<method>"`. The corpus does not record manual/adapted/generated provenance per task. |
| — | `population_wording_changes`, `population_instruction`, `rationale` | Extra; the first two are how the scenario is made N-neutral. |

Per-task validation stats (§1.4's ≥0.80 / ≤0.20 thresholds) are **not** carried
in `canonical/tasks.json`. They are produced separately by
`run_information_sufficiency_audit.py` into `results/`, which has not been run
(§6). `HiddenProfileTask.validation_stats` is therefore `None` for every task
today, and the §1.4 assertion has nothing to assert against.

## 5. Prompt templates: the script wins over the brief

The brief §1.5 quotes the paper's templates with markdown line wrapping applied.
`hiddenbench_evaluation.py` contains the same templates as running code. Where
they disagree, `vanilla/prompts.py` follows the script, because that is what
produced the numbers the repository already has. Differences that change
rendered characters:

| | Brief §1.5 | Implemented |
| --- | --- | --- |
| Line break | after "…these information are" | after "…these information" |
| Blank line before the facts | yes | **no** — `please reason carefully:` then the facts on the next line |
| Fact rendering | unspecified | `- ` bullets, one per line |
| `%possible_answers%` | unspecified | Python list repr: `['West City', 'East Town', 'North Hill']` |
| JSON skeleton indent | 4 spaces | **2 spaces** |
| What gets shuffled | "facts" | shared **and** private together, in one list |
| `%extra%` when unset | `"… sentences. "` | `"… sentences."` — no trailing space |

The paper's typos (`"randomly shuffle"`, `"concise-just"`) are reproduced
deliberately and pinned by
`tests/mas_cc/test_hidden_bench_prompts.py::test_the_papers_typos_are_reproduced_not_corrected`.

### The reimplemented primitives, and what pins them

| Reimplemented | Original | Pinned by |
| --- | --- | --- |
| `stable_seed`, `_balanced_type_assignment` | `hiddenbench_common.py` | `test_hidden_bench_data.py::test_derived_exact_replication_matches_the_pipelines_own_output` — reproduces all 65 tasks of `N_32.json` agent-for-agent |
| `normalize_vote` | `hiddenbench_common.py` | `test_hidden_bench_prompts.py` |
| Prompt templates | `hiddenbench_evaluation.py` | `test_hidden_bench_prompts.py` golden files |

Because the allocation reproduces exactly, `exact_replication` is derived
in-process for **any** N rather than requiring a prebuilt file per N. A prebuilt
file is still preferred whenever one exists.

## 6. What is finished, and what is not

> §0 of the brief says to stop and report rather than regenerate if any of the
> preprocessing is half-finished. This section is that report. Nothing was
> regenerated.

| Item | Status |
| --- | --- |
| Corpus downloaded and preserved | **Complete** — 65 tasks, SHA recorded |
| Canonical normalization | **Complete** — `canonical/tasks.json` |
| `exact_replication` at N = 32 | **Complete** — `scaled/exact_replication/N_32.json` |
| `exact_replication` at other N | Not on disk; **derived in-process**, provably identical to the pipeline's rule |
| Paraphrase annotation pool | ⚠️ **Incomplete** — 43 of 65 tasks, `status` is unset rather than `"frozen"` |
| Factorization annotation pool | ⚠️ **Incomplete** — 42 of 65 tasks, `status` unset |
| `paraphrased_replication` populations | ❌ **Not built** — `prepare_hiddenbench.py` refuses unbuilt/unfrozen annotations |
| `factorized_evidence` populations | ❌ **Not built** — same |
| Information-sufficiency audit (§1.4) | ❌ **Not run** — `results/` is empty |

Both annotation pools were generated with `microsoft/gpt-5-mini` as generator
*and* verifier. That is worth flagging independently of completeness: a
self-verified pool has no independent check, and the pipeline README's design
(§5) describes generator and verifier as separate stages.

**Consequence for the games.** `assignment_scheme: paraphrased_replication` and
`factorized_evidence` raise a `HiddenBenchDataError` naming the exact command
that would build them. They are never synthesized at run time, because doing so
would invent evidence content that the pipeline deliberately routes through an
LLM verifier.

**To finish them** (needs a reachable university proxy), from
`scripts/local_llms/hiddenbench_population_pipeline/`:

```bash
python scripts/generate_semantic_annotations.py \
  --input data/hidden_bench/canonical/tasks.json --output-dir data/hidden_bench/annotations \
  --mode both --paraphrases-per-type 10 --factorization-alternatives 4 \
  --max-components 4 --resume
python scripts/prepare_hiddenbench.py --agents 32 --method paraphrased_replication \
  --annotations data/hidden_bench/annotations/paraphrases.json --data-root data/hidden_bench
python scripts/prepare_hiddenbench.py --agents 32 --method factorized_evidence \
  --annotations data/hidden_bench/annotations/factorizations.json --data-root data/hidden_bench
python scripts/run_information_sufficiency_audit.py \
  --input data/hidden_bench/canonical/tasks.json \
  --output results/canonical_information_sufficiency.json
```

Once the audit exists, wire its per-task output into
`HiddenProfileTask.validation_stats` and turn §1.4 into a real load-time
assertion — the hook is already in `schemas.py`.
