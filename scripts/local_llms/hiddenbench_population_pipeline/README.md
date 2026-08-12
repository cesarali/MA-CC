# HiddenBench Population Scripts

This directory is a collection of directly runnable scripts for:

1. downloading and preserving the official HiddenBench benchmark;
2. producing a population-neutral canonical representation;
3. using an LLM API to generate and verify reusable paraphrase pools;
4. using an LLM API to propose and verify semantic factorizations;
5. constructing exact, paraphrased, or factorized populations for arbitrary \(N\);
6. auditing whether the original and transformed tasks preserve the Hidden Profile information gap;
7. running a paper-style HiddenBench test as a separate experiment.

The official source is the 65-task dataset:

```text
YuxuanLi1225/HiddenBench
```

The raw `benchmark.json` is preserved unchanged. All transformations are written to separate directories.

---

## 1. Running the scripts

```bash
conda env update -n MA-CC -f environment.yml
cd scripts/local_llms/hiddenbench_population_pipeline
conda run --live-stream -n MA-CC python \
  scripts/prepare_hiddenbench.py --agents 0
```

No local installation or virtual environment is needed. The scripts use the
project's `MA-CC` conda environment; either prefix each command with
`conda run --live-stream -n MA-CC`, as above, or activate that environment
first with `conda activate MA-CC`.

**Every default path is anchored to the repository, not to the working
directory**, so a command produces the same files wherever it is launched from.
The corpus goes to the repository's `data/hidden_bench/` - the one location
`mas_cc` reads it from - while `annotations/` and `results/` stay inside this
directory. The path examples below are written relative to the repository root;
`--data-root` and `--input` only need passing when overriding a default.

API-backed scripts use the repository-root `.env` (never commit it):

```text
POTSDAM_API_KEY=...
BASE_POTSDAM_LLM_URL=https://…
LLM_PROTOCOL=chat_completions

# Optional proxy-model overrides
LLM_PARAPHRASE_MODEL=microsoft/gpt-5
LLM_FACTORIZATION_MODEL=microsoft/gpt-5.5
LLM_PARAPHRASE_VERIFIER_MODEL=microsoft/gpt-5.5
LLM_FACTORIZATION_VERIFIER_MODEL=microsoft/gpt-5.5
LLM_AUDIT_MODEL=microsoft/gpt-5.5
```

The client uses the OpenAI Python SDK against the proxy's OpenAI-compatible
Chat Completions endpoint. `LLM_API_KEY` and `LLM_BASE_URL` override the
university variables for another compatible service. No credential is embedded
in code, annotations, audit logs, or result files.

`LLM_TEMPERATURE` is the global fallback and defaults to `0.2`. Semantic
annotation generation supports more specific settings, each of which takes
precedence over the global value:

```text
LLM_PARAPHRASE_TEMPERATURE=1.0
LLM_FACTORIZATION_TEMPERATURE=1.0
LLM_PARAPHRASE_VERIFIER_TEMPERATURE=1.0
LLM_FACTORIZATION_VERIFIER_TEMPERATURE=1.0
```

The current proxy deployments `microsoft/gpt-5`, `microsoft/gpt-5-codex`, and
`microsoft/gpt-5.5` reject temperatures other than `1`. The client
automatically omits any incompatible value and records that adjustment, but the
recommended explicit configuration for these models is `1.0`.

For a generic endpoint instead, use:

```text
LLM_BASE_URL=https://provider.example/v1
LLM_PROTOCOL=chat_completions
```

For the OpenAI Responses API, keep:

```text
LLM_PROTOCOL=responses
```

---

## 2. Data layout

The preparation script creates:

```text
data/hidden_bench/
├── source/
│   ├── benchmark.json
│   └── source_metadata.json
├── canonical/
│   └── tasks.json
└── scaled/
    ├── exact_replication/
    │   └── N_32.json
    ├── paraphrased_replication/
    │   └── ...
    └── factorized_evidence/
        └── ...
```

Semantic annotations are stored independently:

```text
annotations/
├── paraphrases.json
├── factorizations.json
└── generation_audit.jsonl
```

Experimental outputs are stored separately:

```text
results/
├── information_sufficiency_audit.json
└── hiddenbench_standard.json
```

---

## 3. Download the basic benchmark

The agreed interface is:

```bash
conda run --live-stream -n MA-CC python \
  scripts/local_llms/hiddenbench_population_pipeline/scripts/prepare_hiddenbench.py \
  --agents 0 \
  --data-root data/hidden_bench
```

`--agents 0` means:

- download the authors' `benchmark.json`;
- preserve it without transformation;
- record its commit/hash metadata;
- create `canonical/tasks.json`;
- do not create a scaled population.

The separate raw-only downloader is unnecessary.

## 3.1 Reproducible 32-agent run

Run these commands in order after activating `MA-CC`. They are the only population-building commands in
this bundle; do not run the expensive `run_hiddenbench_standard.py` protocol
unless it has been explicitly requested.

```bash
python scripts/prepare_hiddenbench.py --agents 0 --data-root data/hidden_bench
python scripts/generate_semantic_annotations.py \
  --input data/hidden_bench/canonical/tasks.json --output-dir annotations \
  --mode both --paraphrases-per-type 10 --factorization-alternatives 4 \
  --max-components 4 --resume
python scripts/run_information_sufficiency_audit.py \
  --input data/hidden_bench/canonical/tasks.json \
  --output results/canonical_information_sufficiency.json
python scripts/prepare_hiddenbench.py --agents 32 --method exact_replication \
  --data-root data/hidden_bench
python scripts/prepare_hiddenbench.py --agents 32 --method paraphrased_replication \
  --annotations annotations/paraphrases.json --data-root data/hidden_bench
python scripts/prepare_hiddenbench.py --agents 32 --method factorized_evidence \
  --annotations annotations/factorizations.json --data-root data/hidden_bench
python scripts/run_information_sufficiency_audit.py \
  --input data/hidden_bench/scaled/paraphrased_replication/N_32.json \
  --output results/paraphrased_N32_information_sufficiency.json
python scripts/run_information_sufficiency_audit.py \
  --input data/hidden_bench/scaled/factorized_evidence/N_32.json \
  --output results/factorized_N32_information_sufficiency.json
```

The original source benchmark remains untouched in `data/hidden_bench/source/`;
canonical tasks live in `data/hidden_bench/canonical/`; all derived datasets are
under `data/hidden_bench/scaled/*/N_32.json`; and generated audit material lives
under `annotations/` and `results/`.

### Current execution record

The source-only preparation command and the deterministic exact `N_32` command
have completed for the checked-in working tree. Annotation generation, the two
annotation-derived datasets, and LLM sufficiency audits require a reachable
university proxy; they should be resumed with the commands above if proxy
connectivity is unavailable during a run. Returned per-call token usage is
preserved in annotation and result metadata when the proxy supplies it.

---

## 4. Exact replication

Exact replication requires no LLM-generated annotation.

```bash
python scripts/prepare_hiddenbench.py \
  --agents 32 \
  --method exact_replication \
  --data-root data/hidden_bench
```

For an original task with \(C=4\) hidden evidence types and \(N=16\), the script assigns approximately four agents to each type.

The source facts are not changed. Only their population multiplicity changes.

---

## 5. Generate reusable semantic annotations

### Paraphrases and factorizations together

```bash
python scripts/generate_semantic_annotations.py \
  --input data/hidden_bench/canonical/tasks.json \
  --output-dir annotations \
  --mode both \
  --paraphrases-per-type 10 \
  --factorization-alternatives 4 \
  --max-components 4
```

For initial development, restrict the call:

```bash
python scripts/generate_semantic_annotations.py \
  --input data/hidden_bench/canonical/tasks.json \
  --output-dir annotations \
  --mode both \
  --task-ids 1 2 3 \
  --paraphrases-per-type 10
```

To continue a partially completed pool:

```bash
python scripts/generate_semantic_annotations.py \
  --input data/hidden_bench/canonical/tasks.json \
  --output-dir annotations \
  --mode paraphrases \
  --paraphrases-per-type 10 \
  --resume
```

### What the API does

The semantic generation program has two LLM stages:

```text
generator
   ↓
candidate paraphrases / factorizations
   ↓
verifier
   ↓
accepted annotation pool
```

The generator receives the full task so it can avoid answer leakage and overlap with other hidden facts. The verifier independently checks the candidates.

The audit log records each generation and verification call.

Generation is strictly task-sequential: it completes and verifies every
paraphrase pool for a task, completes and verifies its factorizations, writes
both JSON checkpoints, and then advances. `tqdm` displays the task number,
current evidence item, accepted paraphrase count, factorization alternatives,
verification result, and task checkpoint. Re-run the same command with
`--resume` after an interruption; accepted variants and finalized evidence
records are retained without another generation call. A completed traversal
marks `annotations/paraphrases.json` and `annotations/factorizations.json` as
`frozen`; build datasets only from these frozen files.

### Why 10 paraphrases support the 32-agent condition

The pool is prepared **per evidence type**, independently of \(N\).

If a type has 10 accepted variants, a later population can select:

- 2 variants for a small population;
- 8 variants for a medium population;
- up to 10 variants for the largest supported allocation of that type.

No new API call is needed when moving from a maximum design to a smaller \(N\).

For \(C\) balanced evidence types and \(R\) variants per type, unique assignment supports approximately:

\[
N_{\max}=C R.
\]

For four evidence types and 10 variants each:

\[
N_{\max}\approx 40.
\]

### Why factorization is not generated to a target pool size

Factorization describes semantic structure, not population size.

A source clue may naturally have:

\[
L_c \in \{2,3,4\}
\]

meaningful components. Those components are generated once. The population allocator later decides how they are divided or replicated across \(N\) agents.

Some clues are not meaningfully factorizable. The generator is instructed to mark them as such instead of performing arbitrary sentence splitting.

---

## 6. Build paraphrased populations

```bash
python scripts/prepare_hiddenbench.py \
  --agents 32 \
  --method paraphrased_replication \
  --annotations annotations/paraphrases.json \
  --data-root data/hidden_bench
```

By default, a paraphrase is not reused within one task/population. If the requested population exceeds the available pool:

```bash
python scripts/prepare_hiddenbench.py \
  --agents 256 \
  --method paraphrased_replication \
  --annotations annotations/paraphrases.json \
  --allow-paraphrase-reuse \
  --data-root data/hidden_bench
```

For controlled experiments, generating a larger pool is preferable to reuse.

---

## 7. Build factorized populations

```bash
python scripts/prepare_hiddenbench.py \
  --agents 32 \
  --method factorized_evidence \
  --annotations annotations/factorizations.json \
  --data-root data/hidden_bench
```

The factorization and the division over agents are separate objects:

```text
factorizations.json
    defines semantic components
            ↓
prepare_hiddenbench.py
    constructs the N-agent allocation
```

If there are \(M\) factor components:

- \(N=M\): normally one component per agent;
- \(N>M\): components are replicated;
- \(N<M\): some agents receive several components.

The allocator tries not to place two components of the same evidence type in one agent. When unavoidable, it records a diagnostic warning.

If a clue is verified as non-factorizable, the task is intentionally excluded
from this condition rather than split arbitrarily. The exact exclusion list and
reasons are stored in `metadata.excluded_tasks` in the resulting
`N_32.json`; exact and paraphrased conditions retain all benchmark tasks.

The allocation must still be empirically audited: a bundle of several components may accidentally give one agent enough information to answer correctly.

---

## 8. Run the information-sufficiency audit first

### Original benchmark

```bash
python scripts/run_information_sufficiency_audit.py \
  --input data/hidden_bench/canonical/tasks.json \
  --output results/original_sufficiency.json \
  --sessions 10
```

### A scaled benchmark

```bash
python scripts/run_information_sufficiency_audit.py \
  --input data/hidden_bench/scaled/factorized_evidence/N_32.json \
  --output results/factorized_N32_sufficiency.json \
  --sessions 10
```

The script separately tests:

- shared information only;
- shared plus each original hidden clue;
- original complete information;
- original complete information with one hidden clue removed;
- every unique transformed agent packet;
- pooled transformed information;
- each individual factor component.

The default validity criterion reproduces the paper's task-generation filter:

\[
Y^{\mathrm{full}}\geq 0.80,
\qquad
Y^{\mathrm{partial}}\leq 0.20,
\]

using ten independent sessions.

For a transformed dataset, the desired pattern is:

```text
pooled transformed information     -> high accuracy
each transformed local packet      -> low accuracy
```

A transformed dataset that fails this audit should not enter the multi-agent experiment.

---

## 9. Run the standard HiddenBench protocol separately

### Original benchmark

```bash
python scripts/run_hiddenbench_standard.py \
  --input data/hidden_bench/canonical/tasks.json \
  --output results/original_standard.json \
  --sessions 10 \
  --rounds 15
```

### Exact replication

```bash
python scripts/run_hiddenbench_standard.py \
  --input data/hidden_bench/scaled/exact_replication/N_32.json \
  --output results/exact_N32_standard.json \
  --sessions 10 \
  --rounds 15
```

### Paraphrased or factorized condition

```bash
python scripts/run_hiddenbench_standard.py \
  --input data/hidden_bench/scaled/paraphrased_replication/N_32.json \
  --output results/paraphrased_N32_standard.json \
  --sessions 10 \
  --rounds 15
```

The runner executes:

1. Hidden Profile pre-discussion votes;
2. sequential public discussion;
3. Hidden Profile post-discussion votes;
4. Full Profile votes.

It reports:

- average-rule pre accuracy;
- average-rule post accuracy;
- average-rule Full Profile accuracy;
- majority-rule accuracy;
- information-integration gain:
  \[
  Y^{\mathrm{post}}-Y^{\mathrm{pre}};
  \]
- collective-reasoning gap:
  \[
  Y^{\mathrm{full}}-Y^{\mathrm{post}}.
  \]

The default does not stop early. This matches the fixed communication depth used in the paper.

### Meaning of a communication round in this implementation

The paper's prompt gives one speaker the complete prior public transcript and then says, “It's your turn to speak.” The runner therefore implements one communication round as one sequential public speaking event.

At \(T=15\), the transcript contains at most 15 public utterances.

The default speaker schedule is round-robin with a reproducibly randomized initial offset. A random-speaker condition can be selected with:

```bash
--speaker-order random
```

---

## 10. Recommended experimental order

Run the project in this order:

```text
1. Download and canonicalize
2. Reproduce the original sufficiency pattern
3. Reproduce the original standard benchmark
4. Build exact-replication populations
5. Audit exact-replication populations
6. Run exact-replication standard tests
7. Generate paraphrase pools
8. Build and audit paraphrased populations
9. Generate selected factorizations
10. Build and audit factorized populations
11. Run standard tests only on validated transformed datasets
```

The first scientific comparison should be exact replication, because it changes \(N\) while holding evidence semantics fixed.

---

## 11. Cost control

Generating up to 10 paraphrases for every hidden fact across all 65 tasks can require many API calls.

Begin with:

```bash
--task-ids 1 2 3
--paraphrases-per-type 10
```

After inspecting the audit log and outputs, scale to the full dataset.

For large runs:

- use a pinned model snapshot when possible;
- preserve the generated pools;
- never regenerate annotations per experimental seed;
- reuse one frozen annotation release across all \(N\);
- checkpoint outputs with `--resume`.

---

## 12. Reproducibility

Record:

- Hugging Face source commit and SHA-256;
- generator model;
- verifier model or model configuration;
- API protocol and base URL;
- annotation release;
- population size;
- scaling method;
- allocation seed;
- benchmark model;
- sampling temperature;
- number of sessions;
- communication depth;
- speaker-order rule.

The generated source metadata and result JSON files store most of this automatically.

### Freezing a complete task subset from an unfinished pool

Do not mark an incomplete global annotation file frozen. When selected tasks
are complete, create a deterministic task-scoped release instead:

```bash
python scripts/freeze_paraphrase_subset.py \
  --annotations annotations/paraphrases.json \
  --task-ids 2 --agents 4 8 16 32 \
  --output annotations/paraphrases_task_2_frozen.json
```

The command checks every canonical evidence type, accepted-variant uniqueness,
source-text identity, and the capacity required by the largest requested
population. It adds stable IDs to accepted legacy variants that lack them and
preserves their generation and verification metadata. The release records the
source annotation SHA-256, selected tasks, populations, and required capacity;
only this validated subset receives `status: frozen`.
