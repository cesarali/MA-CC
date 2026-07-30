# Task: Add Minimal E5 / Sentence-Transformer Model Tests

## Goal

Add a small, self-contained test area to the repository that verifies that the embedding models needed for a future empowerment estimator can be downloaded, loaded, and used successfully.

This task is **not** to implement the empowerment estimator yet. The goal is only to establish a clean, reproducible model-loading and testing setup.

## Required models

Use the following Hugging Face checkpoints:

1. `intfloat/e5-small-v2`
2. `intfloat/e5-base-v2`
3. `sentence-transformers/all-MiniLM-L6-v2`

Do not add `intfloat/e5-mistral-7b-instruct` in this task because it is substantially larger and not needed for the initial validation.

## Environment and dependency changes

Inspect the repository's existing environment setup and add the required dependencies in the appropriate places.

Update:

- the Conda environment file, such as `environment.yml`, if present;
- `requirements.txt`, if present;
- any other dependency file already used by the project.

Add only the libraries required for these tests. At minimum, this will likely include:

```text
torch
sentence-transformers
scikit-learn
numpy
```

Reuse existing pinned versions where possible. Avoid introducing incompatible duplicate version constraints.

The tests must work after installing the project through its normal Conda/environment setup.

## Directory structure

Create a separate script directory, for example:

```text
scripts/embedding_model_tests/
├── README.md
├── test_embeddings.py
├── test_classification.py
└── test_info_nce.py
```

The scripts should be directly runnable from the repository root.

## Test 1: Model loading and embeddings

Create:

```text
scripts/embedding_model_tests/test_embeddings.py
```

Requirements:

- load each of the three required models;
- allow selecting one model through a command-line argument;
- automatically download the model from Hugging Face when it is not cached;
- embed a small list of example agent-trajectory sentences;
- use normalized embeddings;
- print:
  - model name,
  - embedding tensor shape,
  - embedding dimension,
  - a cosine-similarity matrix;
- assert that:
  - all outputs are finite;
  - the number of embeddings matches the number of inputs;
  - normalized embeddings have approximately unit norm.

Example interface:

```bash
python scripts/embedding_model_tests/test_embeddings.py     --model intfloat/e5-small-v2
```

Also support running all three models:

```bash
python scripts/embedding_model_tests/test_embeddings.py --all
```

For E5 models, prepend an appropriate task prefix such as:

```text
query: Represent this agent trajectory for behavioral analysis.
```

The MiniLM model does not require the E5 prefix, although using the same serialized text content is fine.

## Test 2: Basic trajectory classification

Create:

```text
scripts/embedding_model_tests/test_classification.py
```

Implement a minimal binary classification experiment using frozen sentence embeddings and scikit-learn logistic regression.

Use a small synthetic dataset of textual agent trajectories labeled as:

- `1`: successful or coordinated;
- `0`: unsuccessful or uncoordinated.

Requirements:

- use `intfloat/e5-small-v2` by default;
- support overriding the checkpoint with `--model`;
- encode all texts with the sentence-transformer model;
- keep the encoder frozen;
- split the synthetic examples into training and test sets;
- train `LogisticRegression`;
- print:
  - training and test sizes,
  - accuracy,
  - classification report,
  - confusion matrix;
- set all relevant random seeds;
- include enough synthetic examples for a stratified split to work reliably;
- assert that predictions have the expected shape and contain no invalid values.

This test is only a smoke test. Do not interpret the resulting accuracy as a scientific benchmark.

Example interface:

```bash
python scripts/embedding_model_tests/test_classification.py
```

## Test 3: Minimal contrastive / InfoNCE test

Create:

```text
scripts/embedding_model_tests/test_info_nce.py
```

Implement a minimal version of the contrastive structure that will later be useful for an empowerment estimator.

Construct a small synthetic batch of matched text pairs:

```text
context/action text  <->  resulting future-outcome text
```

Examples:

- an agent communicates the correct key location;
- the group subsequently reaches the exit;
- an agent sends misleading evidence;
- the group subsequently selects the wrong answer.

Requirements:

- encode both sides using `intfloat/e5-small-v2` by default;
- keep the text encoder frozen;
- add two small trainable projection MLPs, `phi` and `psi`;
- compute the full pairwise similarity matrix;
- use matched diagonal pairs as positives and off-diagonal pairs as negatives;
- implement a symmetric InfoNCE loss;
- train only the projection heads for a small number of steps;
- print:
  - initial loss,
  - final loss,
  - similarity matrix before training,
  - similarity matrix after training;
- assert that:
  - the loss is finite;
  - gradients are finite;
  - the final loss is lower than the initial loss, allowing a reasonable tolerance;
  - tensor shapes are correct.

Example interface:

```bash
python scripts/embedding_model_tests/test_info_nce.py
```

## Shared implementation requirements

Keep the scripts simple and readable.

Use:

- `argparse` for command-line arguments;
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")`;
- deterministic random seeds where practical;
- clear error messages;
- type hints for nontrivial functions;
- `if __name__ == "__main__":` entry points.

Do not:

- modify the main model architecture of the repository;
- add the empowerment estimator itself;
- fine-tune the sentence-transformer backbone;
- add LoRA in this task;
- require a GPU;
- commit downloaded Hugging Face model weights to the repository.

Hugging Face should manage model caching through its normal cache directory.

## README

Create:

```text
scripts/embedding_model_tests/README.md
```

Document:

- the purpose of the three scripts;
- the supported model identifiers;
- installation instructions using the repository's Conda environment;
- exact commands to run all tests;
- expected outputs;
- where Hugging Face stores downloaded models;
- how to override the cache using `HF_HOME`;
- that the first execution requires network access;
- that subsequent executions can use the local cache.

Include an example:

```bash
export HF_HOME=/path/to/huggingface-cache
python scripts/embedding_model_tests/test_embeddings.py --all
python scripts/embedding_model_tests/test_classification.py
python scripts/embedding_model_tests/test_info_nce.py
```

## Optional convenience runner

If it fits the repository conventions, add:

```text
scripts/embedding_model_tests/run_all.sh
```

It should run the three tests sequentially and exit immediately if one fails.

Example:

```bash
bash scripts/embedding_model_tests/run_all.sh
```

## Validation

Before finishing:

1. Install or update the Conda environment using the repository's normal workflow.
2. Run the embedding test for all three checkpoints.
3. Run the classification smoke test.
4. Run the InfoNCE smoke test.
5. Confirm that all scripts work on CPU.
6. Confirm that no downloaded model files are added to Git.
7. Report:
   - files created;
   - dependency files changed;
   - commands executed;
   - whether each test passed;
   - any platform-specific issue encountered.

## Acceptance criteria

The task is complete when:

- all three Hugging Face models can be downloaded and loaded;
- all three scripts run from the repository root;
- the classification script trains and evaluates successfully;
- the InfoNCE loss decreases during its smoke test;
- the Conda and requirements configuration contains the needed dependencies;
- the setup works without requiring a GPU;
- the README contains reproducible commands;
- no model weights or cache files are tracked by Git.
