# Embedding model smoke tests

These scripts verify the model-loading pieces needed by a future empowerment
estimator. They do not implement or train that estimator, and they never
fine-tune the sentence-transformer backbone.

The supported Hugging Face checkpoints are:

- `intfloat/e5-small-v2`
- `intfloat/e5-base-v2`
- `sentence-transformers/all-MiniLM-L6-v2`

`test_embeddings.py` downloads and loads one or all checkpoints, embeds example
agent trajectories, and validates normalized outputs and cosine similarities.
`test_classification.py` fits a scikit-learn logistic regression on frozen E5
embeddings from a small synthetic trajectory dataset. `test_info_nce.py` keeps
the E5 encoder frozen and trains only two small projection MLPs on matched
context/action and future-outcome pairs.

## Installation

From the repository root, update the normal Conda environment and install the
project in editable mode:

```bash
conda env update -n MA-CC -f environment.yml
conda run -n MA-CC python -m pip install -e .
```

The first execution of each checkpoint requires network access so Hugging Face
can download it. Later executions can use the local cache. By default, model
snapshots are stored below `~/.cache/huggingface/hub`.

The scripts load the repository-root `.env` before importing Hugging Face. Set
`HF_HOME` there to put models, tokenizers, and other Hugging Face cache data on
a high-capacity cluster filesystem. Use an absolute path that exists on compute
nodes, for example:

```bash
HF_HOME=/cluster/scratch/your-username/huggingface-cache
```

An `HF_HOME` already exported by the shell or a cluster job takes precedence
over the `.env` value. Model snapshots are stored in `$HF_HOME/hub`. Existing
files in the default cache are not automatically moved when this setting is
changed. Do not commit the cache or downloaded model weights to this
repository.

`HF_HOME` controls disk storage, not the RAM or GPU memory required after a
model is loaded. On a memory-constrained job, test one model with `--model`
instead of `--all`; E5-small and MiniLM also require less runtime memory than
E5-base.

## Running the tests

Run these commands from the repository root:

```bash
conda run --live-stream -n MA-CC python scripts/embedding_model_tests/test_embeddings.py --all
conda run --live-stream -n MA-CC python scripts/embedding_model_tests/test_classification.py
conda run --live-stream -n MA-CC python scripts/embedding_model_tests/test_info_nce.py
```

To test one checkpoint or override the classifier/InfoNCE default:

```bash
conda run --live-stream -n MA-CC python scripts/embedding_model_tests/test_embeddings.py --model intfloat/e5-small-v2
conda run --live-stream -n MA-CC python scripts/embedding_model_tests/test_classification.py --model sentence-transformers/all-MiniLM-L6-v2
conda run --live-stream -n MA-CC python scripts/embedding_model_tests/test_info_nce.py --model intfloat/e5-base-v2
```

The embedding test prints each model's embedding tensor shape, embedding
dimension, and cosine-similarity matrix. The classification test prints split
sizes, accuracy, a classification report, and a confusion matrix. The InfoNCE
test prints its initial and final losses and the similarity matrices before and
after projection-head training. Each script raises an assertion with a clear
message if it encounters invalid values, shapes, norms, gradients, or loss
behavior.
