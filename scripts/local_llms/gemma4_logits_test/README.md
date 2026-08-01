# Gemma 4 12B logits smoke test

This is a minimal text-only infrastructure test for exactly
`google/gemma-4-12B-it`. It loads the multimodal instruction checkpoint once,
runs a forward pass, inspects full-vocabulary next-token logits, scores a small
allowed answer set both as individual tokens and as complete token sequences,
and optionally performs a short deterministic generation. It does not implement
an agent, planning, fine-tuning, or game logic.

## Prerequisites

- An NVIDIA A100 is recommended. BF16 weights occupy roughly 24 GB; allow at
  least 30 GB of cache space for weights and download overhead.
- Accept any applicable terms on the
  [model page](https://huggingface.co/google/gemma-4-12B-it), then authenticate:

  ```bash
  huggingface-cli login
  # or export HF_TOKEN without putting it in source control
  ```

- The repository's ignored `.env` must contain a real high-capacity location:

  ```dotenv
  HF_HOME=/path/to/high-capacity-storage/huggingface-cache
  ```

  The script loads this file before importing Hugging Face libraries, prints the
  resolved location, creates it if needed, and refuses a missing/placeholder path
  or less than 30 GiB free space.

## Install and run

From the repository root, use the existing environment:

```bash
conda activate MA-CC
conda install -c conda-forge torchvision
python -m pip install 'transformers>=5.5' 'accelerate>=1.0' \
  'huggingface-hub>=0.30' 'safetensors>=0.4' python-dotenv

python scripts/gemma4_logits_test/test_gemma4_logits.py

python scripts/gemma4_logits_test/test_gemma4_logits.py \
  --choices A B C \
  --top-k 20 \
  --max-new-tokens 8
```

The validated `MA-CC` setup is Python 3.11.15, PyTorch 2.13.0/CUDA 12.9,
torchvision 0.28.0, Transformers 5.14.1, and accelerate 1.14.0. Install
torchvision from the same Conda channel/build family as PyTorch; mixing the
conda-forge PyTorch build with a PyPI torchvision binary can leave compiled ops
unavailable. Transformers 5 uses the `dtype=` interface, while the code also
detects older installations that require `torch_dtype=`. Do not downgrade
PyTorch: its CUDA build must remain compatible with the host driver. CUDA is
required unless the deliberately slow `--cpu` option is supplied. Use
`--no-generate` to skip only the final generation check.

## Reading the output

- **Full-vocabulary logits** are the unnormalized scores for every possible next
  token. Their softmax sums to one; top-k shows the most likely tokens globally.
- **Single-token constrained probabilities** renormalize only the logits of
  choices represented by one token in the answer context.
- **Sequence-level constrained probabilities** teacher-force every complete
  answer, sum only its conditional token log-probabilities, then normalize across
  choices. This remains valid for multi-token choices.

The prompt asks whether 7 or 3 is larger, so `A` should normally win. A different
winner emits a warning; shape, finiteness, and probability-normalization errors
remain hard failures. Output also includes the device map, GPU capacity, allocated
and reserved memory, peak use, tensor shapes, tokenizer IDs, and raw generation.

## Common failures and cache inspection

- HTTP 401/403 or a gated-repository error means authentication or model-access
  acceptance is missing.
- CUDA errors mean no visible GPU, incompatible PyTorch/CUDA, or insufficient GPU
  memory. The unquantized checkpoint needs about 24 GB plus runtime overhead.
- A Triton `cuda.h`/`ptxas` error means the activated CUDA-enabled Conda
  environment lacks its matching CUDA development headers/tools or its `bin`
  directory is not on `PATH`. Activate `MA-CC` before running.
- Interrupted downloads are resumable by re-running the command.
- An import error generally means Transformers is too old for
  `AutoModelForMultimodalLM`/Gemma 4, `accelerate` is absent, or the unified
  processor's import-time `torchvision` dependency is absent (even for text-only use).

Inspect the configured cache without copying weights into the repository:

```bash
python -c 'from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ["HF_HOME"])'
huggingface-cli scan-cache
du -sh "$HF_HOME"/hub/models--google--gemma-4-12B-it 2>/dev/null
```
