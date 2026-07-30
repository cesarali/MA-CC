# Task: Minimal Gemma 4 12B Logits and Constrained-Vocabulary Test

## Goal

Add a minimal, self-contained script that:

1. downloads and caches `google/gemma-4-12B-it` from Hugging Face;
2. loads it on the available GPU, preferably an NVIDIA A100;
3. performs one basic text-only forward pass;
4. inspects the next-token logits over the full vocabulary;
5. computes a normalized probability distribution over a constrained set of allowed answers;
6. optionally generates a very short response to confirm end-to-end operation.

This is only an infrastructure and logits-access smoke test. Do **not** implement planning, expected information gain, social games, fine-tuning, LoRA, or an agent framework yet.

## Model

Use exactly:

```text
google/gemma-4-12B-it
```

Use the instruction-tuned checkpoint. The model is multimodal, but this task should use **text input only**.

The Hugging Face model files are approximately 24 GB, so confirm that the configured cache location has sufficient free disk space before downloading.

## Existing Hugging Face cache configuration

The repository already contains a `.env` file with:

```bash
HF_HOME=/replace/with/high-capacity-storage/huggingface-cache
```

The implementation must:

- load variables from the existing `.env`;
- use the existing `HF_HOME` value;
- not hard-code another cache directory;
- not overwrite `.env`;
- not commit `.env`;
- not print Hugging Face access tokens;
- print the resolved `HF_HOME` path at startup;
- fail with a clear error if `HF_HOME` is absent or still contains the placeholder `/replace/with/...`;
- create the cache directory if the configured path is valid but does not yet exist.

Load `.env` before initializing Hugging Face objects:

```python
from dotenv import load_dotenv

load_dotenv()
```

## Hugging Face authentication

The Gemma checkpoint may require the user to accept the model license on Hugging Face and authenticate locally.

Support the standard mechanisms:

```bash
huggingface-cli login
```

or:

```bash
HF_TOKEN=...
```

Do not place a token in source code. If access fails with HTTP 401 or 403, explain clearly that license acceptance or authentication may be missing.

## Environment and dependencies

Inspect the repository's current dependency setup and add the minimum required libraries to the existing Conda and requirements files.

Likely requirements:

```text
torch
transformers
accelerate
python-dotenv
huggingface-hub
safetensors
```

Use a recent Transformers version that supports:

```python
AutoProcessor
AutoModelForMultimodalLM
```

Do not blindly downgrade an existing PyTorch/CUDA installation. Preserve existing CUDA compatibility and document any version pin needed for Gemma 4 support.

After installation, record:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

## Directory structure

Create:

```text
scripts/gemma4_logits_test/
├── README.md
└── test_gemma4_logits.py
```

Optionally add:

```text
scripts/gemma4_logits_test/run.sh
```

The script must run from the repository root.

## Main script

Create:

```text
scripts/gemma4_logits_test/test_gemma4_logits.py
```

### Command-line interface

Use `argparse`.

Support:

```bash
python scripts/gemma4_logits_test/test_gemma4_logits.py
```

and:

```bash
python scripts/gemma4_logits_test/test_gemma4_logits.py \
    --choices A B C \
    --top-k 10
```

Recommended arguments:

```text
--model
--choices
--top-k
--max-new-tokens
--temperature
--dtype
--no-generate
```

Defaults:

```text
--model google/gemma-4-12B-it
--choices A B C
--top-k 10
--max-new-tokens 32
--temperature 1.0
--dtype bfloat16
```

## Loading the model

Use the official Hugging Face interface:

```python
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

Requirements:

- load `.env` before model initialization;
- use `torch.bfloat16` by default on an A100;
- call `model.eval()`;
- use `torch.inference_mode()`;
- use `device_map="auto"`;
- print the model identifier and inferred device map;
- print GPU name and total GPU memory when CUDA is available;
- fail clearly if CUDA is unavailable unless CPU fallback is explicitly requested;
- do not load the model more than once;
- do not use quantization in this first test.

If the installed Transformers version expects `torch_dtype=` rather than `dtype=`, use the compatible keyword and document the choice.

## Test prompt

Use this simple multiple-choice prompt:

```text
You are participating in a small decision game.

Question:
Which number is larger: 7 or 3?

Choose exactly one option:
A. 7
B. 3
C. They are equal

Return only A, B, or C.
```

Use the checkpoint's chat template:

```python
messages = [
    {
        "role": "system",
        "content": "Answer using exactly one allowed option.",
    },
    {
        "role": "user",
        "content": prompt,
    },
]
```

For the Gemma 4 processor, follow the currently supported `apply_chat_template` format. Keep the test text-only.

Disable optional thinking mode with `enable_thinking=False` if supported. If the installed chat template does not support it, handle that compatibility issue cleanly.

## Test 1: Full-vocabulary next-token logits

Run a direct forward pass:

```python
with torch.inference_mode():
    outputs = model(**inputs)

next_token_logits = outputs.logits[:, -1, :]
```

Print:

- shape of `outputs.logits`;
- shape of `next_token_logits`;
- vocabulary size;
- dtype;
- device;
- minimum and maximum logit;
- whether all logits are finite.

Compute probabilities stably:

```python
next_token_probs = torch.softmax(
    next_token_logits.float(),
    dim=-1,
)
```

Print the top `k` next-token predictions with:

```text
rank
token_id
token representation
logit
probability
```

Use `repr()` for decoded tokens so whitespace is visible.

Assertions:

- batch dimension is 1;
- the vocabulary dimension is consistent with the model/tokenizer;
- logits and probabilities are finite;
- full-vocabulary probabilities sum approximately to 1.

## Test 2: Constrained answer vocabulary

Compute probabilities restricted to:

```text
A
B
C
```

### Tokenizer inspection

Do not assume each answer is represented by one token. Inspect:

```text
A
 A
B
 B
C
 C
```

Print the token IDs for each variant.

### Single-token constrained distribution

If each selected choice is exactly one token in the actual generation context:

1. collect the relevant token IDs;
2. extract their logits;
3. normalize only across them:

```python
choice_probs = torch.softmax(
    choice_logits.float() / temperature,
    dim=-1,
)
```

Print:

```text
choice
token_id
raw logit
constrained probability
```

Assert that constrained probabilities sum approximately to 1.

The expected winning choice is `A`, but treat this as a diagnostic rather than a hard scientific guarantee.

### Robust sequence-level choice scoring

Also implement robust scoring for complete choice strings, because a choice may contain multiple tokens.

Suggested interface:

```python
def score_choice_sequences(
    model,
    processor,
    prompt_inputs,
    choices: list[str],
) -> dict[str, float]:
    ...
```

For each choice:

1. append the complete candidate answer to the prompt;
2. run teacher-forced scoring;
3. sum conditional log probabilities only over answer tokens;
4. return one total log-likelihood per choice;
5. normalize scores across choices.

Conceptually:

```text
log p(choice | prompt)
    = sum_t log p(choice_token_t | prompt, previous_choice_tokens)
```

Requirements:

- exclude prompt tokens from the score;
- handle one-token and multi-token choices;
- use float32 for log-softmax calculations;
- avoid retaining computation graphs;
- print raw sequence log-likelihood and normalized constrained probability;
- assert that normalized probabilities sum approximately to 1.

If `A` is not highest, print a warning rather than failing the entire script, while still enforcing numerical correctness.

## Optional generation check

Unless `--no-generate` is supplied, generate a short deterministic answer:

```python
generated = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
)
```

Decode only newly generated tokens and print the raw response using `repr()`.

## Memory diagnostics

At startup print:

```python
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0).total_memory
```

After loading and after the forward pass print:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
```

Convert bytes to GiB.

Optionally call:

```python
torch.cuda.reset_peak_memory_stats()
```

before the forward pass. Do not clear memory before reporting peak usage.

## Error handling

Provide clear errors for:

- missing or placeholder `HF_HOME`;
- insufficient disk space;
- missing CUDA;
- insufficient GPU memory;
- missing Gemma license acceptance;
- missing Hugging Face authentication;
- unsupported Transformers version;
- interrupted model download;
- non-finite logits;
- empty or malformed choices.

Do not suppress useful tracebacks.

## README

Create:

```text
scripts/gemma4_logits_test/README.md
```

Document:

1. purpose of the smoke test;
2. exact model ID;
3. expected disk requirement of approximately 24 GB plus cache overhead;
4. expected use on an A100;
5. required Hugging Face license acceptance;
6. authentication commands;
7. use of the existing `.env`;
8. required `HF_HOME` entry;
9. dependency installation commands;
10. exact run commands;
11. meaning of full-vocabulary logits, top-k probabilities, constrained token probabilities, and sequence-level constrained probabilities;
12. expected output;
13. common failure modes;
14. how to inspect the Hugging Face cache.

Example:

```bash
conda activate <existing-environment>

python scripts/gemma4_logits_test/test_gemma4_logits.py

python scripts/gemma4_logits_test/test_gemma4_logits.py \
    --choices A B C \
    --top-k 20 \
    --max-new-tokens 8
```

## Git hygiene

Ensure these remain untracked:

```text
.env
.env.*
*.safetensors
huggingface-cache/
.cache/
```

Do not add an overly broad ignore rule. Verify with:

```bash
git status --short
```

No model weights or cache artifacts should appear.

## Validation procedure

1. Confirm `.env` is loaded.
2. Confirm `HF_HOME` points to high-capacity storage.
3. Confirm sufficient free disk space.
4. Confirm Hugging Face authentication.
5. Download `google/gemma-4-12B-it`.
6. Load the model in BF16 on the A100.
7. Run the full-vocabulary logits test.
8. Print top-k tokens.
9. Run single-token constrained scoring where valid.
10. Run robust sequence-level constrained scoring.
11. Run deterministic short generation.
12. Confirm numerical assertions pass.
13. Record allocated, reserved, and peak GPU memory.
14. Confirm no model files are tracked by Git.

## Expected report from the implementing agent

Report:

- files created;
- dependency files modified;
- exact package versions used;
- resolved `HF_HOME` path, without secrets;
- GPU model and total memory;
- model download status;
- model load dtype;
- logits tensor shape;
- vocabulary size;
- top predicted tokens;
- constrained probabilities for `A`, `B`, and `C`;
- generated answer;
- peak GPU memory;
- commands executed;
- compatibility changes required;
- whether every validation step passed.

## Acceptance criteria

The task is complete when:

- `google/gemma-4-12B-it` downloads into the configured `HF_HOME`;
- the model loads successfully on the A100;
- one text-only forward pass returns finite logits;
- top-k full-vocabulary predictions are displayed;
- constrained probabilities over `A`, `B`, and `C` are computed;
- robust full-choice sequence likelihoods are computed;
- constrained probabilities sum to one;
- deterministic short generation succeeds;
- GPU memory use is reported;
- dependencies are integrated into the existing environment configuration;
- `.env`, model weights, and cache files remain untracked.
