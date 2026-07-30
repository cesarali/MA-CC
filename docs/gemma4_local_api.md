# Local Gemma 4 API

The repository exposes the cached `google/gemma-4-12B-it` checkpoint through
`GemmaLocalAsyncLLMClient`. The client has the same asynchronous interface used
by the games, but performs inference locally with Transformers on one GPU. It
does not start an HTTP server: "API" here means the application's Python client
interface.

## Requirements

- Activate the `MA-CC` Conda environment so its CUDA tools, including `ptxas`,
  are on `PATH`.
- Run from the repository root.
- Keep an ignored `.env` containing `HF_HOME` pointed at the Hugging Face cache.
- The configured checkpoint is fixed to `google/gemma-4-12B-it` and CUDA is
  required by default.

On the validated host the environment is located at:

```bash
export PATH=/home/ojedamarin/.local/share/miniforge3/envs/MA-CC/bin:$PATH
```

Normal `conda activate MA-CC` is equivalent and is preferable in an interactive
shell. Merely invoking the environment's Python by absolute path is insufficient
when its `bin` directory is not also on `PATH`, because Triton must find `ptxas`.

## Call the API directly

`complete_decision()` is the game-facing method. It scores every allowed choice
at the actual assistant continuation boundary, normalizes the sequence
log-likelihoods over that allowed set, and selects an authoritative action.

```python
import asyncio

from naming_game import GemmaLocalAsyncLLMClient


async def main() -> None:
    client = GemmaLocalAsyncLLMClient()
    try:
        response = await client.complete_decision(
            [{"role": "user", "content": "Choose the larger number: A. 7 B. 3"}],
            choices=["A", "B"],
            output_format="choice_reason",
            choice_temperature=1.0,
            selection_policy="argmax",
            generation_temperature=0.0,
            max_reason_tokens=32,
            seed=7,
        )
        print(response.content)
        for score in response.scores:
            print(score.choice, score.token_ids, score.log_likelihood, score.probability)
        print(response.usage)
        print(client.diagnostics)
    finally:
        client.close()


asyncio.run(main())
```

Use `output_format="choice_only"` in experiments that need only the selected
action. This avoids rationale generation while retaining token IDs,
log-likelihoods, and normalized choice probabilities in `response.scores`.
Use `complete()` only for unconstrained text generation.

The public decision response contains choice log-likelihoods rather than the
entire vocabulary tensor. To inspect raw next-token logits, tensor shape,
finiteness, and top tokens, run the dedicated logits diagnostic.

## Test the checkpoint and public API

Run the CPU contract tests before loading the model:

```bash
pytest
```

Then run the two opt-in GPU diagnostics:

```bash
python scripts/gemma4_logits_test/test_gemma4_logits.py
python scripts/gemma4_api_test/test_internal_api.py
```

The first script proves that the checkpoint is on CUDA and returns finite
full-vocabulary logits. The second uses only the public client API and verifies
single-token and multi-token scoring, normalized probabilities, action
selection, optional rationale generation, token accounting, runtime reuse, and
GPU memory diagnostics.

## Run a small game

The checked-in smoke config runs one neutral convention-game episode with three
agents for three population rounds. Each round has three pair interactions and
each interaction asks both players, for nine interactions and 18 constrained
Gemma decisions:

```bash
python -m naming_game.cli experiment \
  --config configs/empowerment_gemma4_3_agents_3_rounds.yaml \
  --no-resume \
  --output-dir results/gemma4_3_agents_3_rounds
```

Results are written under the ignored `results/` directory. The primary files
are `interactions.parquet`, containing each agent decision and its scored choice
distribution, and `episodes.parquet`, containing episode-level outcomes.

For the simpler binary Naming Game runner, Gemma is also selectable directly:

```bash
python -m naming_game.cli run \
  --provider gemma_local \
  --update-mode synchronous_parallel \
  --num-agents 3 \
  --rounds 3 \
  --reasoning-fraction 0 \
  --concurrency 1 \
  --output-dir results/gemma4_basic_3x3
```

Local inference is serialized, so `request_concurrency`/`--concurrency` must be
`1`. Do not enable provider fallback for validation runs: a fallback could make
a successful command appear to be a local Gemma result when it was not.
