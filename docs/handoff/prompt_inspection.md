# Prompt inspection and Markdown logging

`mas_cc` binds immutable game-owned `FullPrompt` objects without loading or
calling an LLM provider. Each semantic `PromptBlock` owns its value,
validation, role, version, and renderer. `FullPrompt.compile()` produces the
only prompt-layer artifact used to create a provider-neutral request.

`UNBOUND` is distinct from `None` and empty values. A required unbound block
fails compilation, an optional unbound block is omitted and recorded, and a
bound empty memory renders normally. `bind()` always returns a new object.

The registered concrete prompt owns block order. Prompt schema version 2 YAML
selects family/version and permitted presentation policy; it does not repeat or
reorder the blocks.

## Generate the paper examples

From the repository root:

```bash
conda run -n MA-CC mas-cc prompt examples \
  --output-dir inspection/paper_prompts
```

The default reads the checked-in 32-agent HiddenBench exact-replication data.
It writes five readable requests:

- one social-conventions decision;
- the first HiddenBench discussion speaker;
- a later HiddenBench discussion speaker with the public transcript;
- a HiddenBench pre-discussion vote;
- a HiddenBench post-discussion vote.

Read `inspection/paper_prompts/all_requests.md` to see every request in one
document, or open an example's `request.md` for one interaction. The messages
under **Exact messages sent to the LLM** are shown verbatim and in transmission
order. The JSON files beside them are retained for machine checks.

No LLM is called by this command. HiddenBench's `correct_answer` field is audit
data and is not copied into a bound prompt.

## Learn how to add a prompt for a new game

Open `notebooks/tutorial_create_full_prompt_new_game.ipynb`. It defines six new
concrete blocks and a `PrivateSignalChoiceFullPrompt`, demonstrates immutable
two-agent binding, omission, token counts and fingerprints, constructs one
normalized request, and provides separate University and OpenAI preflight and
completion cells. Its non-network path is executable without credentials.

## Log a compiled interaction during a run

The Markdown logger writes one request per interaction:

```python
from mas_cc.llm_runtime.prompts import PromptMarkdownLogger

logger = PromptMarkdownLogger("run_artifacts/prompts")
logger.log(
    compiled_prompt,
    "interaction-000042-agent-7",
    metadata={"interaction": 42, "agent": 7},
)
```

This produces:

```text
run_artifacts/prompts/interaction-000042-agent-7.md
```

Existing logs are not overwritten by default. Pass `overwrite=True` when
constructing the logger only for intentionally reproducible inspection output.
