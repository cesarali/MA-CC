# Prompt inspection and Markdown logging

`mas_cc` composes prompts without loading or calling an LLM provider. The
compiled `Message` sequence is the provider-independent request that later
adapters receive.

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
data and is not copied into the prompt context.

## Log a compiled interaction during a run

The Markdown logger writes one request per interaction:

```python
from mas_cc.prompts import PromptMarkdownLogger

logger = PromptMarkdownLogger("run_artifacts/prompts")
logger.log(
    prompt_instance,
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
