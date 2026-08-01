"""Provider-independent, versioned composition of inspectable prompt blocks."""

from .blocks import PromptBlock, RenderedPromptBlock
from .composer import PromptComposer, PromptInstance, RegexTokenCounter, TokenCounter
from .context import PromptContext
from .contracts import ResponseContract
from .examples import hiddenbench_example_context, social_conventions_example_context
from .reporting import (
    PromptMarkdownLogger,
    render_prompt_request_markdown,
)
from .registry import PromptDefinition, PromptRegistry, create_default_prompt_registry
from .versions import PromptVersion

__all__ = [
    "PromptBlock",
    "PromptComposer",
    "PromptContext",
    "PromptDefinition",
    "PromptInstance",
    "PromptMarkdownLogger",
    "PromptRegistry",
    "PromptVersion",
    "RegexTokenCounter",
    "RenderedPromptBlock",
    "ResponseContract",
    "TokenCounter",
    "create_default_prompt_registry",
    "hiddenbench_example_context",
    "render_prompt_request_markdown",
    "social_conventions_example_context",
]
