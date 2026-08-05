"""Provider-independent immutable prompt kernel.

Ships no game- or paper-specific prompt content; :func:`create_default_prompt_registry`
returns an empty registry that callers populate with their own definitions.
"""

from .blocks import UNBOUND, PromptBlock, RenderedPromptBlock, Unbound
from .compiled import CompiledPrompt
from .contracts import ResponseContract
from .fingerprints import canonical_json, fingerprint
from .full_prompt import CompilablePrompt, FullPrompt
from .registry import PromptRegistry, create_default_prompt_registry
from .reporting import PromptMarkdownLogger, render_prompt_request_markdown
from .tokenization import RegexTokenCounter, TokenCounter
from .versions import PromptVersion

__all__ = [
    "UNBOUND",
    "CompiledPrompt",
    "CompilablePrompt",
    "FullPrompt",
    "PromptBlock",
    "PromptMarkdownLogger",
    "PromptRegistry",
    "PromptVersion",
    "RegexTokenCounter",
    "RenderedPromptBlock",
    "ResponseContract",
    "TokenCounter",
    "Unbound",
    "canonical_json",
    "create_default_prompt_registry",
    "fingerprint",
    "render_prompt_request_markdown",
]
