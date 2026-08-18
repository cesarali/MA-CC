"""Does possessing the supporting facts decide whether a model solves the task?

The relational task generator produces tasks whose answer is supposed to follow
from a chain of exactly ``L`` designated supporting facts.  Before those tasks
carry a multi-agent experiment, that premise has to be *measured* rather than
assumed.  This benchmark measures it directly:

    A_k = P(correct | k of the L supporting facts are shown),   k = 0 .. L

If the generator's tasks are well posed then ``A_L`` is high and ``A_k`` for
``k < L`` sits at chance, because a strict subset of the chain leaves the query
displacement genuinely undetermined - a fact this package verifies
*symbolically*, on every single prompt, before any model is asked anything.

There is no agent, no vote, no controller and no peer anywhere in this package.
One task plus one evidence condition equals one prompt equals one answer.
"""

from .conditions import EvidenceCondition, build_evidence_conditions
from .config import BenchmarkConfig, ParameterCondition, load_benchmark_config
from .prompting import BenchmarkPrompt, parse_answer, render_prompt
from .validation import PromptValidationError, validate_condition_prompts

__all__ = [
    "BenchmarkConfig",
    "BenchmarkPrompt",
    "EvidenceCondition",
    "ParameterCondition",
    "PromptValidationError",
    "build_evidence_conditions",
    "load_benchmark_config",
    "parse_answer",
    "render_prompt",
    "validate_condition_prompts",
]
