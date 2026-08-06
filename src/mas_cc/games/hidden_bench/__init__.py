"""HiddenBench: Hidden Profile collective-reasoning tasks, two protocols.

Li, Naito & Shirado, *Systematic Failures in Collective Reasoning under
Distributed Information in Multi-Agent LLMs* (ICML 2026).

| Game | Protocol |
| --- | --- |
| `hidden_bench_vanilla` | The paper's own: N agents, round-robin plenary discussion, a vote before and after. |
| `hidden_bench_naming` | Dyadic: private pairs with private per-partner memory, on the expanded (N > 4) populations. |

The corpus and its N > 4 expansions are **not** produced here - they come from
`scripts/local_llms/hiddenbench_population_pipeline/`. See
`docs/hidden_bench/data_provenance.md` for what lives where and
`docs/hidden_bench/README.md` for how to run either game.
"""

from .data import DEFAULT_CORPUS_ROOT, SCHEMES, assign, load_task_set
from .records import (
    COMMIT,
    DISCUSS,
    EXCHANGE,
    POST_VOTE,
    PRE_VOTE,
    HiddenBenchAgentState,
    HiddenBenchGameState,
    HiddenBenchTransition,
)
from .runtime import run_hidden_bench_game, run_hidden_bench_game_sync
from .schemas import AgentInfoSet, HiddenBenchDataError, HiddenProfileTask

__all__ = [
    "COMMIT",
    "DEFAULT_CORPUS_ROOT",
    "DISCUSS",
    "EXCHANGE",
    "POST_VOTE",
    "PRE_VOTE",
    "SCHEMES",
    "AgentInfoSet",
    "HiddenBenchAgentState",
    "HiddenBenchDataError",
    "HiddenBenchGameState",
    "HiddenBenchTransition",
    "HiddenProfileTask",
    "assign",
    "load_task_set",
    "run_hidden_bench_game",
    "run_hidden_bench_game_sync",
]
