# 15_agent_simulation / task003

The task003 allocation puzzle (three workers, two tasks, three candidate allocations; A0 is correct, A2 is the false
controller's target). World: `../../task003_llm_free/llmfree_core/world_task003.json` (49 true facts).

- [`symmetric/`](symmetric/): 15 agents whose facts jointly prove A0; the original A0 and A2 controller pools.
- [`no_solution/`](no_solution/): 15 agents whose facts are jointly uniform; one uniform controller pool.

The 24-agent predecessors are in `../../task003_llm_free/experimental_setup/` (frozen; historical).
