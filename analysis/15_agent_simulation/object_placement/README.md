# 15_agent_simulation / object_placement

The MuSR Object Placements puzzle (where Casey would look for the keys; A0 = drawer is correct, A2 = desk is the false
controller's target). World: `../../object_placements_llm_free/world/world_op1.json` (34 true facts, 18 of them
neutral when read alone).

- [`symmetric/`](symmetric/): 15 agents whose facts jointly prove A0; the original A0 and A2 controller pools.
- [`no_solution/`](no_solution/): 15 agents whose facts are jointly uniform; one uniform controller pool.

The 24-agent predecessors are in `../../object_placements_llm_free/experimental_setup/` (frozen; historical).
