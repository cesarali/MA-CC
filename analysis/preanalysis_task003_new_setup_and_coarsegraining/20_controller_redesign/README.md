# 20_controller_redesign — the controller redesign

> **Status, 5 October 2026.** This is the 2 October design note. What became of
> each part:
>
> | part of the note | status |
> |---|---|
> | §2, the two symmetric controller pools | **frozen** as the controller pools of `task003-symmetric` |
> | §1, "agents' facts stay as they are" | **superseded**: task003-symmetric now has new agents, 8 per allocation |
> | §3, agents without the decisive facts | became `task003-nosolution`, with 24 agents and its own single pool |
> | §5, the coarse-graining | tested in `../30_coarse_graining/` on the archived study |
>
> The frozen setups: [`../50_next_experiment/designs/README.md`](../50_next_experiment/designs/README.md).

[`symmetric_controller_and_coarse_graining.md`](symmetric_controller_and_coarse_graining.md)
is the design note. It covers five points:

1. agents' facts stay untouched (they hold the only complete proof route);
2. **two symmetric controller pools** — 12 facts each, magnitude-matched, disjoint,
   neither able to prove its own target;
3. a second experiment pair with the decisive facts removed from agents;
4. whether a two-allocation world is possible;
5. the epistemic coarse-graining — proposed here, tested in `../30_coarse_graining/`.

Pool membership is in [`../10_task_and_facts/symmetric_pools.json`](../10_task_and_facts/symmetric_pools.json). These two pools are the controller pools of the frozen **task003-symmetric** setup; see [`../50_next_experiment/designs/README.md`](../50_next_experiment/designs/README.md).

The problem it solves: the executed study admits controller facts by a rule
evaluated once for the **false** target and reused for both configurations, so
both controllers quote the same 24 decoy-favouring facts and differ mainly in how
often the activation gate fires (32% vs 54%).
