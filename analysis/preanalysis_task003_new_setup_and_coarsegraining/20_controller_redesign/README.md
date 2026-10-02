# 01_design — the controller redesign

[`symmetric_controller_and_coarse_graining.md`](symmetric_controller_and_coarse_graining.md)
is the design note. It covers five points:

1. agents' facts stay untouched (they hold the only complete proof route);
2. **two symmetric controller pools** — 12 facts each, magnitude-matched, disjoint,
   neither able to prove its own target;
3. a second experiment pair with the decisive facts removed from agents;
4. whether a two-allocation world is possible;
5. the epistemic coarse-graining — proposed here, tested in `../02_markov_tests/`.

Pool membership is in `../../game_analysis/symmetric_pools.json`.

The problem it solves: the executed study admits controller facts by a rule
evaluated once for the **false** target and reused for both configurations, so
both controllers quote the same 24 decoy-favouring facts and differ mainly in how
often the activation gate fires (32% vs 54%).
