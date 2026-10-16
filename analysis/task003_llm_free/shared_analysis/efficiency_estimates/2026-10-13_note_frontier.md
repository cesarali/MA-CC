# Success against actual cost — findings

**13 October 2026.** Plan: [`2026-10-13_plan_frontier.md`](2026-10-13_plan_frontier.md). Code: `frontier.py`
(tests: `tests/test_frontier.py`, the handoff's synthetic cases, all pass), `frontier_figures.py`.
Tables and figures (not in git): `results/efficiency_estimates/2026-10-13/` — `frontier_points.csv`
(every configuration, outcome, rule, time; Δ with 95% intervals, cost, η's), `frontier_thresholds.csv`,
`frontier_downsample.csv`, `frontier_{target,task}_T{30,5}.png`. Numbers below: q 6, plurality rule.

## Results

**1. In nosolution, success saturates with very little control.** At t = 30 every configuration
reaches about the same gain (Simulation 2, ρ 0.75: target attainment +0.28 to +0.33 for the false
controller, +0.62 to +0.65 for the truth controller), whatever it spent (7 to 30 facts). The cheapest
tested configuration — slowest rate, gate on — is on the frontier and reaches every threshold up to
+0.3 (6.9 facts in Simulation 2; 8.7–10 in Simulation 1). Most others are dominated: median
η_cost 0.26 (false) and 0.48 (truth); the least efficient spend four times more (η_cost 0.23).
Simulation 1 is the same, with larger waste (η_cost down to 0.03–0.09 at b = 6–9).

**2. Earlier, speed is worth paying for, with diminishing returns.** At t = 5 (Simulation 2,
nosolution) the gain rises with the rate: false controller +0.17 at 2 facts, +0.39 at 18–30 facts;
truth +0.32 to +0.60 — a concave frontier: the first facts buy most. By t = 10 the slow controllers
have caught up. Efficiency therefore depends on when success is scored.

**3. Manipulating a swarm that owns a proof has increasing returns.** Symmetric, false controller:
the gain grows faster than linearly with facts (Simulation 1 up to +0.88 for 173 facts; Simulation 2
up to +0.20 at the 30-message cap). The frontier is convex, so randomised mixtures beat every tested
configuration (dashed above solid: reaching +0.1 costs 27 facts with the cheapest pure configuration,
15 with a mixture of no control and a strong one in Simulation 2) — concentrating effort in some runs
pays. Most configurations lie on the tested-point frontier (median η_cost 0.99): the saturation the
handoff predicts for a monotone sweep.

**4. Manipulation costs correctness one for one.** For the false controller, the fall in task
correctness equals the gain in target attainment (median ratio 0.96–1.00): the decision is a contest
between A0 and A2; A1 almost never wins.

**5. Ceilings and harm.** Symmetric with the truth controller: silence already gives A0 the plurality in
99% of runs (100% with full memory), so the gain is at most +0.013 — nothing to measure. Simulation 1,
nosolution, large b: negative gains (false controller down to −0.47 at ρ 1), harmful configurations,
efficiency not applicable (the backfire again, now in success terms).

**6. Precision at 100 paths:** Δ ±0.017 (Simulation 2), ±0.04 (Simulation 1); η_cost ±0.02–0.035 where
defined. The split-half selection check gives medians close to in-sample (false controller 0.26 vs
0.26); it is undefined for the truth controller where no configuration on one half reaches the other
half's best gain.

## Interpretation and what to sample next

- Under this success measure, the main inefficiency is **spending after the decision is settled**:
  in nosolution the plurality is decided within about ten time units, and later messages change
  nothing. The time of scoring is part of the question.
- **The frontier is not identified below the cheapest tested cost** (about 7 facts in Simulation 2):
  the dashed hull from (0, 0) assumes mixing with no control. To see the real low-cost frontier,
  sample smaller budgets (B = 1, 2, 3, 5 messages) and lower rates — the handoff's §12.6.
- For manipulation against a proof (symmetric, false), increasing returns mean the efficient policy
  is concentrated (strong control in fewer runs, or bursts), which the current policy family does not
  include; a burst controller would be the natural next policy to test.
