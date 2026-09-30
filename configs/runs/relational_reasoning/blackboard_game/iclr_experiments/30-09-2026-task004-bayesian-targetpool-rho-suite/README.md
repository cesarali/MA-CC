# Provider-free Bayesian task_004 target-pool comparison

100 new episodes: truth and false classical controllers at rho=0.60, 0.75,
0.85, 0.95 and 1.00, ten paired episodes per condition. Existing no-controller
and balanced-27 controller results are retained as references, not rerun here.

The sole scientific change versus `30-09-2026-task004-bayesian-classical-rho-suite`
controlled arms is `controller_fact_pool_mode: balanced_target`: nine facts from
the existing audited balanced pool labelled as leaning toward A0 for truth or
A2 for false. Some audit lean labels resolve posterior ties. Decisive facts
remain excluded. Fact ranking within each menu remains least-used, previous-board
occurrence, then seeded hash; the target affects the eligible menu and the gate.

Retains N=15, q=7, report-only, one-day messages, 30 days, exact Bayesian agents,
full previous-day controller board sensing, sigmoid beta=4/theta=0.5, day-1 HOLD,
exactly b=3 posts on active days, fresh per-round budget, deterministic authoring.
No LLM or provider inference, including initialization. Reuses the exact ten
Bayesian initialization artifacts and seeds from the balanced-27 suite.

Controlled arms keep dawn forgetting, including before day 1. Existing
no-controller baselines retain end-of-day forgetting; comparisons against them
at rho<1 retain that known initial-information/timing limitation.

Generic Cesar config-array launcher: ten single-CPU/4G workers, two-hour limit.
Ten async episode slots per worker do not imply ten CPU processes; exact
updates are CPU-bound. Expected duration about 60–90 minutes at prior observed
pace, with zero API requests/tokens/spend. Source staged to an isolated checkout.
