# Provider-free task_004 memory sweep

Exact Bayesian agents with no controller, truth classical control, or false classical control; rho = 0.60, 0.75, 0.85, 0.95, 1.00; ten episodes per condition (150 total).

Retains N=15, q=7 uniformly sampled messages, one-round message lifetime, 30 rounds, report-only grounded communication, and the existing private assignments. All conditions share ten newly generated Bayesian initial states, never the old LLM votes.

Controlled arms preserve the classical baseline: a seeded sigmoid gate (theta=0.5, beta=4) on previous-day board target support, then exactly b=3 reports on active days. The 27-fact balanced pool contains nine facts leaning toward each allocation and excludes the decisive facts. The selector is target-blind least-used/seeded-hash; target enters the gate, not fact ranking. Day 1 is HOLD under completed-day board sensing.

No inference at any stage. A mock provider is an inert placeholder; preflight and actual provider requests must be zero. The generic Cesar config-array launcher uses an isolated source snapshot, up to 15 single-CPU workers, and ten asynchronous episode slots per worker (exact decisions are CPU-bound). Full trajectories retain votes, fact IDs, posteriors, controller actions and budget for existing plots. Results stay under the declared /shared/MA-CC-results root.
