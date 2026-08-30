# Relational Population Study 09d

Study 09d refines the finite-persistence transition suggested by the preserved,
partial Study 09c scout. It varies only active epistemic persistence `rho` and
controller budget `b/N` on frozen `task_0002`.

The exact grid is `rho={0.70,0.75,0.80,0.85,0.90}` by `b={3,6,9,12}`, with
ten deterministic repetitions per cell: 20 structural cells and 200 episodes.
All production game, q=2 controller-slot, strategic true-evidence, active-fact
persistence, prompt, and validation semantics are inherited unchanged from
Study 09c.

Rounds 21--30 are described only as **late-time**. Active `phi` and `kappa` are
the primary epistemic observables; historical versions remain diagnostics.
The current q=1 non-persistence theory is inapplicable, so
`theoretical_reference: none`.

## Analysis notation

- paper susceptibility `chi` maps to the repository's empirical
  `susceptibility_occupancy_weighted`, derived from
  `round_target_susceptibility`;
- information efficiency maps to the existing
  `round_target_information_fraction` (also called `eta_IF` in study prose);
- `eta_ir` maps to the existing derived `eta_ir_state_local` family;
- `eta_th` maps to the existing derived `eta_th` observable.

No new information estimator or theoretical curve is introduced. Endpoint
tables retain raw episode classifications and cell-level `k/10` counts plus
descriptive distribution summaries.
