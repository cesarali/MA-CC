# Relational Population Study 09c

Study 09c is a coarse, descriptive exploration of finite epistemic persistence.
It is not a probability estimate and does not use a matched theoretical curve.

## Exact design

- Population `N=12`
- Supporting chain length `L=3`
- Initial supporting-fact redundancy `r=3`
- Frozen task `task_0002` only
- Truth `NORTH`
- False controller target `NORTHWEST`
- Ordinary social slots `q=2`
- Controller sensor size `q_c=6`
- Population rounds `30`
- Receiver `naive`
- Controller evidence `strategic`
- Message mode `recommendation_plus_fact`
- Soft schedule with `beta=4.0` and `theta=0.75`
- One repetition
- Persistence `rho={0.6,0.8,0.9}`
- Budget `b={3,6,9,12}`, giving `b/N={0.25,0.50,0.75,1.00}`

The grid contains `3 x 4 = 12` cells and exactly 12 episodes. At a controlled
`q=2` update, one social slot is the controller and one remains an ordinary
peer. The runtime never creates two controller slots.

## Persistence semantics

Historical `known_fact_ids` retain every valid fact an agent has received.
Active `active_fact_ids` are the facts available to its current prompt and to
ordinary peer transmission. Persistence is applied once after each full
population round. A later valid exposure can reactivate an inactive fact.

## Analysis

The full 30-round trajectory is retained under `results_only`. The descriptive
endpoint table reports the requested final and late-time quantities. “Late-time”
means one-based rounds 21 through 30; it does not claim stationarity.

The retained knowledge strata directly provide:

- `P(truth vote | active full proof)`;
- `P(truth vote | not active full proof)`.

They do not separate the two wrong semantic options within each proof stratum.
Therefore `P(false target | active full proof)` is not exactly recoverable from
the current compact round records. The runtime was not changed solely to add
that diagnostic.

`round_target_information_fraction` is the existing normalized information
fraction requested as `eta_IF`. The recipe also requests empirical
susceptibility, `eta_ir`, `eta_th`, effective affinity, kinetic compliance,
and current observables. These are empirical calculations from the trajectory.
The current non-persistence theory is disabled with `theoretical_reference: none`.

## Preflight only

The strict contract checks every requested scientific coordinate and rejects an
extra task, repetition, rho value, budget, receiver, target, schedule, or
controller semantic before submission. No provider-backed run is launched by
preflight.
