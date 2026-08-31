# Study 09k provider-free preflight

Preflight date: 2026-08-31. **No study or provider job was submitted.** Static
status is permitted, but submission is blocked until the shared 20-state
initialization bundle is generated and audited. Study 09k is the truth-target
counterpart of Study 09j and the N=24 successor to Study 09g.

## Resolved design

- Deliberate epistemic change: `naive_strategic` -> `vigilant_strategic`
- `N=24`, `q=1`, `L=2`, 30 rounds
- `rho={0.75,0.85,1.00}`; `b={6,8,12,16,18,24}`
- 18 cells x 20 repetitions = 360 episodes
- Truth and controller target: `SOUTHWEST`
- `q_c=12`, preserving the 09g sensing fraction `6/12=12/24=0.5`
- `beta=4.0`, `theta=0.75`, soft schedule
- `recommendation_plus_fact`, strategic evidence
- Strategic evidence is real frozen fact `f2`, “Zani is southwest of Pelo.”
- Controlled q=1 updates contain one controller slot and zero ordinary peers.

Dataset choice and its explicit difficulty mismatch are identical to Study
09j: frozen `pop24_L2_r06/task_0002`, with `r=6`, support coverage 1/4,
four distractors, `K=3`, and no single-agent solution. No frozen r=8 dataset
exists to exactly preserve Study 09g’s 1/3 support coverage.

The target does not enter local-vote initialization, so Study 09k reuses the
same complete state artifact as Study 09j for every repetition. Strict final
validation compares the vote vector, active and historical knowledge, and the
complete physical-state hash across all cells. The real `x_0` range remains
unknown until the approved one-time initialization run is complete.

## Provider, tokens, runtime, and scheduler

Study 09k dynamics: 259,200 nominal, 272,880 expected, and 518,400
conservative calls. Input-token estimates are 120,528,000; 126,889,200; and
241,056,000. Output-token bounds are 259,200; 1,117,716,480; and
2,123,366,400. The shared initialization cost is paid once for the pair, not
again here: 480 nominal, 520 expected, and 960 conservative calls.

Static serial-equivalent runtime is 22.74 hours. Expected call time is about
9.10 hours at 500 RPM or 17.06 hours at the 266.7-RPM latency-plan ceiling,
before queueing/outages. Proposed execution is 18 generic cell shards,
`0-17%2`, 10 CPUs, 12 GB, 20 hours, with shared adaptive provider control.
Run sequentially with Study 09j.