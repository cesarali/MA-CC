# 50_next_experiment — planning the symmetric controller study

**Everything here is PRELIMINARY.** No configuration is frozen and no run has
been launched.

## Read in this order

| # | file | what it is |
|---|---|---|
| 1 | [`prior_art_2026-10-02.md`](prior_art_2026-10-02.md) | What already exists: Darius's balanced controller pool and `task_004` studies on `darius-MA-v1`, plan v0.4, and how the two externally-written files in `../10_task_and_facts/` fit |
| 2 | [`design_preliminary_2026-10-02.md`](design_preliminary_2026-10-02.md) | The proposed design, the controller recommendation, and the sample-size calculation |
| 3 | [`protocols_2026-10-02.md`](protocols_2026-10-02.md) | **The three protocols** — agent comms, controller comms, controller budget — the full option surface, what Darius set, and why `darius-MA-v1` must be merged |
| 4 | [`designs/`](designs/) | **The three concrete setups** — v0 reference, symmetric-v2, nosolution-v2 — with fact lists, and what slot lean means |
| 5 | [`builders/`](builders/) | The search that produced them, and the scarcity constraint that bounds it |

## The short version

**Most of this is built already.** `darius-MA-v1` has a frozen *balanced*
27-fact controller pool for `task_004` — 9 facts leaning to each allocation,
strength-matched by Wasserstein-1 distance — plus a `balanced` runtime mode and
twelve study directories at 30 rounds crossing persistence, budget protocol and
communication profile. Reuse it; do not rebuild it.

**One balanced pool beats two symmetric pools.** With a single pool the
controller's menu is identical whichever target it is given, so any directional
difference is the *selection policy*. Two target-specific pools confound menu
with policy. This supersedes our draft `symmetric_pools.json`.

**The truth/decoy asymmetry cannot be removed.** The balanced pool as a whole
proves A0 (joint posterior 1.000); its 9 A2-leaning facts reach only 0.933. All
facts are true statements about one world, so any large enough set of them is
consistent only with worlds near the truth. Three independent analyses agree.
The comparison must be reported as *correction versus selective persuasion under
the world's inherent asymmetry*, not as symmetric steering.

**Keep the decisive-fact factor, but it is not new.** `task_004` *is* `task_003`
with the decisive facts removed — its `task.json` says so. It is still worth
keeping because Darius removed them together with their holders (24 agents → 15),
confounding proof availability with population size. Both corrected setups, at a
fixed 24 agents, are in [`designs/`](designs/).

**30 rounds, and push for 100 episodes per cell.** 15 rounds truncates the path
divergence while it is still growing. And the bootstrap resamples whole
episodes, so at v0.4's initial 50 the confidence interval on the headline
quantity lands near ±33% — extrapolated from the 18% we measured at 168
episodes. v0.4 already permits extension to 100; the calculation is in
`design_preliminary_2026-10-02.md` §3.

**Recommended first controller:** the deterministic per-round one (`b = 3`), not
the LLM episode-budget one. It is the configuration our exact path-KL estimator
is most likely to survive, and it is the reference the LLM controller needs. This
inverts v0.4's priority order and needs agreeing with César.

## Blocking questions

0. **`darius-MA-v1` must be merged before anything can be configured.** The
   options plan v0.4 requires — a full-episode budget scope, and a
   target-independent controller menu — do not exist on `dev/rsanchez`. See
   [`protocols_2026-10-02.md`](protocols_2026-10-02.md) §1 and §4.
1. Does v0.4's "50 episodes per cell" mean 50 **initializations** or 50
   trajectories? The initialization is the only independent unit, so the two
   readings differ by a large factor in effective sample size.
2. Deterministic controller first, or LLM controller first?
3. `task_003` or `task_004`? (We lean `task_004`.)
