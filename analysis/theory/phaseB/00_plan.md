# Phase B — attribution of the M3 → M4 gain (written 24 October 2026, before M3s is run)

**Question.** M4 differs from M3 in two ways at once: (i) every agent reads one shared, realised board instead of its own
sample of the expected board; (ii) one stochastic controller acts instead of the expected controller. Which change
produced M4's gain (held-out G error 0.12 → 0.027 on task003)?

**New model M3s** (`theory/models/mean_field.py`, `closure="M3s"`): M3 exactly, except that the controller is one
stochastic controller, as in M4. It acts with probability λ_c dt per step, reads q_c agent posts from one realisation of
the expected agent-only board, applies the simulator's rule, and adds one whole post. M3s differs from M4 only in (i).
Regression check: M3 itself reproduces its stored output exactly after the change (max difference 0.0).

**Runs** (same sets and settings as M3/M4): task003 `dev`, `all`, `post_stop` (40 realisations, R = 30); Object
Placements `op` (20, R = 20) and `op_post_stop` (40, R = 20).

**Status:** post-hoc. All data are known, and M3s is not used to choose anything. Its scorecards are written with the
existing scoring code (`held_out_validation/score.py`, `score_post_stop.py`, and the Object Placements scoring
functions), so they are directly comparable to M3 and M4.

**Attribution measure, fixed now.** For each test with a G error (task003 held-out; Object Placements reversal), the
share of the M3 → M4 improvement that M3s already achieves:

  f_controller = (err_M3 − err_M3s) / (err_M3 − err_M4).

- f ≈ 1: the stochastic controller explains the gain.
- f ≈ 0: the shared board explains it.
- In between: both contribute.
- Outside [0, 1]: the changes interact.

Also reported, without a single summary number:
- resolved signs of D;
- crossover brackets;
- the post-stop ordering, half-times and removal-residual signs.

**Consequence.** The theory reports say M4's gain comes from the shared board, with an addendum noting that this is
not yet attributed. The Phase B result replaces that statement in a new dated addendum, whichever way it comes out.
