# Wave-2 optimisation DAG, ordered by System One (Jev)

Nodes and evidence were drafted from profiles of the merged finaliser on 2026-09-20. System One (`jev-1.13.0`)
scored each node's payoff (0 negligible … 3 large) and its risk to byte-identity (0 none … 3 high), answered whether it
was ready to implement, and chose the next node at each step among those whose dependencies were satisfied: 13 typed
calls, 28,785 input tokens, $0.0012. Execution was sequential in one session; each node was a branch gated by tests
and the byte-identity regression run (`scripts/Cygnus/analysis/regress.sh`) before any merge.

| # | node | stage it targets | expected saving | Jev payoff | Jev risk | ready | depends on |
|---|---|---|---|---|---|---|---|
| 1 | `A-classifier-pool` one persistent worker pool for the checkpoint classifier | `checkpoint_classifier` (7,019 s of the checkpoint finalize, 16 CPUs) | 4,000 s | large (2.99) | none (0.17) | 0.76 | – |
| 2 | `B-classifier-fit-overhead` trim sklearn per-fit overhead in the refit | `checkpoint_classifier` | 700 s | large (2.89) | moderate (1.99) | 0.52 | A |
| 3 | `C-joint-efficiency-draws` single-affinity bootstrap draws on index arrays | `joint_efficiency_bootstrap` (29 s b9/b15, 43 s Potsdam) | 24 s | moderate (2.02) | low (1.06) | 0.52 | – |
| 4 | `E-canonical-build-pool` build canonical tables per cell in a pool | `canonical_discovery_read` (31–71 s b9/b15) | 20 s | moderate (1.57) | none (0.48) | 0.42 | – |
| 5 | `F-table-write-threads` write scientific tables concurrently | `canonical_writing` 16 s + `table_writing` 9 s | 15 s | small (1.01) | none (0.01) | 0.76 | – |
| 6 | `G-information-residual` information stage residual | `information_resampling` (26 s b9/b15) | 10 s | small (1.01) | low (0.99) | 0.68 | – |
| 7 | `D-calibration-adapter-assembly` per-cell frames instead of 259k records through pickle | `blackboard_calibration` (35.7 s b9/b15) | 15 s | small (1.01) | moderate (2.01) | 0.48 | – |

## Results

| node | outcome | measured | landed |
|---|---|---|---|
| A classifier pool | tables identical to the previous tree (job 105) | `checkpoint_classifier` 6,954 → **3,316 s**; checkpoint finalize 7,019 → 3,382 s on 16 CPUs | this PR |
| B refit overhead | exact (A+B tables byte-equivalent to A; `_binary_log_loss` bit-identical to sklearn on 3,000 cases) but **no gain**: 3,316 → 3,299 s. The profile that motivated it ran under cProfile, which inflates Python-side validation against the C solver; unprofiled, the refit is the 65 logistic fits | not merged; branch deleted |
| C single-affinity draws | byte-identical on b9/b15 and Potsdam | `joint_efficiency_bootstrap` 29.4 → 12.3 s (b9/b15), 42.9 → 15.2 s (Potsdam); Potsdam finalize 400 → 181 s together with the derived-aggregates change | mirror #22 |
| E canonical pool | byte-identical on b9/b15 | `canonical_discovery_read` 44.9 → 34.1 s, inside a stage that swings by tens of seconds with the shared filesystem | mirror #23 |
| F threaded table writes | byte-identical but **slower**: `canonical_writing` 14.0 → 23.0 s, `table_writing` 9.2 → 13.7 s (pyarrow already threads inside each write) | not merged; branch deleted |
| G bits-bootstrap tensors | byte-identical on b9/b15; re-scoped after profiling (the cost was an int64 matmul, not the Python axis orders) | `information_resampling` 25.9 → 21.7 s | mirror #24 |
| D calibration adapter | **deferred**: at 8 workers the adapter is ≈ 11 s (0.4 slicing + 8.9 pool + 0.6 sort + 0.9 frame + 0.2 finalise) with no dominant phase; not worth its dtype-inference risk | – |

## What the ordering got right and wrong

Jev put the one large lever first and the two nodes that did not pay (F, D) last, which is the order that
wasted the least time. It rated B "large payoff" on evidence that turned out to be a profiling artefact; its
"moderate risk" call on B was cautious in the right direction (B was exact). Typed scoring is a cheap second
opinion on a drafted plan; it cannot see past the evidence it is given.

## Where the time is now

b9/b15 local finalize: 3,779 s two days ago → 263 s after wave 1 → **≈ 223 s** after wave 2. Checkpoint finalize
on 16 CPUs: 9,586 s → 7,019 s (wave 1) → **3,382 s**, of which the classifier is 3,316 s and is now bound by
the 128,640 × 65 logistic fits themselves (0.27 s per refit standalone, ~0.41 s per refit-core on 8 physical
cores with SMT). Going further there means fewer or cheaper fits (fewer permutations, a warm-started or
batched solver), which changes numbers and is a scientific decision, not an engineering one.
