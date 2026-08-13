# First LLM Pilot Grid

## Common setup

- Population size: `N = 24`
- Rounds per episode: `10`
- Repetitions per cell: `30`
- Prefer the same HiddenBench tasks and random seeds across all cells.
- Suggested pilot composition: `3 fixed tasks × 10 seeds = 30 episodes/cell`.
- Controller acts once per round.
- If `U_k = ADVOCATE`, exactly `b` of the `N` microscopic update positions are selected uniformly at random in advance.
- At a controlled position, one of the `q` social-input slots is replaced by the controller advocacy input.

The main observables should include:
- final and round-wise `m_truth`,
- `m_ctrl`,
- truth/target current and activity,
- controller action entropy / overlap diagnostics,
- round-level controller-to-truth CMI / transfer entropy,
  `I(U_k ; n^{truth}_{k+1} | n^{truth}_k)`,
- round-level controller-to-target CMI / transfer entropy,
  `I(U_k ; n^{ctrl}_{k+1} | n^{ctrl}_k)`.

For the first LLM pilot, **do not use the full three-option population-state CMI**. The projected truth/target channels are the primary information-flow quantities.

---

## Grid A — Onset of control

| Parameter | Values |
|---|---|
| `q` | `{1, 2}` |
| `b` | `{0, 6}` |
| fixed | `q_c = 2`, target = truth |

Purpose: establish the no-control baseline and test whether modest actuation already changes the population dynamics.

---

## Grid B — Control strength

| Parameter | Values |
|---|---|
| `q` | `{1, 2}` |
| `b` | `{12, 18}` |
| fixed | `q_c = 2`, target = truth |

Purpose: probe stronger actuation and look for nonlinear or threshold-like control behavior as `b/N` increases.

---

## Grid C — Sensing versus actuation

Use **three sensing levels**:

| Parameter | Values |
|---|---|
| `q_c` | `{2, 16, 24}` |
| `b` | `{6, 12}` |
| fixed | `q = 2`, target = truth |

Purpose: separate the effect of better sensing from the effect of stronger actuation.

This grid contains **6 cells** rather than 4. Therefore, together with Grids A, B, and D below, the pilot contains **18 cells total**.

---

## Grid D — Reverse the controller target

| Parameter | Values |
|---|---|
| `q` | `{1, 2}` |
| controller target | `{truth, decoy}` |
| `b` | `{6, 12}` |
| fixed | `q_c = 12` |

For the wrong-target condition, use the HiddenBench **shared-information decoy** rather than an arbitrary incorrect option.

Purpose: test whether the controller has genuinely bidirectional leverage, and whether that reversibility of influence depends on the social interaction size `q`.

This is an **8-cell grid**:
- `q ∈ {1, 2}`
- target ∈ `{truth, decoy}`
- `b ∈ {6, 12}`

A convincing result would be:
- control increases `m_ctrl` toward whichever target it is assigned,
- `m_truth` increases under truth-target control,
- `m_truth` decreases under decoy-target control,
- and the magnitude of these effects can be compared directly between `q = 1` and `q = 2`.

---

## Pilot size

- Grid A: `4` cells
- Grid B: `4` cells
- Grid C: `6` cells
- Grid D: `8` cells
- Total: `22` cells
- At `30` repetitions/cell: `660` episodes

If the experiment budget must be reduced, **do not shrink Grid D first**. Grid D is the strongest causal test of whether the controller genuinely steers the population rather than merely helping truth convergence. A better reduction would be to trim Grid B or reduce Grid C to fewer sensing levels.
