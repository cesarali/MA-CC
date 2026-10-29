# Deduplicated reading as a defence: pre-registration

29 October 2026. **Author: Claude.** Written before any run. Requested by the researcher (2026-10-29). Earlier plans:
`task003_llm_free/simulation_2/key_phenomena_hypothesis/00_theory_plan_2026-10-21.md`, line 192, and
`.../defense/README.md`.

## The rule

- **Standard reading:** an agent draws q posts from its readable window, so a fact is read in proportion to its number
  of copies.
- **Deduplicated reading:** the agent draws q distinct facts, uniformly among the distinct facts present in its window.
  Each background post (finite model) counts as its own item. Everything else is unchanged, including q, the window,
  posting and the controller.

## Why the theory expects it to protect

The controller's advantage in the finite fact-flow model comes from copy number:
- its few facts carry many copies each, so they are read often and survive;
- the swarm's many facts carry few copies each and die out (asymmetric extinction; copies per fact is the system size).

Deduplicated reading makes the reading probability of a fact depend on its presence, not its copy number. Two
consequences:
- in the mean field, the controller's board share no longer raises the chance of reading a given controller fact;
- in a finite swarm, rare native facts are read as often as common controller facts while present, which slows their
  extinction.

## Stage 1: finite fact-flow model (engine-free)

**Settings:** `phenomenological_fact_flow/finite_model.py`, P0, ε = 1, q ∈ {3, 6}, ρ ∈ {0.25, 0.5, 0.75}, 400 paired
episodes per cell, controlled and silent, standard against deduplicated. One added option, `read_mode`; the default keeps
earlier results unchanged.

**Predictions:**
- **D1:** G(30) falls under deduplication in every cell; by at least 25% at ρ ≤ 0.5.
- **D2:** native facts still circulating at t = 30 (silent arm) increase under deduplication at ρ ≤ 0.5.
- **D3:** the silent arm's A2 share does not rise by more than 0.02 (the defence does not make agents worse unattacked).
- **D4 (persistence, q = 6, normal continuation to t = 130):** under deduplication the retained effect δ(130)/δ(30) is
  no larger than under standard reading.
- **D5 (reversal):** the reversal survives (D(3→6) still changes sign across ρ), because sensitivity is unaffected.
  Confidence: low.

## Stage 2: Simulation 2 (Bayesian swarm)

Stage 2 runs only after stage 1, on the new 15-agent setups the researcher plans (symmetric and no_solution_uniform, both
tasks), so that defence and baseline share one clean campaign. Its design will be pre-registered separately before
running. Success criteria, carried over from the 21 October plan:
- equal reading budget;
- lower controlled A2 share;
- no loss of silent accuracy;
- persistence reported.
