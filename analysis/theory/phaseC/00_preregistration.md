# Phase C — pre-registration of C1 and C2 (written 24 October 2026, before either is run)

Background:
- **Phase A** (`../phaseA/phaseA_results.md`): given the board, agents are independent, and the memory and evidence
  half is solved exactly. Counting facts by kind is not a sufficient evidence state.
- **Phase B** (`../phaseB/phaseB_results.md`): a finite stochastic controller and a shared finite board each give
  about half of M4's gain, and the board alone gives the crossover and the post-stop ordering.

Working hypothesis: **the board is the only finite object that matters.**

## C1 — is the board's finiteness enough? (model ablation, post-hoc)

**M4b** = M4 without the integer holder counts and extinction rule. It is an unlimited population of exact single-agent
chains (R = 30 copies of the 24 agents) reading one shared, realised board of W = 48 posts that receives the swarm's
posting stream (each population post enters with probability 1/R), plus one stochastic controller.

Runs: `all`, `post_stop` (40 realisations, R = 30), `op` (20, R = 20), `op_post_stop` (40, R = 20); scored with the
Phase B scorer.

**Decision rule.** "The board's finiteness is enough" if M4b meets all of the following:
- task003 held-out G error ≤ 0.04;
- Object Placements G error ≤ 0.04;
- post-stop ordering ≥ 6/8 in both tasks.

Otherwise the swarm's own finiteness (integer holder counts) is also needed, and Phase C keeps it.

## C2 — a board-only theory of persistence under normal continuation (no fitted parameter)

**Model NBD (neutral board drift).** From t = 30, in each episode and arm:
- **Board.** It starts as the stored board at t = 30. New posts arrive as a Poisson process at rate ν. Each new post
  carries the fact of a post drawn uniformly from the current W = 48 newest posts (neutral copying: no selection, no new
  facts). The window keeps the W newest.
- **Rate.** ν is the agents' posting rate measured in the same cell before the stop (posts per time unit, 20 ≤ t ≤ 30).
  It is a microscopic rate, not fitted to δ.
- **Agents.** Each starts from its stored memory and last vote at t = 30. It is the exact single-agent chain of
  Phase A4: it acts at rate 1, reads q = 6 posts from the current drifted board, forgets at rate γ, and votes by
  probability matching. Agents' posts do not feed the board; the board evolves only by the neutral copying above.
- **Output.** δ(t) = x_A2(false) − x_A2(silent) at t = 30.25 … 130, and the number of distinct controller-only facts on
  the board at t = 40, 80, 130.

Cells: normal continuation, both tasks, both setups, ρ 0.25 and 0.75, q 6. Object Placements symmetric ρ 0.75 is
reported but not scored, because there is no effect to persist (δ(30) = 0.005).

**Expectations, stated in advance** (7 scored cells):
1. *Variety.* The model's mean number of distinct controller-only facts per board is within 25% of the data
   (`../reports/2026-10-23_task_comparison/ctrl_diversity.csv`) at t = 80 and 130, in every scored cell.
2. *Persistence.* The retained fraction δ(130)/δ(30) is within 0.2 of the data's in every scored cell. In particular,
   task003 retains ≥ 0.85, while Object Placements no-proof fades (data 0.37 at ρ 0.25 and 0.33 at ρ 0.75).

If expectation 2 fails while expectation 1 holds, drift of the board is not the whole story, and selection by the
posting rule matters. The planned next step would then be one revision adding the posting rule's selection, computed
from the exact single-agent posting probabilities.

## Change log
(none yet)
