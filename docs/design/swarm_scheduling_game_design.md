# A distributed scheduling game for LLM swarm control

*Design proposal, 2026-09-28. Status: for discussion, nothing built.*

## The pitch

Fifteen LLM agents must jointly produce a schedule. Each agent controls one
decision (its own slot) and knows only its own constraints: who it cannot
share a slot with, written in ordinary language. No agent can see the whole
problem, and no agent can finish alone. Underneath the story sits graph
colouring, so every state has an exact, cheap-to-compute score, and one
parameter (how dense the constraints are) moves the game from easy to
impossible.

A controller (another LLM, invisible to the agents, with a message budget)
tries to steer the swarm. Because most instances have many equally good
solutions, the controller can be asked to steer toward a *particular* valid
schedule without ever lying or fighting the truth. That is the clean control
question the voting game could not pose.

## Why not the voting game

| | MuSR voting game | scheduling game |
|---|---|---|
| outcome | one of 3 labels | an energy E(t) over time |
| cooperation | pass facts, vote | agents' choices constrain each other |
| steering target | the wrong answer, fighting true facts | one of many *equally good* answers |
| difficulty | fixed per task | one continuous knob (constraint density) |
| round structure | 15 agents in sequence, 480 serial calls | 15 agents in parallel, 30 serial steps |
| exact answer key | Bayesian posterior over 14,388 worlds | exact energy of any state, exact solution set |

## The task

**Hidden instance (exact, programmatic, never decided by an LLM).**
A random graph on N = 15 nodes and q = 3 colours. Nodes are agents, colours
are slots, an edge means "these two must not share a slot". Energy
E = number of edges whose endpoints share a slot. E = 0 is a valid schedule.

**Rendered instance (what agents read).** Each edge becomes a sentence with a
reason, generated once and frozen, the same way the MuSR generator renders
exact latent facts: "Alice and Bruno both need the only projector, so they
cannot present in the same session." The sentence is flavour; the edge is
the truth, and the energy is always computed from the edge.

**Cover story (open decision).** Conference sessions, shift rota, or meeting
rooms. Any story where "same slot = clash" reads naturally.

### Difficulty, measured at our scale

Exact counts over 40 random graphs per density, N = 15, q = 3
(distinct solutions = colourings divided by the 3! slot relabellings):

| avg degree | edges | P(solvable) | median solutions | 10-90% range | min violations when unsolvable |
|---|---|---|---|---|---|
| 2.0 | 15 | 98% | 4,164 | 2,880-6,912 | 1 |
| 2.5 | 19 | 100% | 756 | 360-1,368 | - |
| 3.0 | 22 | 98% | 140 | 72-348 | 1 |
| 3.5 | 26 | 85% | 16 | 0-72 | 1 |
| 4.0 | 30 | 52% | 1 | 0-12 | 1 |
| 4.5 | 34 | 5% | 0 | 0-0 | 1 |
| 5.0 | 38 | 0% | 0 | - | 2 |
| 6.0 | 45 | 0% | 0 | - | 3.5 |

Rigorously, random graphs of average degree <= 4.03 are almost all
3-colourable; the physics (cavity) prediction for the threshold, ~4.69, is widely
quoted but not yet confirmed from a primary source. At N = 15 the measured
midpoint sits near 4.0. Either
way the transition is sharp enough at our scale to sweep.

## Three regimes, three control problems

1. **Degenerate (degree 2-3, hundreds to thousands of solutions).**
   The swarm will usually find *a* schedule. Without a controller, which one
   it lands on is a distribution over the solution set. The controller is
   given a target schedule Z* and succeeds by shifting that distribution.
   This is steering among equals: nothing the controller says needs to be
   false.
2. **Critical (degree 3.5-4, 0 to a handful of solutions).**
   Finding any schedule is hard. A helpful controller speeds the search; an
   adversarial one tries to trap the swarm in a near-miss (E = 1 or 2).
   This is where swarm dynamics are richest: plateaus, oscillation,
   getting stuck.
3. **Over-constrained (degree >= 4.5, no solution).**
   The best achievable is a few unavoidable clashes. The question becomes
   *who absorbs them*. A controller steering which agent bears the conflict
   is a fairness-flavoured control target, again without deception.

A single sweep over density, with and without each controller, gives a
controllability-versus-hardness phase diagram.

## Agents

- **Knows:** its own edges (the rendered sentences) and nothing else about
  the graph.
- **Sees each round:** its neighbours' current slots, plus a sample of up to
  k board messages (the same sampling machinery as now).
- **Does each round:** picks a slot, and may post one short message
  ("I'm moving to session 2; Carol, can you take 3?").
- **How they cooperate.** Two channels. *Implicitly*, through choices: a
  clash with a neighbour is visible from the slots alone, which is all the
  classical algorithms use. *Explicitly*, through the board: intentions and
  requests ("I'll take session 2", "Carol, can you move to 3?"). The board is
  where LLMs can add something over DSA/MGM, and the only lever the
  controller has.
- **Update schedule (open decision).** Same clash, Alice and Bruno both in
  session 1:

  | schedule | what happens | serial steps / episode |
  |---|---|---|
  | sequential (current runtime) | Alice moves; Bruno, later, sees it resolved and stays. No oscillation, but the board is nearly redundant and acting order shapes outcomes. | ~480 |
  | synchronous | both move to 2, clash again, possibly forever | ~30 |
  | synchronous + random activation (DSA-style, p ~ 0.4) | usually only one moves; the fix is a coin, not the conversation | ~30 |
  | **talk, then move (recommended)** | phase 1: all post intentions in parallel ("moving to 2"); phase 2: all read the board and commit in parallel. Bruno reads Alice and stays. | ~60 |

  Talk-then-move makes the board *the* coordination mechanism (agents avoid
  collisions by announcing before acting), stays parallel, and puts the
  controller where it naturally acts: in the talk phase, shaping intentions
  before commitments. It also gives a free ablation: the same instances with
  the board switched off measure what talking adds, before any controller.

## The controller

- **Sees:** the previous round's board and every agent's current slot
  (it may be given partial observation later, as the status doc proposes).
- **Budget:** a whole-episode message budget with `HOLD`, as now.
- **May say (truthfulness contract, open decision):** true statements about
  the instance or the current state ("Alice and Bruno currently clash",
  "session 3 is free of all of Carol's conflicts"), and recommendations
  ("Carol should take session 3"). Never a false statement about a
  constraint.
- **Posts as** an ordinary-looking participant, as now.

## What we measure

- E(t): the energy trajectory, per episode.
- Time to first E = 0, and whether it stays at 0.
- Final schedule: Hamming distance to the controller's target Z*, and,
  across episodes, the distribution over solutions reached (the controller's
  effect is the shift of that distribution, e.g. as a KL divergence from the
  no-controller distribution).
- In the over-constrained regime: which agents end up in clashes.
- Controller spend over time (where it chose to use the budget).
- Baselines on identical instances: classical distributed algorithms
  (DSA, MGM), a random-walk null, and the no-controller LLM swarm.

## What carries over and what is new

Carries over: the blackboard and its sampling, the controller budget and
`HOLD`, dawn timing, the LLM runtime and providers, Phoenix tracing, study
orchestration on Cesar, and paired initializations.

New: a game module (state, energy, rules), an instance generator with
frozen natural-language rendering, agent and controller prompts, a
synchronous round runtime if we choose it (the current runtime is built
around the sequential cascade), and analysis for E(t) and solution
distributions. This is a new game type, not a configuration change.

## Pilot and cost

Synchronous rounds: 30 rounds x 15 parallel agent calls + up to 30 controller
calls, about 480 calls per episode as now, but about 30 serial steps, so
roughly 3-5 minutes per episode instead of about an hour.

1. No controller, 3 densities (2.5, 3.5, 4.5), 10 episodes each:
   30 episodes, about 14k calls. Checks that the swarm solves easy
   instances, struggles near the threshold, and that E(t) is informative.
2. Add a target-steering controller in the degenerate regime and a
   trapping controller at the threshold: 20 more episodes.
3. The full density sweep only after 1 and 2 show usable signal.

## Risks

- **LLM agents may simply be bad at local search**, oscillating or never
  settling. That is itself a result (compare against DSA/MGM), but it could
  swamp the control signal. The pilot tests this first.
- **Symmetry:** the 3! slot relabellings make "which solution" ambiguous
  unless slots carry meaning in the story (session 1 is morning, and so on).
  Anchoring slots in the cover story breaks the symmetry and makes a target
  schedule meaningful.
- **Too easy for the model:** at low density the swarm may coordinate
  instantly, leaving no dynamics. The density knob is the remedy.

## Mathematical form

State x in [q]^N. Energy H(x) = sum over edges (i,j) of delta(x_i, x_j): the
zero-temperature q-state antiferromagnetic Potts model on G. Agent cost
c_i(x) = sum over neighbours j of delta(x_i, x_j). A unilateral change of x_i
changes H by exactly the change in c_i, so this is an exact potential game
with potential H (Monderer-Shapley):

- asynchronous best response must stop, at a Nash equilibrium = a local
  minimum of H under single moves; proper colourings are equilibria, but so
  are frustrated traps with H > 0;
- synchronous best response can cycle (the formal reason for talk-then-move).

Asynchronous best response is zero-temperature Glauber dynamics; LLM
randomness plays the role of temperature. Steering toward a target colouring
z* is formally an external field, H_h = H - h * sum_i delta(x_i, z*_i); at
equilibrium an infinitesimal h breaks the ground-state degeneracy. The
controller cannot touch H, only post messages, so the effective h it
achieves per unit budget, as a function of hardness, is the measured quantity.

## Literature

Statistical physics: Mulet, Pagnani, Weigt, Zecchina, PRL 89, 268701 (2002);
Zdeborova & Krzakala, PRE 76, 031131 (2007) (clustering, condensation,
freezing, colourability).

Human network experiments: Kearns, Suri, Montfort, Science 313:824 (2006);
Judd, Kearns, Vorobeychik, PNAS (2010); Chaudhuri, Chung, Jamall, "A network
coloring game" (2008).

Closest precedent for the controller: Shirado & Christakis, Nature 545:370
(2017). Three bots in 20-node human networks in a colour coordination game;
noisy, centrally placed bots cut median solution time by 55.6%. The bots aided
coordination; they did not steer toward a target.

Distributed algorithms (from memory, re-check before citing): distributed
CSP (Yokoo et al.), DSA, MGM, max-sum.

LLM swarms: AgentsNet (Grotschla et al., arXiv 2507.08616, 2025), colouring
and four other distributed tasks up to 100 agents; LoopBench (Parsaee et al.,
arXiv 2512.13713, 2025), odd-cycle colouring and symmetry breaking; arXiv
2501.14189, linguistic weighted graph colouring. Neither AgentsNet nor
LoopBench includes a steering or adversarial agent (only these two read).

**Positioning:** the task is not new. The open question is a covert,
budgeted LLM controller steering an LLM swarm to a specific solution among
degenerate ones (or trapping it), with controllability measured across the
hardness phase diagram. A fuller search for adversarial/steering work on LLM
coordination benchmarks is needed before claiming novelty.
