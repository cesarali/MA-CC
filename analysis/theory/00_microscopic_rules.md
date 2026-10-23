# Microscopic rules of Simulation 2 (extracted from the code)

**Purpose.** The exact transition rules that any reduced theory must use or approximate (theory plan §3.2 step 1).
Source: [`../task003_llm_free/simulation_2/simulator/sim2/simulation_2.py`](../task003_llm_free/simulation_2/simulator/sim2/simulation_2.py) (function `run_episode` and
helpers), `llmfree_core/world.py`, `llmfree_core/setups.py`, spec `../task003_llm_free/simulation_2/simulator/simulation_2_spec.md`. Line numbers
refer to commit `3083fc9` and later. Nothing here is fitted.

## 1. World and inference

- 49 facts, 14,388 possible worlds (4,796 per allocation A0, A1, A2). Each fact is a **logical constraint**: the set
  of worlds consistent with it (`World._fact_mask`).
- An agent's belief given remembered facts M: P(Aa | M) = (worlds surviving every fact in M that belong to Aa) /
  (all surviving worlds) (`World.posterior`). Empty memory gives (1/3, 1/3, 1/3). **Not additive:** facts combine
  by intersection, so a set can prove an allocation (all surviving worlds in one allocation) although no member does.
- Setups (`llmfree_core/setups.py`, `experimental_setup/`):

| | symmetric | nosolution |
|---|---|---|
| agents / distinct agent facts | 24 / 18 (6 facts held by 2 agents) | 24 / 15 (9 facts held by 2 agents) |
| all agent facts → worlds left, posterior | 16, (1, 0, 0): proof of A0 | 46, (0.5, 0, 0.5) |
| A2 pool / controller-only facts | 12 / 10 (2 shared with agents) | 12 / 12 (the same pool serves both targets) |
| all controller-only facts alone → posterior | (0.065, 0, 0.935) | (0.37, 0.26, 0.37): **neutral between A0 and A2** |
| agent facts + controller-only facts | (1, 0, 0) | (0.75, 0, 0.25) |
| single fact: P(A2) range, agent facts / A2 pool | 0.205–0.449 / 0.347–0.585 | 0.205–0.449 / 0.207–0.585 |

## 2. Clocks and order (§4, `run_episode` schedules)

- Each agent acts at the events of its own Poisson clock, rate λa = 1 (`keyed_rng(seed, e, "agent_clock", a)`), up
  to t_end.
- The controller (false arm: target A2) acts at the events of a Poisson clock, rate λc, for t < t_control = 30, while
  budget is left. The unit-exponential draws are shared across rates.
- At equal times events run in the order measure, control, clear, dawn, forget, act (`MEASURE ... ACT`).

## 3. One agent action (`kind == "act"`)

1. **Read.** Window = the W = 48 newest posts on the board, **excluding the agent's own posts, which are removed
   before the window is taken** (`read_window`). The agent draws min(q, |window|) **posts** uniformly **without
   replacement** (`.sample`). Several posts can carry the same fact, so a fact on k of the 48 posts is read with
   probability 1 − C(48−k, q)/C(48, q) (hypergeometric).
2. **Remember.** Every fact read becomes active (newly, or again), and its lifetime is **restarted** (a new
   Exponential(γ) lifetime, γ = −ln ρ; ρ = 1: never forgotten). Facts are added only at actions.
3. **Believe and vote.** P = posterior(active memory); the vote is drawn from P (**probability matching**). It becomes
   the agent's last vote.
4. **Post** (`choose_post`, at most one fact):
   - candidates "in context": remembered facts f with P(vote | M) > P(vote | M \ f), i.e. facts whose removal would
     lower the belief in the agent's own vote;
   - if none, the fallback: remembered facts f with P(vote | {f}) > 1/3 (supporting the vote on their own);
   - if none: no post;
   - **prefer facts not read in this action**: if any candidate was not just read, choose only among those;
   - pick one candidate uniformly. The post records the fact and the agent's vote.

## 4. Forgetting (`kind == "forget"`)

Each remembered fact has its own lifetime, started when it was last read (the starting fact: at t = 0). When it ends
the fact leaves memory (if not renewed meanwhile). **The starting fact is forgettable**: with no reading, memory
empties and the belief returns to the prior.

## 5. Controller action (`kind == "control"`, `controller_decision`)

1. Reads min(q_c, |window|) of the W newest **agent** posts (own posts excluded), without replacement.
   R = facts read; v̂ = share of the read posts whose recorded author vote is the target (votes can be old).
2. Silent if R already proves the target and the proof stop is on (never for A2), or if the gate is on and
   v̂ ≥ 0.75.
3. Otherwise posts the pool fact f that maximises P(target | R ∪ {f}) (ties: first fact ID); if that probability is 0,
   the pool fact that maximises P(target | {f}). **Deterministic given R**, which is why it delivers only about 4–6
   distinct facts. Each post uses one unit of budget B.

## 6. Board, measurement, pairing

- **Board:** posts are never deleted; readers see only the newest W eligible posts (a sliding window, first in, first
  out). Agents post about 24 times per time unit, so a post stays readable about W/24 = 2 time units.
- **Measurement** every 0.25: vote shares among agents who have voted; `agent_snapshots` stores every agent's memory
  (bitmask), last vote and time since it.
- **Pairing:** every random stream is `keyed_rng(seed, episode, purpose, …)`, never keyed by arm, q, ρ or other
  settings. Arms and settings share clocks, starting order and lifetime draws.
- **Post-stop modes** (`post_stop_mode`, from t_control): normal; no_comm (no reading, no posting); frozen_board (read
  only the board at t_control, no posting); purge_controller_facts (controller-only facts removed from memories and
  board, then normal).

## 7. Consequences for a reduced theory

- **Vote lag:** each agent revotes at rate 1 whatever its state, so, exactly in expectation,
  dE[x_a]/dt = E[P_a just after reading at an action] − E[x_a] (among agents who have voted).
- **Posting depends on the whole memory set** (support in context) and against just-read facts. A fact-by-fact model
  cannot produce the posting probabilities without an assumption about joint memory.
- **The controller's selection is a feedback:** what it posts depends on the agent posts it reads.
- **Repetition matters:** reading samples posts, not facts, so a fact's exposure grows with its number of copies on
  the board.
