# How the Santa Fe synthetic-control game works

This guide describes one generic game, independent of any parameter sweep or
cluster run. The implementation is in [`src/santa_fe/game.py`](../../src/santa_fe/game.py),
with its state and parameters in [`src/santa_fe/state.py`](../../src/santa_fe/state.py).
It is a synthetic population model: no language model, external provider, or
real-world fact checker participates.

## The basic picture

There are `N` agents voting on a binary question. A vote is `+1` (the true
answer) or `-1` (the false answer). There are `F` distinct facts. Each fact has
an evidence sign, `+1` or `-1`; the complete collection has a majority of
`+1` signs. Agents usually see only some facts, so their local evidence can
favor either answer even though the complete collection favors truth.

The agents communicate through a public board of messages. A normal message
contains the author's current vote and, if the author has any active facts,
one randomly selected fact. A controller can add messages recommending its
chosen target vote. Controller messages carry **no fact**. The model therefore
lets us study how factual evidence, social signals, and repeated
recommendations interact.

An **episode** initializes a new fact assignment and population, then runs for
`rounds` rounds. Episodes use independent random seeds. A **cell** is one fixed
set of model parameters; repeated episodes in that cell show how variable its
outcomes are.

## Before the first round

1. The game assigns a sign to each of the `F` facts. It uses
   `truth_fact_fraction` to choose the intended fraction of truth-favoring
   facts, but enforces a strict truth-favoring majority. The signs are shuffled
   across fact IDs for each episode.
2. Each fact is initially placed with `initial_fact_redundancy` distinct
   agents, capped at `N`. Thus a fact can have several holders. An agent's
   **active facts** are the facts it currently retains.
3. Every agent draws an initial vote from its active evidence. There is no
   social signal yet.
4. Every agent makes one initial board message: its vote and one randomly
   selected active fact, if it has one. This is the board read during round 1.

The game records this initial population as round `0`. The controller has not
sensed a board or posted a recommendation at this point.

## What happens in one round

The following sequence repeats for each round `t = 1, …, rounds`:

```text
previous board
    → N agent update slots, producing N new ordinary messages
    → controller senses the new, closed board
    → controller draws an action and may add recommendations
    → new board becomes the previous board for round t+1
```

### 1. Agents update and make a new board

There are exactly `N` update **slots**, but each slot selects an agent at
random *with replacement*. An agent may update more than once in a round;
another may not update at all. These updates occur in order, so an agent
selected later can retain the vote and facts it gained in an earlier slot.

In each slot, the selected agent:

1. Independently retains each of its currently active facts with probability
   `rho`. This is a retention chance **when that agent updates**, not a global
   once-per-round deletion of everyone else's facts.
2. Samples `q` distinct messages from the **previous round's board** (or all
   available messages if the board has fewer than `q`). Every slot reads that
   same previous board; messages written in the current round cannot be sampled
   until the next round.
3. Adds any fact carried by sampled messages to its retained facts. Sampling a
   controller message can affect the social signal, but cannot add a fact.
4. Draws a new vote using its updated evidence and the votes in the sampled
   messages.
5. Writes one ordinary message containing that vote and one randomly chosen
   active fact, if any.

After `N` slots, these `N` ordinary messages form the **closed board** for the
round. They are new posts, not a persistent accumulation of all past posts.

### 2. How an agent chooses its vote

The agent's **evidence signal** `e` is the average sign of its active facts. It
is zero if the agent has none. The **social signal** `s` is the average vote in
the messages it sampled. It is zero if it sampled none. Both signals lie
between `-1` and `+1`.

The probability of a truth vote is

```text
P(vote = +1) = sigmoid(beta_evidence × e + beta_social × s)
sigmoid(z) = 1 / (1 + exp(-z))
```

The vote is sampled from this probability; it is not simply the sign of the
larger term. `beta_evidence` controls the weight of currently active factual
evidence, while `beta_social` controls the weight of sampled votes. For
positive weights, a larger value makes that signal more influential. Local
evidence and the sampled social majority can disagree.

### 3. The controller observes and acts

Only after the agents finish does the controller sample messages from the
closed board. It samples approximately `sensing_fraction × N` messages,
rounded to an integer and capped at `N`. When the board is nonempty, it senses
at least **one** message even if `sensing_fraction` is zero. Its observation
`y` is the share of sampled messages voting for `controller_target`.

The controller then draws a binary policy decision `U`:

```text
P(U = 1) = sigmoid(policy_beta × (policy_threshold − y))
```

With the usual positive `policy_beta`, the controller is more likely to act
when observed support for its target is below `policy_threshold`. It uses its
sampled observation, not the true population share.

If `U = 1` and the integer budget `b` is positive, the controller adds **exactly
`b` identical recommendation messages** to the board. Each recommends
`controller_target` and carries no fact. The budget is a per-round number of
posts, computed as `round(budget_fraction × N)`. If `U = 0`, it adds none. If
`b = 0`, the policy may still draw `U = 1`, but no message is posted; the
recorded **effective action** is then zero.

The next round's board consists of the `N` ordinary messages from the closed
board plus any controller recommendations. Agents can encounter those
recommendations only when they sample that board in the **next** round. This
one-round delay matters when interpreting controller information measures.

## Parameters at a glance

| Parameter | What it changes |
| --- | --- |
| `N` | Number of agents and update slots per round. |
| `F` | Number of distinct facts. |
| `q` | Number of previous-board messages sampled in each update. |
| `rounds` | Number of rounds after initialization. |
| `initial_fact_redundancy` | Initial number of holders of each fact, capped at `N`. |
| `truth_fact_fraction` | Intended fraction of truth-favoring facts; a strict majority is enforced. |
| `rho` | Chance that one active fact survives when its holder updates. |
| `beta_evidence` | Weight of the agent's active-fact evidence signal. |
| `beta_social` | Weight of the agent's sampled-vote social signal. |
| `sensing_fraction` | Fraction of the closed board sampled by the controller, with a one-message minimum. |
| `controller_target` | Vote recommended by the controller: `+1` or `-1`. |
| `policy_beta` | Steepness and direction of the controller's feedback policy. |
| `policy_threshold` | Observed target-share level around which the policy changes. |
| `budget_fraction` | Controller posts per active round as a fraction of `N`, rounded to integer `b`. |
| `save_micro` | Whether to retain one diagnostic record for each update slot. It does not change the dynamics. |

The YAML also has execution and analysis settings such as `episodes`, `seed`,
`processes`, information bins, bootstrap draws, and null draws. These control
replication, computation, and measurement; they are not part of an individual
agent's decision rule.

## What the simulator records

Each episode has a round-0 row and one row after each subsequent round. The
round row includes the share voting for truth, the share voting for the
controller target, vote entropy, fact-coverage summaries, the controller's
observation and action probability, and the number of controller posts added
to the board for the next round. For example:

- `kappa_mean_coverage` is the average fraction of all `F` facts active in an
  agent.
- `kappa_population_coverage` is the fraction of all `F` facts active in at
  least one agent.

These two coverage measures answer different questions: the population may
collectively retain every fact while each individual knows only a few.
Optional microscopic rows record the selected agent, before/after vote and
fact counts, sampled controller messages, and the two signals in each slot.

Information measures and plots are calculated **after** simulation from saved
trajectories. For controller-to-population CMI, the relevant sequence is the
population after round `t`, the controller action added after round `t`, and
the population after round `t+1`. The action recorded after the final round
has no observed successor round. See the [study and analysis guide](santa_fe_synthetic_control.md)
for estimators, null calculations, and output files.
