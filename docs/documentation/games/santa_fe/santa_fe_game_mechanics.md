# How the Santa Fe synthetic-control game works

The Santa Fe game is a finite population of agents voting on a binary question.
Agents exchange votes and fact IDs through a public board. A feedback controller
can add messages favoring a chosen answer. The code samples individual agents,
facts, messages, and votes; it does **not** simulate the population by solving
its mean-field equation. The simulator is the stochastic reference against
which that theory is tested.

The implementation is in [the simulator](../../../../src/santa_fe/game.py),
[the v3 round clock](../../../../src/santa_fe/v3_game.py), and
[the state definitions](../../../../src/santa_fe/state.py). It makes no LLM or
provider calls.

## Choose the model version explicitly

| Version | Persistence | Peer fact in a post | Controller post |
| --- | --- | --- | --- |
| `santa_fe_legacy_v2` | Each time an agent is selected for an update | A random active fact, possibly contrary to its vote | Target vote without a fact |
| `santa_fe_epistemic_feedback_v3` | Once for **every agent** at the start of each round | An active fact supporting its **new vote**, or no fact | Target vote **with a target-aligned fact** |

Existing configs default to the legacy version. A v3 YAML must opt in with
`model_version: santa_fe_epistemic_feedback_v3` and its corresponding semantic
switches. The simulator rejects mixed settings. The
[v3 pilot recipe](../../../../configs/santa_fe/v3_pilot.yaml) shows the required
keys. The frozen front-page board is used in both versions. The rest of this
guide describes **v3**, followed by a note on legacy behavior.

## What exists in one episode?

There are `N` agents and `F` distinct synthetic facts. Each fact has an
unchanging sign, `+1` for evidence favoring truth or `-1` for evidence favoring
falsehood. The complete set contains a strict majority of `+1` facts. An agent
has a vote `X_i ∈ {-1,+1}` and a set `K_i` of **active fact IDs**. Its active
facts can change over time; v3 does not store a separate historical-memory
set.

A board message has an author, a vote, an optional fact ID, and a peer or
controller source. The fact ID matters: two messages may repeat the same fact,
and sampling either can reactivate that fact. Controller facts never appear
directly inside an agent's `K_i`; an agent must sample a controller message on
the board to acquire them.

An **episode** draws a new assignment of fact signs and initializes the agents.
A **cell** fixes all model parameters and runs several independently seeded
episodes to measure variability.

### Initialization

1. `F_plus` can set the exact number of truth-favoring facts in v3. If omitted,
   `truth_fact_fraction` sets the intended count by rounding and enforcing a
   strict `+1` majority. The generator then shuffles signs across fact IDs.
2. Each fact starts with `initial_fact_redundancy` distinct holders, capped at
   `N`. Every agent draws an initial vote from its local active evidence. No
   social messages have been sampled yet.
3. Each agent places one initial message on the first front page. In v3 it
   includes a randomly chosen active fact aligned with that initial vote, if
   such a fact exists. Otherwise the message is factless.

The initial population and board are saved as round `0`.

## One v3 round: night, day, and controller

```text
Start: agent votes/facts and a fixed front page B_d
  1. Night: thin every agent's active facts once
  2. Day: N sequential focal-agent update opportunities read B_d
  3. Close the N-message peer board C_d
  4. Controller senses C_d and draws action U_d
  5. If U_d=1, add b target-vote, target-fact messages
End: B_(d+1) = C_d + controller messages
```

### 1. Night persistence

At the start of the round, **every active fact of every agent** independently
survives with probability `rho`. This happens once per round, even if an agent
will be selected several times during the day or not selected at all. Lost
facts are removed from the active set. A later sampled message can reactivate
a lost fact ID.

### 2. Daytime agent updates

There are exactly `N` update slots. Each slot chooses one focal agent uniformly
**with replacement**. An agent may update more than once; another may not
update. The selected agent samples `q` distinct messages without replacement
from the fixed front page `B_d` (or all its messages if fewer than `q` exist).
Every slot reads the same `B_d`. Messages produced during this day are appended
to a separate buffer `C_d` and cannot be read until the next day.

The focal agent adds fact IDs from sampled messages to its active set. Its
**evidence signal** `E` is the average sign of its active facts, or zero if it
has none. Its **social signal** `H` is the average vote in the sampled messages,
or zero if it sampled none. Both are in `[-1,+1]`. It draws a new vote with

```text
P(new vote = +1) = sigmoid(beta_evidence × E + beta_social × H)
sigmoid(z) = 1 / (1 + exp(-z))
```

This is a probability, so even strong evidence does not make a vote perfectly
deterministic. After drawing its new vote, the agent chooses **uniformly among
its active facts with the same sign as that vote** and includes one in its new
message. If it has no supporting active fact, it posts its vote without a fact.
Cross-sign peer messages are impossible in v3.

The `N` peer messages created during the day form the closed board `C_d`. It
contains one message per update slot, so a repeatedly selected agent may have
several posts while an unselected agent has none.

### 3. Controller sensing and action

The controller samples `q_c` messages without replacement from `C_d`. The
current implementation derives `q_c` from `sensing_fraction × N`, rounded and
capped at `N`, with a one-message minimum when `N>0`. Let `Y_d` be the **count**
of sampled messages voting for `controller_target = c`; the observed share is
`y = Y_d/q_c`. This may differ from the true target share of the `N` agents.

The existing feedback policy draws `U_d ∈ {0,1}` with

```text
P(U_d = 1 | y) = sigmoid(policy_beta × (policy_threshold − y))
```

For positive `policy_beta`, action becomes more likely when observed support
for `c` falls below the threshold. The controller's fact pool contains exactly
the facts whose sign is `c`. In v3 `controller_fact_selection` is
`uniform_with_replacement`: if `U_d=1` and budget `b>0`, the controller posts
**exactly `b` messages**, each voting for `c` and carrying one independently
selected fact from that pool. Repeated fact IDs are allowed. The budget is
`round(budget_fraction × N)` per round. If `U_d=0`, it posts nothing. At `b=0`
the policy can still draw `U_d=1`, but the **effective action** is zero because
nothing is posted.

The next front page is the `N` peer posts plus the controller posts. Controller
posts influence agents only through sampling **on the next day**. In
particular, a controller fact cannot change the epistemic state in the round
in which it is posted.

## The state the theory tries to describe

For one agent, the v3 theory class is `a=(r,s,v)`:

- `r`: number of currently active `+1` facts;
- `s`: number of currently active `-1` facts;
- `v`: current vote, `-1` or `+1`.

`Z_(r,s,v)` counts agents in that class, and `z_(r,s,v)=Z_(r,s,v)/N` is its
fraction. The simulator retains the **complete occupied class vector** at
round boundaries, together with each agent's fact IDs. It also records
`x` (truth-vote share), `kappa_plus` (mean active fraction of all `+1` facts),
`kappa_minus` (mean active fraction of all `-1` facts), and `kappa_ctrl`
(the one aligned with the controller target). These are summaries of the
microscopic state, not replacement state variables inside the simulator.

The board is recorded at three distinct stages: the peer emissions `C_d`, the
controller emissions, and the combined next front page `B_(d+1)`. Messages
retain their fact IDs and source. Four peer message categories are possible:
`(+,+)`, `(+,0)`, `(-,-)`, and `(-,0)`, where `0` means no fact.

## Parameters

| Parameter | Meaning in v3 |
| --- | --- |
| `N`, `F` | Agents and distinct facts. |
| `q` | Previous-front-page messages sampled in one agent update. |
| `rounds` | Number of night/day/controller cycles after initialization. |
| `initial_fact_redundancy` | Initial holders of each fact. |
| `F_plus` | Optional exact `+1` fact count in v3; must be a strict majority below `F`. |
| `truth_fact_fraction` | Intended fraction of `+1` facts when `F_plus` is omitted; strict majority enforced. |
| `rho` | Chance an active fact survives the **once-per-round global night**. |
| `beta_evidence`, `beta_social` | Weights of the active-evidence and sampled-vote signals. |
| `sensing_fraction` | Determines `q_c`, the controller's peer-board sample size. |
| `controller_target` | Vote and fact sign promoted by the controller. |
| `policy_beta`, `policy_threshold` | Shape the feedback action probability. |
| `budget_fraction` | Determines integer posts `b` if the controller acts. |
| `controller_fact_selection` | V3 fact-pool sampling rule; currently `uniform_with_replacement`. |
| `model_version` and semantic switches | Select the v3 or historical microscopic clock. |

`experiment.episodes`, seeds, process counts, output settings, estimator bins,
bootstrap draws, and null draws govern replication, retention, or analysis.
They do not enter an agent's voting formula. V3 requires microscopic trajectory
retention because transition and path checks need the individual events.

## What is saved and what is inferred

Every v3 round row includes the before/after night fact sets, agent states,
occupied classes, boards, sensor sample IDs, action probability, controller
fact IDs, reduced coverage measures, and primitive persistence/sensing/action
probabilities. Every microscopic slot records the sampled message IDs and
signs, acquired fact IDs, the before/after `a` classes, both voting signals,
the realized vote probability, and the posted fact and its selection
probability. The saved downstream RNG boundary state lets a validation utility
replay the factual next day exactly and branch the controller action while
matching downstream random streams.

The exact finite-`N` simulator should be distinguished from two analysis
layers: the **heterogeneous stochastic mean-field approximation** for the
class vector `z`, and a further **reduced closure** using only `x`,
`kappa_plus`, and `kappa_minus`. The
[v3 validation plan](../../../tdd/santa_fe/santa_fe_v3_simulation_update_plan.md)
and [theory validation prompt](../../../tdd/santa_fe/SANTA_FE_V3_THEORY_VALIDATION_AGENT_PROMPT.md)
set out tests of both approximations. The
[post-hoc validator](../../../../src/santa_fe/v3_validation.py) checks exact
one-step probabilities, peer emission, board composition, acquisition,
closure, sensing, and paired controller responses from saved trajectories.
MI/CMI, episode bootstrap, policy nulls, and overlap diagnostics continue to
use the shared MA-CC information engine; they are calculated after simulation. The
[reduced theory integration guide](santa_fe_theory_integration.md) shows how
the supplied Langevin runner is matched to retained v3 episodes and how its
plots and residual tables are produced.

## Historical configurations

Historical Santa Fe YAMLs still default to `santa_fe_legacy_v2`. In that
version, fact retention is applied only when an agent is chosen for an update;
a peer post may contain any active fact; and controller recommendations have
no fact. Historical results must be interpreted with those rules. To compare
v3 with old runs, use separate result roots and retain each resolved YAML and
model-version provenance.
