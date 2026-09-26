# Control Efficiency Project Plan V4 — Post-Meeting Plan

## 1. Fixed definitions and design

We study control of interacting LLM agents solving **MuSR under distributed partial information**. The MuSR task is fixed. The immediate objective is to implement the epistemic and control harnesses cleanly and execute the agreed experimental sequence.

### Epistemic harness
Task evidence is represented as individually identifiable sentences/facts. Individual facts can provide evidence in either the **truth** or **false** direction, and combinations of partial evidence may have nontrivial effects. When all relevant evidence is available, the intended behavior is that the agent selects the true answer with essentially certainty. The harness must track which facts each agent encounters and retains.

### Persistence
All primary experiments use

\[
\rho \in \{0.75, 1.0\}.
\]

Persistence controls the availability/retention of task-relevant evidence in the agents' epistemic state.

### Interaction horizon and message board
Episodes run for **30 rounds**, giving agents sufficient time to exchange information. The system uses the **front-page message board**: the previous round/day's board is available rather than an indefinitely accumulated history.

### Communication modes
- **Full communication:** agents/controller can use reports and requests, permitting richer information exchange and coordination.
- **Report-only:** communication is restricted to reports.

### Budget protocols
Two separate control harnesses are required:

1. **Full-episode budget:** the controller receives a finite intervention reservoir for the complete 30-round episode,
   \[
   B \in \{90,180\},
   \]
   and decides how to allocate it over time. This is the more difficult protocol to model because temporal resource allocation becomes part of the controller policy.

2. **Per-round budget:** the controller receives a fresh fixed intervention allowance each round,
   \[
   b \in \{3,6\}.
   \]

The budgets are matched because \(90=30\times3\) and \(180=30\times6\). Therefore, differences between the protocols can be studied without changing the maximum total intervention capacity.

### Sensing
Each controlled experiment is crossed with two sensing conditions:

- **Full-board sensing:** the controller observes the complete available board.
- **Partial-board sensing:** the controller observes only the prescribed subset of the available board.

Sensing must remain distinct from the intervention budget: it determines what population information the controller observes, while the budget determines how much it can intervene.

### Control direction and baseline
Controlled experiments are run under both **truth control** and **false control**. A common **no-control baseline** is generated separately and reused whenever the underlying uncontrolled dynamics are identical; it is not redundantly repeated across budget or sensing conditions.

---

## 2. Experimental sequence

| Priority | Budget protocol | Communication | Budget |
|---|---|---|---|
| **1** | Full episode | **Full communication** | \(B=90,180\) |
| **2** | Per round | **Full communication** | \(b=3,6\) |
| **3** | Full episode | Report-only | \(B=90,180\) |
| **4** | Per round | Report-only | \(b=3,6\) |

**Experiments 1 and 2 are the immediate priority.** Experiments 3 and 4 remain part of the planned design but are postponed until the first two are implemented and evaluated.

For Experiments 1–2, every controlled configuration is crossed with:

- \(\rho\in\{0.75,1.0\}\),
- two matched budget levels,
- full-board vs partial-board sensing,
- truth vs false control,
- **50 independent episodes per cell initially**.

This gives

\[
2\ \text{experiments}\times2\rho\times2\ \text{budgets}\times2\ \text{sensing conditions}\times2\ \text{control directions}\times50
= \mathbf{1,600}
\]

controlled episodes.

The shared no-control baseline requires

\[
2\rho\times50 = \mathbf{100}
\]

additional episodes, giving an **initial priority study of 1,700 episodes**. The study can subsequently be extended from 50 to 100 episodes per cell if the statistical analysis indicates that the additional repetitions are necessary.

---

## 3. Concurrent workstreams

| Workstream | Immediate deliverable | Success criterion | Suggested leads |
|---|---|---|---|
| **Parallelization** | Episode-level parallel execution with dynamic work allocation, reliable bookkeeping, and stop/restart support. | Available workers remain occupied while work is pending; completed episodes are neither lost nor duplicated after restart. | **Alan + Chris** |
| **Experimental harnesses** | Implement the **full-episode** and **per-round** controller harnesses, full communication, sensing variants, persistence, fact tracking, and the front-page board. | Both harnesses execute the agreed protocol correctly; facts and interventions are traceable; complete evidence produces the intended true solution. Once validated, freeze the harness for the primary study. | Team implementation effort |
| **Statistics and theory** | Define the primary response/information statistics and determine whether 50 episodes per cell provide adequate precision. | Analysis can distinguish signal from estimator/null variability sufficiently for the primary comparisons; increase to 100 episodes only if required. | **César + Ramses** |

The three workstreams can proceed **concurrently**. Suggested leads indicate responsibility for delivery, not exclusive ownership.

## 4. Execution rule

**Implement → validate → freeze → run Experiments 1–2 at 50 episodes/cell → assess statistical reliability → increase repetitions only if needed → proceed to Experiments 3–4.**

The objective is not to search indefinitely for the perfect implementation. We need to **choose the mistakes we are willing to make**: approximations should be explicit, each component should have a concrete success criterion, and once that criterion is met the component is frozen unless validation exposes a specific failure.
