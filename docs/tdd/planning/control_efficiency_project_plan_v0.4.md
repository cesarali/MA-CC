# Control Efficiency in Interacting LLM Systems
## Project Plan — v0.4

## Page 1 — What are we trying to do?

### 1. Purpose

This document defines the next stage of the project: **what scientific system we are building, what counts as success, what must be decided and frozen before expensive experiments begin, and who should lead each remaining task.**

**We need to choose the mistakes we are willing to make.** We cannot eliminate every approximation, modelling choice, prompt dependence, or source of bias. Instead, we should decide explicitly which approximations preserve the scientific question, define what counts as success for each component, and freeze components once those criteria are met.

### 2. Scientific objective

We study **control of interacting LLM agents solving a task under partial information**.

The essential ingredients are:

- **LLMs:** language and reasoning are part of the dynamics.
- **Partial task information:** relevant facts are distributed across agents.
- **Interaction:** agents exchange information through a shared message board.
- **Control:** a controller/coordinator can steer the population through a limited intervention channel.
- **Epistemic persistence:** information is not necessarily retained indefinitely; persistence controls how strongly previously acquired facts remain available.
- **Measurable response:** the population state allows us to quantify the direction and magnitude of control.

Our eventual question is one of **control efficiency**: how much directed population change do we obtain for the resources used? Resources may include intervention budget and, if we explicitly study feedback, information obtained through sensing.

### 3. The dream experiment and why we need a controlled one

Ideally, we would recreate an event such as the **Hugging Face multi-agent incident** thousands of times. We would systematically restrict how many messages a coordinator/controller is allowed to send and how many agents it can interact with, and then measure how these constraints change collective task success.

However, we have neither the money nor the time.

Our goal is therefore to construct the **smallest controlled experiment that retains the mechanisms we care about**. We deliberately constrain the system so that the contribution of individual control resources can be identified statistically.

### 4. Immediate experimental target

The immediate target is **not a large phase diagram**.

The current working endpoint is a minimal experiment with:

- **2/3 actuation budgets**
- **2 epistemic persistence values**
- approximately **100 independent repetitions per condition**

This gives **4 experimental cells** and approximately **400 episodes**.

The purpose of this experiment is to establish, under fixed and well-defined conditions, whether budget and epistemic persistence produce measurable differences in population response and control efficiency.

**Open control will be followed by feedback control.** The unresolved design question is how sensing should work in the feedback experiment: whether sensing is externally prescribed and controlled, or whether the LLM controller is allowed to choose what/how much to observe. This choice determines whether sensing itself remains an identifiable experimental variable.

---

## Page 2 — What needs to happen now?

### 5. Three concurrent workstreams

| Workstream | Immediate objective | Success criterion | Suggested leads |
|---|---|---|---|
| **A. Computation** | Make episodes the unit of parallel execution. Workers continuously claim unfinished episodes across conditions. | CPUs remain occupied while work exists; experiments can be started, stopped, and resumed without losing or duplicating completed episodes. | **Chris + Alan** |
| **B. Epistemic task & harness** | Produce one small, auditable epistemic game and a prompt/harness implementing the intended interaction protocol. | Epistemic criteria pass; facts are traceable; agents follow the protocol reliably; the prompt does not directly prescribe the phenomenon of interest. Then **freeze the harness**. | **Darío + Chris** |
| **C. Theory & statistics** | Define response/efficiency quantities, the open-control baseline and feedback-control model, theoretical reference model, and statistical requirements. | A toy model establishes how much data is required for the chosen response/MI/transfer-entropy estimators; the primary analysis is fixed before the expensive run. | **César + Ramses** |

These are **suggested leads, not exclusive ownership**. The three workstreams can proceed concurrently, and everyone can contribute where useful.

### 6. Decisions for the meeting

Some questions should be decided collectively rather than becoming open-ended development projects.

**Message-board memory.**  
Choose and freeze how previous board states remain available: complete history, a finite history, or age-dependent retrieval.

**Communication mode.**  
Choose the first experimental mode. **Report-only** primarily transports task information. **Full communication** additionally permits requests and coordination. Full communication is richer but introduces coordination as another phenomenon and makes simulations longer.

**Sensing in feedback control.**  
Open control is the baseline and feedback control follows. The decision is whether the feedback experiment uses an externally prescribed sensing mechanism or allows the LLM controller to decide what/how much to sense. The former makes sensing experimentally identifiable; the latter is closer to autonomous operation but entangles sensing with policy and actuation.

### 7. Rules for freezing the experiment

**Epistemic task**
- Complete evidence should reliably solve the task.
- More relevant evidence should generally improve performance.
- Partial true evidence should retain enough ambiguity for competing interpretations and adversarial steering.
- No individual fact should dominate the task disproportionately.

**Harness**
- Agents understand the task and output format.
- Agents can use information made available to them.
- Agents interact through the intended channels.
- The prompt does not directly prescribe the collective behavior we intend to measure.
- We do **not** require invariance to arbitrary prompt changes. Once these functional conditions are satisfied, the harness is frozen.

**Statistics**
- The number of repetitions is not chosen only by intuition.
- A cheap stochastic/toy model should test the estimator at different sample sizes.
- The working value of ~100 repetitions per cell should be confirmed or revised from this analysis.

### First milestone

> **One fixed and auditable experiment, a validated epistemic task and harness, a justified number of repetitions, agreed message-board and communication rules, and infrastructure capable of executing the experiment reliably.**

After this milestone, we stop redesigning the primary experiment and start running the study.

---

# Appendix

## A. Meeting decision checklist

The meeting should begin with a short silent reading period. The goal is not to reopen every design choice, but to leave with the remaining decisions made. Any unresolved item should leave the meeting with an **owner, deadline, and explicit criterion for resolution**.

- [X] **Sensing in feedback control:** Is sensing externally prescribed, or can the LLM controller decide what/how much to observe?
- [X] **Communication mode:** Do we use report-only or full communication for the primary experiment?
- [X] **Message-board memory:** Do we retain full history, a finite window, or use age-dependent retrieval?
- [ ] **Epistemic game:** Do we agree that sufficiency, evidence gradient, partial ambiguity, and balanced informativeness are the criteria for freezing the task?
- [ ] **Harness:** Do we agree on the functional criteria that make the prompt/harness valid and therefore ready to freeze?
- [X] **Statistics:** What toy/reference experiment will determine whether approximately 100 repetitions per cell are sufficient?
- [X] **Execution:** Do we agree on episode-level dynamic scheduling and the required start/stop/restart behavior?
- [ ] **Responsibilities and deadlines:** Confirm the suggested leads for workstreams A–C and assign deadlines to their immediate deliverables.

**Meeting protocol:** silent reading → clarification questions → checklist decisions → assign unresolved items → confirm owners and deadlines.

## B. Controlled experiment versus natural operation

In a naturally operating multi-agent system, a controller could potentially decide what to observe, whom to contact, when to intervene, and how strongly to intervene. Such behavior is scientifically interesting but makes causal identification difficult because sensing, policy, and actuation become endogenous.

The controlled experiment deliberately removes some of this freedom. Resources are externally specified so that their individual effects can be measured. This artificiality is intentional: the experiment is an instrument for isolating mechanisms rather than a complete reproduction of an unconstrained deployment.

A later study can relax these constraints and investigate controllers that autonomously decide how to gather information and allocate their intervention resources.

## C. Open versus feedback control

**Open control** applies an intervention without conditioning it on a current observation of the population.

**Feedback control** observes some prescribed information about the current population and conditions the subsequent intervention on that observation.

Open control serves as the **control baseline**: it measures what the intervention protocol can achieve without conditioning on the current population. It is not generally an upper bound on achievable control—feedback can improve performance by using state information—but it provides the reference against which the value of sensing can be measured.

Feedback control then introduces population information before intervention. The unresolved question is whether that sensing channel is prescribed experimentally or selected autonomously by the LLM controller. Prescribed sensing supports a clean study of sensing as a resource; autonomous sensing is more naturalistic but entangles sensing with the controller policy.

## D. Epistemic persistence

Real agents encounter much more information than can be represented explicitly in our experiment. They selectively attend to, retrieve, and retain only part of it.

We use **epistemic persistence** as a controlled abstraction of this limitation. Persistence determines how strongly previously acquired facts remain available to influence subsequent decisions.

This is scientifically relevant to control. At high persistence, agents may accumulate enough task evidence to decide largely independently of neighbors or controller interventions. At lower persistence, newly encountered social information and control can have greater influence.

Persistence therefore controls part of the competition between **private evidence, social information, and control** without requiring an unrealistically large information environment.

## E. What makes the epistemic task valid?

The epistemic task should satisfy four approximate properties:

1. **Sufficiency:** an agent with all relevant facts should solve the task with probability close to one.
2. **Evidence gradient:** more relevant evidence should, statistically, improve the probability of the correct conclusion.
3. **Partial ambiguity:** individual true facts need not uniquely determine the correct answer; partial evidence can support competing interpretations.
4. **Balanced informativeness:** no single fact should overwhelmingly determine the result; individual facts should have reasonably homogeneous influence.

These criteria provide a concrete stopping rule: once the task satisfies them sufficiently well, the task is ready for the primary experiment.

## F. What makes the harness valid?

Prompt dependence cannot be eliminated.

For example, a prompt explicitly instructing agents to ignore other agents would trivially suppress social transmission. A prompt explicitly instructing them to follow the majority could manufacture susceptibility. These prompts define different dynamical systems.

We therefore do **not** require the phenomenon to be invariant under arbitrary prompt changes.

The harness is valid when it implements the intended experimental protocol without directly prescribing the behavior being measured. Agents should understand the task, use the information they receive, interact through the permitted channels, and produce valid decisions reliably.

Once these conditions are satisfied, the harness is frozen. Alternative prompts or models can subsequently test the **generality and robustness** of the phenomenon; they are not prerequisites for defining the first experimental system.

The intended claim is therefore limited and precise: we characterize a phenomenon in a specified LLM multi-agent system. Whether the same phenomenon persists quantitatively across prompts, models, and interaction protocols is a separate empirical question.

## G. Communication and message-board memory

The message board defines the interaction protocol of the system.

Two communication regimes are currently relevant:

- **Report-only:** agents primarily place task-relevant information on the board.
- **Full communication:** agents can additionally request information and coordinate, while the controller can issue directives.

Full communication may better represent richer multi-agent behavior, but it also introduces coordination as an additional mechanism that we do not currently measure independently and increases simulation time.

The board also requires a memory rule. Candidate choices include:

- retaining the complete history;
- retaining only a fixed number of previous rounds;
- retaining history but reducing retrieval probability with message age.

For the first experiment, one communication mode and one board-memory rule should be agreed upon and frozen.

## H. Efficiency

At the simplest conceptual level,

\[
\text{efficiency} \sim
\frac{\text{directed population response}}
{\text{control resource}}.
\]

The control resource can represent actuation budget or, when feedback is studied, information acquired through sensing.

The response should retain directionality. Positive response indicates movement toward the controller's target; approximately zero indicates little directed response; negative response indicates movement against the intended direction.

More sophisticated quantities—susceptibility, mutual information, transfer entropy, and information-response efficiencies—can formalize different aspects of this basic resource-response question. The project does not need to commit to a single sophisticated efficiency definition before the experimental system itself has been validated.

## I. Statistical validation before scaling

Before committing to expensive LLM runs, we should test the proposed estimators on a cheap stochastic reference system.

The reference simulation should generate a known control effect and allow much larger sample sizes than the LLM experiment. We can then subsample it at different numbers of episodes and apply exactly the same analysis pipeline intended for the LLM study.

This gives an empirical answer to a practical question: **is 100 repetitions per condition enough to resolve the effect we care about?**

The resulting sample-size requirement should determine the final number of LLM episodes rather than the other way around.

