# Study 08: Epistemic Vigilance and Selective Evidence in Population Control

## Short summary

Study 08 asks whether an external controller can steer a population of LLM
agents differently depending on two epistemic mechanisms:

1. whether the receiving agents are **naive** or **vigilant** about potentially
   selective communication; and
2. whether the controller supplies a **neutral** fact or strategically selects
   a true fact that favors its target.

These mechanisms are tested independently under both a **truth-aligned target**
and a **false target**. The study therefore measures not only how controllable
the population is, but whether its response is *semantically selective*: can it
remain responsive to useful, truth-directed control while resisting control
toward an incorrect answer?

## What is being manipulated?

Study 08 has three main experimental factors:

| Factor | Conditions | Meaning |
|---|---|---|
| Receiver disposition | `naive`, `vigilant` | Vigilant agents are explicitly prompted to check whether evidence supports a recommendation and to consider that a sender may have another objective. |
| Controller evidence strategy | `neutral`, `strategic` | Neutral evidence is selected independently of the target. Strategic evidence is a real task fact selected because it is most favorable to the controller's target. |
| Controller target | `truth`, `false` | The controller either promotes the correct answer or a fixed incorrect answer. |

Every controlled condition uses `recommendation_plus_fact`. There are no
recommendation-only cells in Study 08. Crucially, the controller never invents
or falsifies evidence: even in the false-target condition, strategic evidence
is a true fact from the frozen task. The potentially manipulative mechanism is
therefore **selective disclosure of truthful but incomplete evidence**.

Crossing receiver disposition and evidence strategy creates four epistemic
conditions:

- `naive_neutral`
- `naive_strategic`
- `vigilant_neutral`
- `vigilant_strategic`

Each is evaluated for truth and false targets across intervention budgets
`b = 4, 8, 12, 16, 20, 24`.

## Experimental setting

The experiment uses populations of 24 LLM agents solving frozen relational
reasoning tasks through ten rounds of public voting and information exchange.
The population dynamics, tasks, model, controller policy, sensing, and other
parameters are held fixed so that the intended differences come from receiver
vigilance, evidence selection, target semantics, and control strength.

The complete design contains:

```text
2 receiver dispositions
x 2 evidence strategies
x 2 target semantics
x 6 intervention budgets
x 4 tasks
= 192 scientific cells

192 cells x 10 repetitions = 1,920 episodes
```

Matched grid ordering, a common root seed, and common random numbers make the
comparisons as paired as the execution framework permits.

## What does the study measure?

Study 08 follows both population behavior and knowledge accumulation.

- **Population state:** the fraction of agents currently voting for the
  controller's target, `x`.
- **Epistemic state:** the fraction of agents holding a complete proof, `phi`,
  and the average coverage of supporting facts, `kappa`.
- **Information transfer (`T_pi`):** how informative the controller's action is
  about the next population state after conditioning on the current state.
- **Susceptibility (`chi`):** the state-matched difference in target-share
  change between controller intervention and no intervention.
- **Information-response efficiency (`eta_IR`):** how much useful population
  response is obtained relative to the information used by the controller.
- **Thermodynamic control quantities:** controlled current, effective affinity,
  sensing information, and `eta_th`, which connect the empirical LLM dynamics
  to the revised single-affinity control theory.

The analysis studies these quantities across control budget and at matched
social and epistemic states. It also follows `phi(t)`, `kappa(t)`, truth share,
and target share over time. This separates two possible effects:

1. vigilance or evidence selection changes how knowledge spreads; and
2. it changes how agents respond to control even when their current knowledge
   state is comparable.

## What is Study 08 intended to achieve?

The study turns a broad question about "robust reasoning" into a controlled
population-level experiment. It is designed to determine:

- whether epistemic vigilance reduces susceptibility to strategically framed
  evidence;
- whether that reduction differs for truthful and false recommendations;
- whether selective disclosure of true facts can steer a population toward a
  false conclusion without using fabricated evidence;
- whether vigilance preserves useful response to truth-aligned control rather
  than producing indiscriminate distrust;
- how these effects depend on intervention strength and on the population's
  current social and epistemic state; and
- how empirical LLM behavior departs from a prompt-blind coarse control-theory
  reference.

The key outcome is therefore not simply final-answer accuracy. Study 08 aims to
explain **when, how, and at what information cost** an LLM population becomes
controllable under truthful communication, selective disclosure, and different
receiver reasoning instructions.

## Current status of this explanation

The full design has passed the recorded preflight at 192 cells and 1,920
episodes. This document summarizes the scientific purpose and planned analysis;
it does not claim numerical findings from a completed full-study aggregation.

## Repository references

- [Study 08 configuration README](../../configs/runs/relational_reasoning/population_study_08/README.md)
- [Study 08 analysis recipe](../../configs/runs/relational_reasoning/population_study_08/analysis.yaml)
- [Revised Study 08 design](../tdd/experiments/27062026_study08_revised_recommendation_plus_fact.md)
- [Study 08 preflight report](../../configs/runs/relational_reasoning/population_study_08/PREFLIGHT.md)
