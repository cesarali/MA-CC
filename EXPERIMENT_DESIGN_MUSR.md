# MuSR Experiment Design

Here is the current conceptual state of the project and what we are trying to do now.

The main scientific goal is to study the efficiency of feedback control in a population of language-model agents, but in a way that stays compatible with the stochastic-thermodynamic theory we have already developed. The controller variable remains binary:

$$
U_k \in \{0,1\}
$$

where $U_k=0$ means the controller stays silent and $U_k=1$ means it intervenes. This is important because the quantities we care about theoretically, especially

$$
T_\pi = I(U_k ; n_{k+1} \mid n_k),
$$

the susceptibility $\chi$, the information-efficiency quantities, and the binary information bound, are all formulated around a binary intervention variable. We therefore do **not** want REQUEST, REPORT, or DIRECTIVE to become separate values of $U$. Instead, when $U=1$, the controller may choose **how** to realize the intervention. The communication mode is secondary internal structure inside the controlled kernel $Q_1$.

The first blackboard population study taught us that the previous controller was not ideal scientifically. It mostly inserted repeated factless directives onto the board. Increasing $b$ therefore often meant increasing controller occupancy or attention pressure rather than injecting genuinely new information. This produced some interesting control effects, but it was difficult to interpret because the controller was effectively “spamming” the board. We are now replacing that with a more natural communication controller that behaves more like an ordinary participant. Conditional on $U=1$, it may REPORT true evidence, REQUEST missing information, or, if directives are enabled, issue a coordination DIRECTIVE. The controller should decide which mode is useful from the information it is allowed to observe. Importantly, $U$ itself stays binary.

A central new idea is to study control through truthful selective disclosure. In the adversarial condition, the controller is never allowed to lie or fabricate evidence. Instead, it may strategically choose true facts that make a configured false target appear plausible under partial information. Other true, decisive facts should be able to correct this partial picture and recover the actual answer. The intended epistemic structure is therefore:

```text
ambiguous private information
    ->
controller selectively exposes true but target-compatible evidence
    ->
false target becomes more plausible
    ->
decisive peer evidence appears
    ->
truth is recovered
    ->
full information gives the unique correct answer.
```

This is why task generation became a major issue. The old MuSR Team Allocation tasks were not automatically suitable for this experiment. We need tasks with a very specific probability geometry. Private evidence must remain ambiguous; controller-reportable true facts must increase the plausibility of the false target without making truth impossible; decisive evidence must repair the misleading partial picture; and full evidence must uniquely recover the gold answer. We therefore added an exact symbolic preflight before language generation. For each candidate hidden world we enumerate compatible completions and compute the posterior over candidate allocations under ZERO, PRIVATE, C3, C6, C9/C12, DECISIVE, controller+decisive, and FULL evidence states.

The symbolic preflight worked: out of 10,000 candidate worlds, 611 passed all the required gates without changing thresholds. We then selected frozen development tasks and generated natural-language evidence with Terra. This exposed another problem: equality propositions were difficult to express faithfully in natural language. Terra often generated evidence compatible with equality but not logically strong enough to establish equality. We fixed this by deterministically rendering equality evidence in a canonical form rather than weakening the semantic audit. We also discovered that controller fact pools could be logically distinct but still informationally redundant. Some 24-fact controller pools contained many zero-marginal or implied facts. We therefore changed the strategic ranking to prefer new latent-variable coverage, new predicate families, positive marginal false-target lift, and strict reduction in compatible worlds. This produced much cleaner informative prefixes. Because information largely saturated before 24 reports, the currently recommended information-budget grid is:

$$
b \in \{3,6,9,12\}
$$

with $b=24$ retained, if needed, only as a saturation diagnostic rather than interpreted as 24 independent increments of information.

We also found an important prompt-rendering issue caused by randomized option ordering. Votes should be stored semantically, e.g. `ALLOCATION_0/1/2`, and only converted to A/B/C when rendering the prompt for the current recipient. Otherwise an old public message may contain a stale option letter that meant something different for the sender. We are fixing this so the structured vote is always rendered according to the recipient's current option mapping. We are also minimally cleaning the blackboard prompt: public reports should describe allocations semantically rather than using A/B/C in free text; verified shared facts should be distinguished from a participant's interpretation; duplicated rendering of the same fact should be avoided; “current position” should be framed as a previous vote that may be revised; and REQUEST should be explicitly encouraged when important evidence is missing or ambiguous.

The other major experimental axis we now want is cooperation. Ordinary agents can be run with REQUEST disabled or enabled. This gives us a population-level cooperation variable:

$$
Q = 0 : \text{ordinary participants cannot ask questions}
$$

$$
Q = 1 : \text{ordinary participants may actively request missing information.}
$$

Separately, controller directives can be disabled or enabled:

$$
D = 0 : \text{controller cannot use DIRECTIVE}
$$

$$
D = 1 : \text{controller may use DIRECTIVE.}
$$

This gives a clean $2\times2$ communication design:

| Questions ($Q$) | Directives ($D$) | Condition |
|---:|---:|:---|
| 0 | 0 | Q0 D0 |
| 1 | 0 | Q1 D0 |
| 0 | 1 | Q0 D1 |
| 1 | 1 | Q1 D1 |

The same binary controller $U$ is used in every condition. When $U=1$, the controller chooses among the communication modes allowed by that condition. For example, without directives it may choose REPORT or REQUEST; with directives enabled it may choose REPORT, REQUEST, or DIRECTIVE. This allows us to ask whether control becomes more or less efficient when the population itself is more cooperative and information-seeking, and whether coordination directives help beyond truthful evidence exchange.

The theoretical interpretation should remain simple. The main controlled dynamics are still represented by two effective kernels:

$$
Q_0 = \text{uncontrolled dynamics}
$$

$$
Q_1 = \text{dynamics when the controller acts.}
$$

REQUEST, REPORT, and DIRECTIVE are internal realizations of $Q_1$. They can be logged as secondary diagnostics, but they do not replace the binary $U$ in the primary theory. This lets us preserve

$$
T_\pi = I(U_k ; n_{k+1} \mid n_k)
$$

and the existing information-bound framework while making the controlled dynamics much richer.

## Immediate Plan

1. Finish the prompt/rendering patch and regression tests.
2. Run the isolated OSS behavioral calibration on the new frozen tasks using the exact patched production prompt.
3. Verify that ZERO/PRIVATE remain ambiguous, controller-selected true evidence raises false-target choice, decisive evidence restores truth, and FULL evidence solves the task reliably.
4. If that passes, run a small population pilot with the adaptive binary controller.
5. Then scale to the main study over persistence $\rho$, budget $b$, population state $x$, cooperation/questions $Q$, and directives $D$.

## Core Scientific Question

> How does the efficiency and information transfer of binary feedback control change when the multi-agent communication structure changes?

In particular, we want to separate:

- whether the controller acts at all;
- how the active intervention is communicated;
- how cooperative/information-seeking the population is;
- how much truthful information the controller injects;
- and how persistence/memory changes the controllability of the population.

The final target is still a state-resolved control-efficiency picture, including $T_\pi$, $\chi$, information efficiency, and eventually thermodynamic efficiency, but now under a much more meaningful communication mechanism than the original directive-spam controller.
