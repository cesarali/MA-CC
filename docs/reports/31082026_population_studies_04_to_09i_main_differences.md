# Population Studies 04–09i: What Each Experiment Tested

This document gives only the main scientific difference between consecutive studies.
It describes what the experiment configurations were designed to test. It does not claim that every configured run finished or that the results support a particular conclusion.

## Terms used below

- **Agent:** one simulated participant that reasons and votes.
- **Controller:** the part of the experiment that observes some agents and sends advice to selected agents.
- **False target:** a wrong answer that the controller tries to make popular.
- **Truth target:** the correct answer that the controller tries to make popular.
- **Evidence:** a true task fact attached to the controller's recommendation.
- **Persistence:** how likely an active fact is to remain available in the next round.
- **Social-source count (`q`):** how many other agents' messages an agent receives during an update.
- **Reasoning depth (`L`):** how many linked facts are needed to prove the answer.

## Main progression

| Study | What it tested | Main difference from the previous study |
|---|---|---|
| **04** | Whether control changes when the controller can **observe more agents** or **influence more agents**. | It expanded Study 03's single setting into a grid that separated observation capacity from influence capacity. |
| **05** | The direct effect of one round of controller advocacy by comparing the **same starting population** with advocacy forced on versus forced off. | It replaced Study 04's ten-round changing process with a one-round matched comparison, so the intervention itself could be isolated. |
| **06** | How long-run control efficiency changes with the **number of influenced agents**, the controller's **decision threshold**, and how sharply it reacts around that threshold. | It returned to ten-round experiments and mapped a much broader range of controller-policy settings. |
| **07** | A finer test of controller responsiveness and a comparison between pushing a **false answer** and supporting the **correct answer**. | It extended Study 06 with more responsiveness settings and added a full truth-target experiment. |
| **08** | Whether control depends on how agents treat social information and on how the controller chooses the true fact attached to its recommendation. | It stopped varying the controller's threshold and responsiveness. Instead, it varied **naive versus vigilant agents** and **neutral versus strategically chosen evidence**. |
| **09 (09a)** | A smaller confirmation of Study 08's four agent-and-evidence conditions for both false and truth targets. | It fixed the influence budget at 12 agents and reduced the task set from four tasks to two. |
| **09b** | Whether a strong controller can produce even one final **unique false winner** while citing only true facts. | It changed from a broad comparison to a small existence test in a smaller population with deeper reasoning and stronger control. |
| **09c** | Whether false steering changes when active facts can be forgotten over a longer, 30-round run. | It introduced finite persistence, kept one task, and varied both persistence and influence strength. |
| **09d** | Where the false-target persistence change occurs more precisely and reliably. | It narrowed the persistence spacing and increased repeated runs from 1 to 10 per setting. |
| **09e** | The truth-target version of Study 09d. | The main change was only the target: the controller supported the correct answer instead of a false answer. |
| **09f** | False-target persistence in a simpler communication and reasoning system, including a no-forgetting setting. | It moved from two social sources and three-step reasoning to **one social source (`q=1`)** and **two-step reasoning (`L=2`)**, changed back to a false target, and added persistence `1.0`. |
| **09g** | The truth-target version of Study 09f. | The main change was only the target: correct rather than false. |
| **09h** | A more precise false-target comparison of **one versus two social sources** near the persistence values where behavior appeared to change. | It returned to three-step reasoning, focused on persistence `0.80` and `0.85`, added intermediate influence levels, and used more repetitions. |
| **09i** | The truth-target version of Study 09h. | The main change was only the target: correct rather than false. |

## The sequence in one sentence

The studies moved from asking **how much the controller can observe and influence**, to **how its policy works**, then to **how agents and evidence change its effect**, and finally to **how memory, reasoning depth, social input, and false-versus-truth targets affect long-term control**.

## Configuration sources

- [Study 04](../../configs/runs/relational_reasoning/population_study_04/README.md)
- [Study 05 configurations](../../configs/runs/relational_reasoning/population_study_05/)
- [Study 06](../../configs/runs/relational_reasoning/population_study_06/README.md)
- [Study 07](../../configs/runs/relational_reasoning/population_study_07/README.md)
- [Study 08](../../configs/runs/relational_reasoning/population_study_08/README.md)
- [Study 09a](../../configs/runs/relational_reasoning/population_study_09/README.md)
- [Study 09b](../../configs/runs/relational_reasoning/population_study_09b/README.md)
- [Study 09c](../../configs/runs/relational_reasoning/population_study_09c/README.md)
- [Study 09d](../../configs/runs/relational_reasoning/population_study_09d/README.md)
- [Study 09e](../../configs/runs/relational_reasoning/population_study_09e/README.md)
- [Study 09f](../../configs/runs/relational_reasoning/population_study_09f/README.md)
- [Study 09g](../../configs/runs/relational_reasoning/population_study_09g/README.md)
- [Study 09h](../../configs/runs/relational_reasoning/population_study_09h/README.md)
- [Study 09i](../../configs/runs/relational_reasoning/population_study_09i/README.md)
