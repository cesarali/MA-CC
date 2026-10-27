# theory_blackboard — shared space of the two theory agents

**Purpose.** This folder is where the two agents working on the theory — Claude (own work in `../theory_claude/`) and
ChatGPT/Codex (own work in `../theory_chatgpt/`) — talk to each other and to the researcher. Each agent reports here
what it tried, what it found (good or bad), and what it proposes next, so that the other can check it, build on it, or
argue with it.

**Shared goal.** A **simple phenomenological model** of the control of agent swarms that captures the two main
phenomena of the LLM-free simulator. Understanding matters more than numerical agreement with the simulator.
1. **Reading reversal:** reading more posts helps a false controller when agents forget fast, and protects the swarm
   when they remember.
2. **Persistence:** the controller's effect outlasts its posting, depending on what agents repost.

## Rules

1. **Every note is signed and dated in its file name:** `YYYY-MM-DD_<agent>.md`, for example `2026-10-27_claude.md` or
   `2026-10-27_chatgpt.md`. A second note by the same agent on the same day adds a counter: `2026-10-27_claude_02.md`.
   The first line under the title also names the author.
2. **Notes are not edited after posting.** Corrections and replies go in a new dated note, which refers to the earlier
   one by file name.
3. **Each note says:**
   - what was tried;
   - the result, with numbers;
   - where the files are;
   - what failed;
   - what comes next.

   Derivations should be posted here before either agent relies on them, so the other can check them.
4. **Each agent writes only here and in its own folder.** Neither agent edits the other's folder or notes. Both may read
   everything.
5. Plain language with definitions; equations where they help. Separate mathematical results, numerical
   demonstrations, analogies and comparisons with the simulator.

## Background

- **Researcher's plan for the phenomenological model:** `phenomenological_fact_flow_theory_plan.md`, 27 October 2026
  (shared with the agents directly).
- **State of the earlier, simulator-based theory:**
  `../theory_claude/reports/2026-10-25_phaseABC_for_reviewer/findings_for_reviewer.md`.

*Set up 27 October 2026 at the researcher's request. Notes posted before this README (`2026-10-27_01_...` and
`2026-10-27_02_...` by ChatGPT/Codex) predate the naming rule and are kept as they are.*
