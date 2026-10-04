# Instructions for Claude

## How to talk to me

Explain everything in the simplest words that are still accurate.
Assume I am smart but new to this part of the code, so I understand ideas
quickly but I do not yet know the vocabulary.

### Define every technical term the first time you use it

Never use a technical term, an abbreviation, or a name from this codebase
without saying what it means in the same sentence or the one right after.
This applies to:

- Programming words (for example: closure, mutex, idempotent, monad).
- Library and tool names (for example: pyarrow, SLURM, ruff).
- Names invented inside this repo (for example: a study, a round, a game,
  a recorder). These are the ones I am most likely to be missing, because
  I cannot look them up anywhere outside this project.
- Abbreviations. Write the full words once, then the short form:
  "MI (mutual information, a number saying how much one thing tells you
  about another)".

If a definition would take more than a sentence, give the one-sentence
version first and offer the longer one afterwards.

### Prefer plain words over jargon

If a plain word says the same thing, use the plain word.

- "runs again on its own" instead of "recurses"
- "safe to run twice" instead of "idempotent"
- "the setting is read once when the program starts" instead of
  "resolved at import time"

Use the technical term when it is genuinely more precise, or when I will
see that exact word in the code, the logs, or the documentation. In that
case, give the plain meaning next to it so I can connect the two.

### Explain the why, not only the what

When you describe a change, say what problem it solves and what would go
wrong without it. A change I cannot explain to someone else is a change I
cannot review.

### Use concrete examples

Show a small, real example instead of describing a shape in the abstract.
One sample input and its output beats a paragraph about the input format.

### Keep the structure simple too

- Short sentences. One idea per sentence.
- Short paragraphs. Lists where a list fits.
- Say the answer first, then the reasoning. Do not build up to it.
- No filler like "as you know" or "simply" or "just" — if it were
  obvious, I would not have asked.

### When you are unsure

Say so plainly: "I am not sure whether X, because I have not checked Y."
Do not hide uncertainty behind confident-sounding technical language.

## Repository instructions

The environment, experiment, and workflow rules for this repository live in
[AGENTS.md](AGENTS.md). Follow those as well — this file governs *how you
explain things*, `AGENTS.md` governs *what you do*.

### NERSC Perlmutter environment

When working in this checkout on NERSC Perlmutter, use the existing `MA-CC`
Conda environment. It is physically stored outside the limited home directory
at:

```text
/pscratch/sd/d/dfarough/conda_envs/MA-CC
```

The repository is `/pscratch/sd/d/dfarough/MA-CC`, and Conda's writable
package cache is `/pscratch/sd/d/dfarough/conda_pkgs`. The displayed path
`~/.conda/envs/MA-CC` is a symlink to the physical `/pscratch` environment; do
not create another environment under the home directory.

Load the NERSC Python module in every new shell or agent command group, then
run commands through the named environment:

```bash
module load python/3.11-24.1.0
conda run -n MA-CC python ...
conda run -n MA-CC python -m pytest ...
conda run -n MA-CC mas-cc ...
```

Do not use the system Python or install project dependencies into it. Follow
the fuller cluster and experiment rules in [AGENTS.md](AGENTS.md).

All production work on NERSC must run on Perlmutter CPU compute nodes through
an `salloc` interactive allocation. Always use `--qos=interactive` together
with `--constraint=cpu`. Never use `sbatch`, the `regular` quality of service
(QoS, the scheduler queue policy), or a default/omitted QoS for this project.
Interactive allocations may use one to four nodes for at most four hours, and
each CPU node has 128 physical cores. Use the checked launchers in
`scripts/nersc/`; they refuse any other QoS. Keep results and logs under
`/pscratch/sd/d/dfarough/MA-CC-results`, outside the repository and home
directory. Use `scripts/nersc/start_study_supervisor.sh` for studies that may
cross the four-hour walltime; it detaches from the agent/SSH session and safely
resumes the same prepared result root through successive interactive
allocations.
