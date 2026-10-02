# `analysis/` — offline analysis projects

Each subdirectory is one self-contained study: it reads archived simulation
output, computes something, and produces a report. Nothing here runs
simulations or talks to a model provider.

| project | question |
|---|---|
| [`preanalysis_task003_new_setup_and_coarsegraining/`](preanalysis_task003_new_setup_and_coarsegraining/) | What is `task_003`? Why was the first controller unfair? What state representation makes information quantities estimable? |
| [`bound_estimators/`](bound_estimators/) | Control-efficiency bounds: endpoint and path information, cost, efficiency ratios, estimated with learned critics |

## Why this is not `results/`

`results/` is for generated artifacts. These projects are *code plus prose plus
findings*, and they were getting lost: the whole of `results/` is gitignored, so
none of this analysis reached the remote and none of it reached the other four
collaborators.

## What is versioned and what is not

The line is **regenerable bulk**, not folder names:

| | versioned |
|---|---|
| scripts and library code | yes |
| prose: `README.md`, dated reports, `narrative.md` | yes |
| LaTeX sources (`report/main.tex`) | yes |
| small finding tables (`*.csv`, `*.json`, a few KB each) | yes |
| the final report PDF | yes |
| `<project>/**/results/` — parquets, job caches, extracted data, run logs | **no** |
| `report/figures/`, `report/main.pdf`, LaTeX intermediates | **no** |

The rules live in the repository root [`.gitignore`](../.gitignore) under the
`analysis/` heading.

The reason the final PDF *is* versioned: both studies read inputs that
collaborators do not have — a read-only archive under
`/Users/rsanchez/Projects/agents_control` in one case, 94 MB of local job caches
in the other. If the PDFs were ignored too, this directory would ship scripts
and no findings.

## Where the code lives

`bound_estimators` is a real installed Python package at
[`src/bound_estimators/`](../src/bound_estimators/) — `src/rnd_init_metrics/`
and `tests/bound_estimators/` import it, and `pyproject.toml` discovers packages
under `src/`. It stays there. Only its *outputs* live here.

`preanalysis_task003_new_setup_and_coarsegraining` has its own `scripts/`
because they are one-off analysis scripts, not a library.
