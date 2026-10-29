# Analysis pipeline for the 15-agent campaign

After a run (`../run_overnight.sh`), from the repository root:

1. `.venv/bin/python analysis/15_agent_simulation/analysis/aggregate.py analysis/15_agent_simulation/results/<run>`
   for each of the two runs. Pairs every false episode with its silent episode and writes compact per-cell time
   series to `<run>/aggregates/`.
2. `.venv/bin/python analysis/15_agent_simulation/analysis/theory_grid.py` writes `theory_grid.csv`: the mean-field
   fact-flow model (P0, W = 30, ε = 1; nothing fitted) on the campaign grid.
3. `.venv/bin/python analysis/15_agent_simulation/analysis/figures.py --task003 <run> --op <run> --out
   analysis/15_agent_simulation/reports/<date>_<topic>` writes the numbered figures (vote shares, G(t), phase diagrams
   of G and D, crossovers, theory maps, the deduplication effect, retention) and tables.

Report structure (researcher's request, 29 Oct 2026): all four setups; first vote shares against time; then two main
sections, reading reversal and persistence, with numbered figures (including heat-map phase diagrams), numbered
equations and diagrams, compared against the theory.
