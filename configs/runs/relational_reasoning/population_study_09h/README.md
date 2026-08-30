# Study 09h: false-target high-statistics persistence

This is the false-target half of the focused L=3 persistence-crossover family.
Study 09d already mapped the broad rho dependence, so this study concentrates
power at `rho={0.80,0.85}`. It keeps frozen
`n12_L3_r03_k3/task_0002`, fixes evidence selection to `strategic`, and varies
only `q={1,2}` and `b={3,4,6,8,9,12}`. The receiver is always naive.
Each of the 24 structural cells targets 15 deterministic repetitions (360
episodes).

The target is the established false semantic `NORTHWEST`; the frozen truth is
`NORTH`. Both evidence policies expose only real frozen-task facts through the
production selector/renderer. Production control replaces exactly one social
slot: q=1 leaves no ordinary peer and q=2 leaves exactly one ordinary peer.

Analysis is empirical-only and permanent tables are CSV. It emits cell-level
chi, T_pi, eta_IR, signed/bounded thermodynamic efficiency diagnostics,
target/truth/active-phi endpoints, and false-takeover fractions. State-local
chi, T_pi, and eta_IR use eight bins of `x=target_count_before/N`, publish a
full categorical support grid, and are paired with occupancy heatmaps.
State-local eta_th is deliberately absent because the repository defines it as
a finite-horizon cell quantity. Matched tables and heatmaps report `q=2 - q=1`
effects separately for every budget and rho under strategic evidence. The
rho-marginal tables are explicitly labelled as descriptive summaries over only
the two focused rho values.

Study 09d is only a reuse candidate for its eight overlapping q=2/strategic
cells at the four legacy budgets and two focused rho values. At most 80 of its
episodes can be reused. Reuse must be validated from sealed Potsdam artifacts
before submission; source YAML alone never authorizes reuse.
