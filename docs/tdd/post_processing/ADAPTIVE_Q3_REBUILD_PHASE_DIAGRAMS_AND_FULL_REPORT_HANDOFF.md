# Handoff: Rebuild the Adaptive-Communication q=3 Analysis and Full LaTeX Report

**Study:** `musr_blackboard_adaptive_communication_q3_deepinfra`  
**Input archive:** `musr_blackboard_adaptive_communication_q3_deepinfra_analysis(1).zip`  
**Goal:** reconstruct the broken state-resolved phase diagrams directly from the Parquet data and regenerate the full adaptive-communication analysis report with all aggregation levels.

---

# 1. Scope

This is an **analysis/reporting task only**.

Do **not** modify:

- the game;
- controller dynamics;
- prompts;
- persistence;
- q or q_c;
- task generation;
- estimators in the production repository;
- any historical study outputs.

Work from the existing analysis ZIP and produce a new derived analysis/report directory.

The bundled state-local phase-map images are currently gray because the exported phase-map tables are incorrectly marked as `structural_cell_not_run` / unsupported. Treat this as an **analysis/export bug**, not as evidence that the underlying state-resolved estimates are absent.

The main task is to reconstruct the phase diagrams directly from the underlying estimator tables, especially `primary_estimates.parquet`, rather than trusting the broken pre-rendered phase-map export.

---

# 2. Environment setup

The analysis environment must be able to read Parquet.

If necessary install:

```bash
pip install pyarrow pandas numpy scipy matplotlib
```

Optionally also:

```bash
pip install fastparquet
```

Use `pyarrow` as the preferred Parquet backend.

Confirm before proceeding:

```python
import pandas as pd
import pyarrow
pd.read_parquet(...)
```

works on the archive tables.

Do not convert Parquet to CSV until after reading it successfully unless needed for user-facing exports.

---

# 3. Unpack and validate the archive

Unpack:

```text
musr_blackboard_adaptive_communication_q3_deepinfra_analysis(1).zip
```

Inspect at least:

```text
validation.md
validation.json
analysis_manifest.json
analysis_recipe.yaml
reports/methods.md
reports/summary.md
provenance/*.yaml
tables/*.parquet
```

Current known validation state:

```text
expected episodes: 750
completed episodes: 723
missing episode keys: 27
failed episodes: 0
aborted episodes: 0
round rows: 7230
micro-slot rows: 173520
```

The report must clearly mark the study as **incomplete / provisional** until the missing 27 episodes are filled.

Do not silently treat 723/750 as complete.

---

# 4. Recover the exact study configuration

Read the resolved configs in `provenance/`.

Confirm and report the main parameters:

```text
task: MuSR Team Allocation task_001
N = 24
rounds = 10
q = 3
q_c = 12
beta = 4.0
theta = 0.5
board lifetime = 1 round
participant REQUEST enabled
controller mode = adaptive_communication
controller timing = dawn_only
controller REQUEST enabled
controller DIRECTIVE enabled
controller policy = contextual_weighted_v1
rho in {0.70, 0.775, 0.85, 0.925, 1.00}
b in {3, 6, 9, 12, 15, 18, 21}
10 repetitions planned per cell
prompt = relational_blackboard_ballot@4
model = deepseek-ai/DeepSeek-V4-Flash via DeepInfra
```

Read all three arms:

```text
no control
truth control
false control
```

and verify target semantics from the actual config rather than assuming them.

---

# 5. Diagnose the broken state-local export

Inspect:

```text
tables/state_local_phase_maps.parquet
tables/state_resolved_x_b.parquet
tables/rho_aggregated_state_local_maps.parquet
tables/state_occupancy_binned.parquet
tables/rho_aggregated_state_occupancy.parquet
```

Confirm why the existing plots are gray.

Then inspect:

```text
tables/primary_estimates.parquet
```

The prior analysis indicates that the underlying state-resolved estimates exist there even though the derived phase-map export is broken.

Document:

- available estimator names;
- resolution labels;
- state/bin fields;
- target semantics;
- rho;
- intervention budget;
- support fields;
- bootstrap fields;
- permutation/null fields;
- observation counts.

Do not use the broken `phase_status` flag from the faulty export as the source of truth if the primary estimator records contain valid state-local estimates.

---

# 6. Reconstruct the state coordinate x

Use the same definition as the existing analysis:

```text
x = target fraction
```

Use the existing target-fraction bins if already encoded in the primary estimator metadata.

Do not invent new bin boundaries unless absolutely necessary.

If the primary estimator rows contain:

```text
target_fraction_bin_index
target_fraction_bin_lower
target_fraction_bin_upper
target_fraction_bin_center
target_fraction_bin_count
```

use those exact bins.

Preserve occupancy/support information for every `(x, b, rho, arm)` cell.

---

# 7. Metrics to reconstruct

Rebuild state-resolved maps for at least:

\[
T_\pi(x,b)
\]

\[
\chi(x,b)
\]

\[
\eta_{\rm IF}(x,b)
\]

\[
\eta_{\rm IR}(x,b)
\]

and occupancy/support.

Also reconstruct the finite-sample null comparison for transfer information:

\[
T_\pi^{\rm excess}(x,b)
=
T_\pi(x,b)-T_{\rm null}(x,b).
\]

If the state-local estimator records contain permutation p-values, also provide a significance/support map.

Do **not** fabricate `eta_th`.

If `eta_th` remains unsupported because `h` cannot be calibrated for the blackboard actuator, report that explicitly.

---

# 8. Required aggregation levels

This is the most important requirement.

For each principal metric, produce **all relevant aggregation levels**.

## A. Persistence-resolved state maps

For each:

```text
rho = 0.70
rho = 0.775
rho = 0.85
rho = 0.925
rho = 1.00
```

produce separate `x × b` maps for:

```text
truth control
false control
```

for:

```text
T_pi
T_pi - T_null
chi
eta_IF
eta_IR
occupancy / n_observations
```

Use a consistent x- and b-axis across rho panels.

---

## B. rho-aggregated state maps

Pool across all rho values using a clearly documented weighting rule.

Prefer observation weighting / the same weighting used in the existing analysis.

Produce `x × b` maps for:

```text
truth control
false control
```

for all principal metrics.

Label these explicitly as:

```text
rho-aggregated descriptive maps
```

Do not imply they replace the rho-resolved analysis.

---

## C. Truth + false control pooled maps

Create a third target-semantic aggregate:

```text
truth control + false control
```

This should be interpreted only as:

```text
control aggregated over target semantics
```

not as evidence that truth and false control are behaviorally symmetric.

Produce:

```text
T_pi(x,b)
T_pi - T_null(x,b)
chi(x,b)
eta_IF(x,b)
eta_IR(x,b)
```

for:

1. each rho separately;
2. rho aggregated.

Use observation weighting / explicit stratified weighting and state it in the report.

---

## D. Whole-cell rho × b phase diagrams

Using the whole-cell/cell-level estimates, create:

```text
rho × b
```

heatmaps for:

```text
final p_truth
final p_target
T_pi
T_pi - T_null
chi
eta_IF
eta_IR
P(U=1)
realized controller posts per active round
controller exposure
```

for truth and false control separately.

Where meaningful, also create a pooled truth+false-control view.

---

## E. Aggregated line plots versus b

For each metric, create:

1. one line per rho versus b;
2. an all-rho aggregate versus b.

Do this for at least:

```text
final p_truth
final p_target
T_pi
T_pi - T_null
chi
eta_IF
eta_IR
P(U=1)
controller posts per active round
controller exposure
active evidence coverage
historical evidence coverage
peer exposure
REQUEST rate
REPORT rate
DIRECTIVE rate
```

Use uncertainty bands where available.

---

# 9. Uncertainty

Use the existing whole-episode bootstrap philosophy.

For:

```text
chi
eta_IF
eta_IR
T_pi
```

report 95% whole-episode bootstrap confidence intervals where the archive contains the necessary bootstrap outputs.

Do not bootstrap individual rounds independently.

For `eta_IR`, use the already-derived bootstrap output if present; if recomputing, recompute the whole nonlinear estimator inside each bootstrap replicate.

For `T_pi`, always report the permutation/null information alongside the bootstrap interval.

Do **not** interpret "bootstrap CI excludes zero" as sufficient evidence for transfer information because the direct-count CMI has a positive finite-sample null.

The preferred presentation is:

```text
raw T_pi
permutation null
T_pi - null
permutation p-value / significance where available
```

---

# 10. Communication-mode analysis

The adaptive controller chooses among:

```text
REQUEST
REPORT
DIRECTIVE
```

conditional on binary:

```text
U = 1
```

Keep the theoretical control variable binary.

Analyze the secondary communication-mode diagnostics.

At minimum report:

```text
P(REQUEST | U=1)
P(REPORT | U=1)
P(DIRECTIVE | U=1)
```

for:

```text
truth control
false control
```

and preferably by:

```text
rho
b
state x / target support
```

where support permits.

Also compute mode-conditioned immediate response:

```text
mean Δ target share after REQUEST
mean Δ target share after REPORT
mean Δ target share after DIRECTIVE
```

for both truth and false control.

This is secondary mechanistic analysis and must not replace binary U in T_pi.

---

# 11. Realized budget / saturation analysis

This adaptive study has an important distinction:

```text
nominal b != realized number of controller posts
```

because REQUEST and DIRECTIVE often use fewer messages than a REPORT action.

Plot and report:

```text
nominal b
vs
actual controller posts per active round
```

by rho and arm.

Also report:

```text
controller posts per active round
controller reads/exposures
unique readers
target adoptions
target adoption per exposure
exposure per controller post
```

where available.

Highlight saturation if realized posts stop increasing substantially with nominal b.

This matters for interpreting b as a control-cost / information-dose coordinate.

---

# 12. Cooperation / information-seeking diagnostics

Because participant REQUEST is enabled in this study, analyze ordinary-agent requests explicitly.

Report at least:

```text
participant REQUESTs per round
participant REPORTs per round
NONE/no-post rate if available
reply rate to REQUEST
evidence acquisition following REQUEST
peer exposure
controller exposure
```

If requests remain rare, say so.

If they are common, characterize how they vary with:

```text
rho
b
arm
state x
```

This is important because later experiments will turn participant questions ON/OFF as a cooperation axis.

---

# 13. Outcome analysis

For each `(arm, rho, b)` cell report:

```text
n completed episodes
initial p_truth
final p_truth
initial p_target
final p_target
final plurality / winner rate
false-target plurality rate for false control
truth plurality rate
```

Include no-control baselines by rho.

Use them to construct:

```text
effect relative to matched no-control baseline
```

for truth and false control.

Do not pool over rho before showing the rho-resolved behavior.

---

# 14. Comparison to the prior truthful-report-only study

If the prior report/results are available in the same workspace, include a clearly labeled **descriptive comparison** against:

```text
musr_blackboard_truthful_reports_q3_deepinfra
```

Compare at least:

```text
final false-target share
false plurality rate
P(U=1)
posts per active round
mean chi
T_pi - null
mode-specific communication where applicable
```

However, explicitly state that this is not a perfectly controlled ablation if:

```text
prompt version differs
initialization differs
other protocol details differ
```

Prefer within-study no-control normalized effects when making stronger comparisons.

---

# 15. Plot styling

Use the same scientific style as the previous MuSR reports.

Requirements:

- actual numeric heatmaps, not gray placeholders;
- blank / hatched cells only for truly unsupported states;
- common axes across comparable panels;
- color bars with metric units;
- occupancy/support maps adjacent to state-resolved estimator maps;
- clear truth vs false control labeling;
- clear rho labels;
- clear indication when a figure is rho aggregated;
- confidence bands on line plots;
- no misleading interpolation across unsupported states.

For `chi`, use a diverging scale centered at zero where appropriate.

For `T_pi - T_null`, also use a diverging scale centered at zero.

For raw `T_pi`, use a nonnegative scale.

---

# 16. Required tables to export as CSV

Export user-readable CSVs in the new report directory:

```text
cell_metrics.csv
rho_b_outcomes.csv
state_local_metrics.csv
state_local_rho_aggregated.csv
state_local_truth_false_pooled.csv
communication_mode_summary.csv
budget_realization_summary.csv
uncertainty_summary.csv
```

Include enough identifiers to reproduce every plot:

```text
arm
target_semantics
rho
b
x_bin
metric
estimate
ci_low
ci_high
null_estimate
p_value
n_observations
support_status
```

where applicable.

---

# 17. LaTeX report

Produce a full LaTeX report and compiled PDF.

Suggested output:

```text
musr_adaptive_q3_rebuilt_report/
    adaptive_communication_q3_report.tex
    Adaptive_Communication_q3_Analysis_Report.pdf
    figures/
    tables/
```

The report should be plot-heavy and organized approximately as:

```text
1. Executive summary
2. Validation / missing episodes
3. Experimental parameters
4. Behavioral outcome phase diagrams
5. rho-resolved outcomes
6. rho-aggregated outcomes
7. T_pi phase diagrams
8. T_pi-null phase diagrams
9. chi phase diagrams
10. eta_IF phase diagrams
11. eta_IR phase diagrams
12. occupancy/support maps
13. truth+false pooled control maps
14. line plots versus b
15. adaptive mode-selection behavior
16. mode-conditioned immediate response
17. realized budget and saturation
18. participant cooperation / REQUEST behavior
19. active and historical evidence coverage
20. comparison with truthful-report-only study
21. interpretation
22. limitations
23. recommended next experiments
```

---

# 18. Main interpretation to test, not assume

The prior diagnostic read suggested:

```text
adaptive communication trades raw steering power
for information-seeking and coordination
```

and that this is especially damaging to false control because REQUEST can expose corrective evidence.

Do not simply repeat this conclusion.

Test it against the reconstructed tables and plots.

In particular check whether:

```text
REPORT has the strongest target-aligned immediate effect
REQUEST weakens false-target steering
DIRECTIVE is intermediate
adaptive false-control chi is weaker than report-only chi
nominal b saturates in realized communication
T_pi remains near its permutation null
```

If the data disagree, report the data rather than preserving the previous narrative.

---

# 19. Important statistical caution for T_pi

The previous study showed a large finite-sample baseline for the direct-count conditional MI.

Therefore the report must not present raw:

\[
T_\pi = I(U;n_{k+1}\mid n_k)
\]

alone as evidence of causal/control information transfer.

Always pair it with:

\[
T_{\rm null}
\]

and preferably:

\[
T_\pi - T_{\rm null}.
\]

If state-local null estimates are not available, say so explicitly rather than inventing them.

---

# 20. Thermodynamic efficiency

Do not fabricate:

```text
eta_th
```

If the archive still reports:

```text
unsupported
insufficient support for h calibration
```

retain that.

The adaptive communication actuator changes the effective controlled kernel, and the old direct-actuation affinity calibration should not be silently reused.

Show:

```text
J_c
I_sens
other supported quantities
```

if they exist, but keep `eta_th` unavailable until a valid `h` calibration exists.

---

# 21. Final acceptance checklist

Before delivering the report confirm:

```text
[ ] Parquet tables loaded successfully
[ ] broken gray phase maps diagnosed
[ ] x×b maps reconstructed from primary estimator records
[ ] truth-control maps produced
[ ] false-control maps produced
[ ] rho-resolved maps produced
[ ] rho-aggregated maps produced
[ ] truth+false pooled maps produced
[ ] occupancy/support maps produced
[ ] raw T_pi shown with null
[ ] T_pi - null maps produced where possible
[ ] chi maps produced
[ ] eta_IF maps produced
[ ] eta_IR maps produced
[ ] uncertainty shown
[ ] rho×b whole-cell phase diagrams produced
[ ] line plots versus b produced
[ ] adaptive communication modes analyzed
[ ] realized budget saturation analyzed
[ ] participant REQUEST/cooperation diagnostics analyzed
[ ] incomplete 723/750 status clearly stated
[ ] eta_th not fabricated
[ ] LaTeX compiles
[ ] PDF visually inspected
```

---

# 22. Deliverables

At completion provide:

```text
results/report directory
PDF path
LaTeX source path
all figure paths
all exported CSV paths
```

Also give a concise summary of:

```text
1. strongest behavioral result
2. strongest chi result
3. T_pi vs null result
4. whether adaptive communication is weaker/stronger than report-only
5. whether REQUEST appears protective against false control
6. whether b saturates in realized control
7. whether any apparent phase/crossover structure is visible in rho
8. what should be run next
```

Do not launch new game episodes from this task.
This handoff is analysis/report reconstruction only.
