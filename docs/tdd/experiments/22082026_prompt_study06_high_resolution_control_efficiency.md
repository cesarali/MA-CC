# Implementation Prompt: Study 06 — Higher-Resolution Control-Efficiency Experiment

You are working in the existing **MA-CC** repository. Your task is to design and prepare the next relational-population experiment as a more robust successor to **population Study 04**, while preserving the existing scientific semantics and reusing the repository's current execution/analysis infrastructure.

Use the Study 04 configs and documentation as the primary implementation reference, especially:

```text
configs/runs/relational_reasoning/population_study_04/
    relational_population_study04_qc06.yaml
    relational_population_study04_qc12.yaml
    relational_population_study04_qc18.yaml
```

and the existing relational round-feedback analysis / metrics / current-analysis code.

If the new study-level workflow (`mas_cc study submit` / `mas_cc study aggregate`) has already been implemented, structure this experiment to use it. If that implementation is still incomplete, prepare the configs so they are directly compatible with it and use the current execution path only as a fallback.

Do **not** rewrite the scientific estimator machinery.

---

# 1. Scientific goal

Study 04 established a coarse resource grid over sensing size `q_c` and intervention budget `b`. The next experiment should **not** repeat a dense `q_c` sweep.

The purpose of Study 06 is to obtain a higher-resolution map of **control efficiency as a function of actuation strength and feedback-policy shape**, while keeping the sensing resource fixed.

The main experimental plane should be:

\[
(b,\theta)
\]

at fixed:

\[
q=1,\qquad q_c=12,\qquad N=24,\qquad \beta=4.
\]

Interpretation:

- `b`: how much actuation/intervention is available when the controller acts.
- `theta`: when the feedback controller decides intervention is warranted.
- `q_c=12`: fixed intermediate sensing resource, already characterized in Study 04.
- `beta=4`: baseline soft-policy gain from Study 04.

We want enough resolution to detect possible:
- ridges;
- interior optima;
- saturation;
- non-monotone efficiency regions;
- discrepancies between exact/reference theory and the LLM trajectories.

The experiment should retain the same core relational reasoning task family and controller semantics as Study 04 so that changes can be attributed to the new policy/actuation grid rather than to a redesigned environment.

---

# 2. Do not vary these in the main atlas

For the primary experiment, keep fixed:

```text
N = 24
q = 1
q_c = 12
beta = 4
same relational task family / prompt semantics as Study 04
same controller target semantics as Study 04
same recommendation-only control semantics unless Study 04 config says otherwise
same round horizon as Study 04 unless runtime analysis justifies a change
same position-bias / semantic-answer handling
same results_only scientific retention
```

In particular:

- do **not** make `q_c` another main sweep;
- do **not** use `q > 1` in this study;
- do **not** sweep model × b × theta × beta as one giant factorial;
- do **not** change task semantics in the main experiment.

---

# 3. Main atlas: dense `(b, theta)` grid

Use:

```text
b = [4, 8, 12, 16, 20, 24]
theta = [0.20, 0.35, 0.50, 0.65, 0.80]
```

with:

```text
q_c = 12
beta = 4
```

This produces:

```text
6 × 5 = 30 policy/actuation conditions
```

before task/world replication.

The main scientific object is therefore a high-resolution experimental surface over:

\[
(b,\theta).
\]

The central motivation is that `b` and `theta` are genuinely different knobs:

- `b` changes how much actuation is delivered;
- `theta` changes where in state space the feedback policy tends to switch.

This is more informative than another fine sensing-resource grid.

---

# 4. Tasks / worlds and repetitions

Study 04 used only a small number of tasks/worlds. Study 06 should improve robustness across worlds rather than spending all additional budget on repetitions of only two tasks.

Preferred target:

```text
at least 4 relational tasks/worlds
```

with the same task family/difficulty constraints used in Study 04.

All main conditions should use the **same matched task set**.

Use common random numbers across grid cells if the existing experiment framework supports this cleanly and Study 04 semantics allow it:

```yaml
experiment:
  metadata:
    common_random_numbers_across_grid: true
```

Do not add this blindly; verify the current config schema and how Study 04/05 use it.

### Repetition count

The total study should be designed for roughly a **48-hour wall-clock window** on the existing cluster/provider setup.

Do not guess the repetition count.

Before finalizing configs:

1. inspect Study 04 timing information and existing run summaries;
2. inspect the number of calls per episode;
3. run repository preflight / call-budget estimation;
4. estimate wall time, token/request volume, and cost;
5. choose a repetition count that gives good CMI/support statistics while fitting the target window.

Preferred candidate repetition counts are:

```text
8, 10, or 12 repetitions per task × condition
```

Prefer statistical support over making the grid unnecessarily larger.

Do not silently reduce the `(b, theta)` resolution first. If the estimate is too large, reduce repetitions in a documented way and report the resulting expected counts.

---

# 5. Secondary beta ablation

Do **not** multiply the complete main atlas by several beta values.

Instead prepare a targeted beta study at the baseline threshold:

```text
q_c = 12
theta = 0.50
beta = [2, 4, 8]
b = [8, 16, 24]
```

This gives only:

```text
3 × 3 = 9 beta/actuation conditions
```

before task replication.

Scientific question:

> Does changing the sharpness of the feedback decision alter the efficiency structure qualitatively, or does the main behavior persist across soft versus sharper policies?

Use the same task set and, where possible, matched random streams as the main experiment.

This is an ablation, not the main phase diagram.

---

# 6. Optional second-model robustness block

A second model family is scientifically useful, but do **not** repeat the entire Study 06 atlas for multiple models.

Preferred strategy:

1. validate the second model on the relational reasoning task first;
2. only if validation is acceptable, run a sparse anchor grid.

A Qwen-family model is the preferred first alternative if such a model is already supported by the repository/provider configuration.

Do not invent or hard-code a model identifier that is not already supported. Inspect the repository's provider/model registry and report the valid option.

### Required validation before control study

Use the established relational reasoning validation logic:

- full-information condition should perform near ceiling;
- partial-information and zero-information conditions should behave sensibly;
- parser/provider failures should be low;
- answer-position/semantic-shuffle handling must remain correct.

If the candidate model fails this validation, do not interpret its control efficiency.

### Sparse second-model anchor grid

If validation passes, use:

```text
q_c = 12
beta = 4

b = [8, 16, 24]
theta = [0.35, 0.50, 0.65]
```

This is a:

```text
3 × 3 = 9 condition
```

robustness grid.

The purpose is qualitative model-family generalization, not a full factorial replication.

Do not add Gemma and Qwen simultaneously unless budget analysis clearly shows this is cheap enough after the primary experiment is secured.

---

# 7. Analysis requirements

Reuse all existing Study 04 relational analysis machinery that is scientifically applicable.

The output should make it easy to construct, at minimum:

## Primary control/information quantities

```text
signed target response / susceptibility proxy
target actuation CMI / state-local action information
sensing MI
controller action entropy / conditional action entropy
eta_IR where supported
population / truth / order CMI if already configured
```

## Memory / epistemic conditioning

Retain the existing memory-aware hierarchy, e.g. quantities corresponding to:

\[
I(U_k;n_{k+1}\mid n_k),
\]

and the richer versions already implemented in the repository using combinations of:

```text
previous target-state band
phi
kappa
susceptible / epistemic state
```

Do not invent new estimators if equivalent ones already exist.

## Epistemic observables

Retain:

```text
kappa
phi
truth share
controller-target share
order / vote entropy observables already used
```

## Microscopic control calibration

Where the micro-slot data support it, preserve the ability to estimate:

\[
p_+ = P(\text{non-target}\to\text{target}\mid\text{controlled}),
\]

\[
p_- = P(\text{target}\to\text{non-target}\mid\text{controlled}),
\]

and therefore:

\[
h_{\rm eff}=\log(p_+/p_-),
\]

\[
\gamma_{\rm eff}=p_++p_-.
\]

These are **measured outcomes**, not experimental sweep parameters.

## Currents / thermodynamic quantities

Reuse the existing current analysis and any already-implemented finite-time quantities needed for:

```text
J_c
Delta S_sys
Sigma
```

if the current code supports them cleanly.

Do not resurrect the discarded `h_c / h_e / load` decomposition.

---

# 8. Statistical requirements

Information-theoretic estimates must retain the existing statistical machinery:

- episode-level bootstrap confidence intervals;
- policy-conditional randomization nulls for actuation information;
- sensing-permutation nulls for sensing MI;
- unsmoothed / Jeffreys / Miller-Madow variants where already supported;
- support diagnostics;
- dual-action coverage;
- conditioning-state sparsity.

Do not report finite CMI values without their support diagnostics.

Do not average CMI values across arbitrary execution shards.

Any pooled CMI must be calculated from the pooled canonical scientific observations.

---

# 9. Desired automatic plots

The new standardized aggregation pipeline should automatically prepare a broad exploratory plot set.

At minimum request:

## Main `(b, theta)` atlas

For each applicable observable:

```text
x-axis: b
y-axis: theta
```

Produce surfaces/heatmaps for:

```text
signed response
target CMI / T_pi-like empirical quantity
eta_IR
action entropy
sensing MI
final target share
final truth share
kappa
phi
h_eff
gamma_eff
J_c
Sigma if available
```

## State-local views

Where statistical support permits, also prepare:

```text
(x, b) at selected theta
(x, theta) at selected b
```

for:

```text
signed response
CMI
eta_IR
memory-conditioned CMI
```

## Memory comparison

Prepare direct comparisons of:

```text
current state only
+ previous-state/history conditioning
+ history + phi
+ other already-supported epistemic conditioning
```

with support masks.

## Beta ablation

Prepare:

```text
(beta, b)
```

plots for the main efficiency / response / information quantities.

## Second model

If run, produce identical anchor plots for:
- baseline model;
- second model;

and a qualitative comparison using the same axes/scales where sensible.

---

# 10. Primary versus exploratory figures

It is acceptable and desirable for offline post-processing to generate many candidate plots because this costs no additional LLM calls.

However, keep a small predeclared primary analysis family so that the final paper does not look like post-hoc pattern fishing.

Treat the following as primary candidates:

```text
signed response / susceptibility
target actuation CMI
eta_IR
memory-conditioned target CMI
h_eff
gamma_eff
```

Everything else may be produced as exploratory/supporting diagnostics.

The goal is to discover whether the LLM experiment develops non-monotone or interior efficiency structure that is absent or qualitatively different in the simple reference theory, but do not manufacture such a pattern if the data remain monotone.

---

# 11. Storage and reproducibility

Use:

```yaml
storage:
  artifact_profile: results_only
```

unless a small validation run explicitly requires richer debugging output.

Preserve:
- scientific Parquet;
- round trajectories;
- micro-slot trajectories;
- resolved configs;
- overrides;
- completion seals;
- hashes/provenance;
- configured analysis outputs.

Do not rely on Comet as the authoritative scientific store.

---

# 12. Study-level organization

If the new study workflow exists, prepare a folder such as:

```text
configs/runs/relational_reasoning/population_study_06/
    study.yaml
    study06_main_b_theta.yaml
    study06_beta_ablation.yaml
    study06_second_model_validation.yaml
    study06_second_model_anchor.yaml
    analysis.yaml
    README.md
```

Use fewer/more YAMLs only if the existing config restrictions require it.

Remember:

- a YAML config is scientific design;
- a SLURM `.job` is infrastructure.

Do not create `population_study06.job`.

Use the generic config-array submission mechanism.

---

# 13. Preflight and 48-hour resource report

Before launching any provider-consuming run, produce a short report containing:

```text
number of configs
number of grid cells per config
number of tasks
repetitions
total episodes
rounds per episode
estimated provider calls
estimated token volume if available
estimated cost if available
estimated wall-clock based on Study 04 timing
effective concurrency under the proposed SLURM array throttle
```

Explicitly verify that the proposed array throttle × process parallelism × provider request concurrency is safe for the provider limits.

If the estimated main + beta + second-model study does not fit comfortably in the ~48-hour target, prioritize in this order:

1. main `(b, theta)` atlas;
2. beta ablation;
3. second-model validation;
4. second-model anchor run.

Do not sacrifice the main atlas to preserve optional model breadth.

---

# 14. Do not launch automatically

Prepare:
- configs;
- study manifest;
- analysis recipe;
- README;
- preflight/budget summary;
- exact submission command(s).

Do **not** start the paid/provider experiment unless explicitly instructed to do so.

---

# 15. Deliverables

Return a concise implementation summary plus the actual repository files needed for the study.

At minimum:

```text
study.yaml
main experiment config(s)
beta-ablation config(s)
optional second-model validation/anchor config(s)
analysis.yaml
README.md
preflight/resource estimate
```

Also report:

1. which Study 04 configs/files were used as templates;
2. which parameters were intentionally kept identical;
3. which parameters are new sweep axes;
4. total expected cells/episodes/calls;
5. expected execution topology;
6. exact `mas_cc study submit ...` command;
7. exact `mas_cc study aggregate ...` command;
8. any repository limitation that prevents the requested clean organization.

---

# 16. Scientific summary

The intended Study 06 structure is:

\[
\boxed{
\text{Main: }(b,\theta)
}
\]

at fixed:

\[
\boxed{
q=1,\quad q_c=12,\quad \beta=4,\quad N=24.
}
\]

with:

\[
\boxed{
b=\{4,8,12,16,20,24\},
}
\]

\[
\boxed{
\theta=\{0.20,0.35,0.50,0.65,0.80\}.
}
\]

Then:

\[
\boxed{
\text{Targeted beta ablation: }
b=\{8,16,24\},\;
\beta=\{2,4,8\},\;
\theta=0.5.
}
\]

Optionally, after reasoning validation:

\[
\boxed{
\text{Second-model anchor: }
b=\{8,16,24\},\;
\theta=\{0.35,0.5,0.65\}.
}
\]

The central experimental question is no longer simply whether increasing intervention produces more change. It is whether **control efficiency, action information, susceptibility, and memory/epistemic modulation organize into richer regions of the experimentally controllable policy–actuation space**, and whether those regions are robust across tasks and at least one additional LLM family.
