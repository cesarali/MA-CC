# Handoff: Prepare Blackboard Population Study 01 (`blackboard_1`)

## Purpose

Prepare the first frozen population study for the new MuSR blackboard game.

The game/protocol itself should now be treated as **frozen**. This handoff is primarily about:

1. creating the study configuration family;
2. validating that all cells resolve to the intended frozen protocol;
3. preserving the existing control-information / susceptibility / efficiency analysis;
4. adding only the blackboard diagnostics needed to interpret the actuator;
5. leaving exact commands for an overnight run.

Do **not** redesign the task, prompt, message ontology, controller policy, or persistence mechanism while preparing this study unless a blocking implementation bug is discovered.

The configuration root requested for this study is:

```text
configs/runs/relational_reasoning/blackboard_game/blackboard_1/
```

Create that folder and place all study configs/readme/manifests needed to reproduce the experiment there.

---

# 1. Scientific question

Using one frozen MuSR Team Allocation task (`task_001`), compare:

```text
NO CONTROL
TRUTH CONTROL
FALSE CONTROL
```

across persistence and controller budget.

The central question remains the original paper question:

> How efficiently can a limited, imperfectly informed controller alter the collective state of a multi-agent system?

The blackboard is the new actuation substrate. The existing sensing/action variables and the existing information-theoretic/control-efficiency observables should be preserved.

This is **not** primarily a messaging study.

---

# 2. Frozen task and population

Use only:

```text
task_id = task_001
N = 24
rounds = 30
q = 1
q_c = 12
board_lifetime tau_B = 1 round
model = gwdg/openai-gpt-oss-120b
```

Use the already frozen `task_001` F9 N=24 private assignment produced for the pilot:

```text
24 ordinary agents
exactly one canonical F9 card per ordinary agent
exactly one latent value per ordinary agent
population union = all 9 canonical F9 cards
```

Do not regenerate the task, evidence cards, or assignment.

Use the exact hash-pinned assignment artifact already implemented for the pilot.

---

# 3. Frozen night/day controller protocol

All controlled cells must use the already implemented protocol:

```text
END OF DAY
    n_k

NIGHT
    sample q_c ordinary-agent votes -> Y_k
    compute existing P(U_k=1 | Y_k)
    sample binary U_k

DAWN
    U_k = 0 -> coordinator posts nothing
    U_k = 1 -> seed exactly b DIRECTIVE messages

DAY
    24 ordinary agents evolve autonomously
    each samples q live board messages
    agents may emit REQUEST / REPORT / NONE
    coordinator is silent during the day

END OF DAY
    measure n_{k+1}
```

The core causal ordering must remain:

\[
n_k
\rightarrow
Y_k
\rightarrow
P(U_k\mid Y_k)
\rightarrow
U_k
\rightarrow
B_k^{\rm dawn}
\rightarrow
\text{autonomous day}
\rightarrow
n_{k+1}.
\]

Preserve the existing stochastic controller parameters:

```text
beta = existing frozen value
theta = existing frozen value
q_c = 12
```

Use the same values as the validated pilot/current blackboard controller implementation. If the current frozen values are `beta=4`, `theta=0.5`, retain them; verify from code/config rather than silently changing them.

---

# 4. Public communication protocol is frozen

Ordinary agents:

```text
REQUEST
REPORT
NONE
```

Coordinating participant:

```text
DIRECTIVE
```

Every posted public message contains:

```text
author_public_id
current vote
message type
message text
optional shared_fact_id
optional reply_to
```

The coordinator should appear socially as another participant, not as an explicitly labeled "controller".

However, internally retain controller/coordinator provenance for analysis.

The coordinator's public vote is always visible when it posts but is **never counted** in:

```text
n
p_truth
population vote counts
consensus
sensor sample
population thermodynamic observables
```

The controlled population remains exactly N=24.

DIRECTIVEs never carry exact evidence.

Do not alter these semantics in this study.

---

# 5. Persistence values

Use exactly:

\[
\boxed{\rho \in \{0.74,\;0.85,\;1.00\}}
\]

These are the only persistence values for `blackboard_1`.

Interpretation:

```text
rho = 0.74   lower-persistence regime
rho = 0.85   intermediate/high-persistence regime of primary interest
rho = 1.00   no active forgetting / saturation reference
```

Do not add extra rho values.

Persistence acts only on `K_active`.

`K_hist` remains a permanent within-episode record.

---

# 6. Controller budgets

For the two controlled regimes use exactly:

\[
\boxed{b \in \{3,\;6,\;12,\;24\}}
\]

In the dawn-blackboard protocol:

```text
b = number of coordinator DIRECTIVE messages seeded at dawn when U=1
```

Therefore:

```text
U=0 -> 0 DIRECTIVEs
U=1 -> exactly b DIRECTIVEs
```

Do not reinterpret `b` as microscopic controlled positions in this study.

Do not add extra b values.

---

# 7. Three experimental arms

## 7.1 NO CONTROL

Coordinator completely absent.

For each:

```text
rho in {0.74, 0.85, 1.00}
```

run the same ordinary-agent blackboard dynamics but with:

```text
no sensing-driven controller intervention
no coordinator messages
no coordinator vote in social context
```

There is no reason to repeat the no-control baseline across `b`, because `b` is undefined/irrelevant when the controller is absent.

Total structural baseline cells:

```text
3
```

---

## 7.2 TRUTH CONTROL

Use the existing stochastic sensor/policy:

```text
n_k -> Y_k -> P(U|Y) -> U
```

Coordinator target/preference:

```text
gold semantic allocation for task_001
```

When `U=1`, the coordinator's public vote and DIRECTIVE semantics favor/coordinate around the gold target.

Sweep:

```text
rho = 0.74, 0.85, 1.00
b   = 3, 6, 12, 24
```

Total structural cells:

```text
12
```

---

## 7.3 FALSE CONTROL

Use the exact same sensing/policy machinery as TRUTH CONTROL.

The only intended controller-level difference is the target/preference.

Choose one fixed wrong **semantic allocation** before running any cell.

Do not define the false target from shuffled display letters.

Preferred deterministic rule:

```text
use the first canonical non-gold semantic allocation
in the frozen task_001 option ordering
```

unless the repository already has a frozen false-target convention for this task family. If such a convention exists, reuse it.

Record explicitly:

```text
gold target semantic id
false target semantic id
display mapping used in each prompt if applicable
```

Never choose the false target based on pilot outcomes or observed controllability.

Sweep:

```text
rho = 0.74, 0.85, 1.00
b   = 3, 6, 12, 24
```

Total structural cells:

```text
12
```

---

# 8. Total study size

Structural cells:

```text
NO CONTROL     = 3
TRUTH CONTROL  = 12
FALSE CONTROL  = 12
-------------------
TOTAL          = 27
```

Use:

```text
repetitions_per_cell = 10
```

unless an existing repository convention requires a different field name.

Therefore planned episodes:

```text
27 x 10 = 270 episodes
```

Each episode now runs for 30 rounds, so this is intentionally a substantially larger overnight study than the earlier 10-round / 5-repetition draft. The preflight must estimate provider-call volume and wall time using the actual runtime implementation before launch.

Do not silently reduce repetitions or rounds.

If runtime estimation suggests this cannot finish overnight, report the estimate before changing the design.

---

# 9. Matched experimental design

Make comparisons as paired/matched as the runtime permits.

For a given:

```text
rho
replicate
```

reuse the same frozen:

```text
task
initial private assignment
initial ordinary-agent ordering/state construction
experiment seed family
```

across:

```text
NO CONTROL
TRUTH CONTROL b=3
TRUTH CONTROL b=6
TRUTH CONTROL b=12
TRUTH CONTROL b=24
FALSE CONTROL b=3
FALSE CONTROL b=6
FALSE CONTROL b=12
FALSE CONTROL b=24
```

Use deterministic derived seeds for arm-specific stochasticity so runs are reproducible without accidentally forcing identical post-intervention trajectories.

Document the seed derivation.

The purpose is matched initial conditions, not artificial coupling of all LLM outputs.

---

# 10. Required config directory structure

Create:

```text
configs/runs/relational_reasoning/blackboard_game/blackboard_1/
```

Prefer a structure like:

```text
blackboard_1/
├── README.md
├── study_manifest.yaml
├── base.yaml
├── no_control/
│   ├── rho_074.yaml
│   ├── rho_085.yaml
│   └── rho_100.yaml
├── truth_control/
│   ├── rho_074_b03.yaml
│   ├── rho_074_b06.yaml
│   ├── rho_074_b12.yaml
│   ├── rho_074_b24.yaml
│   ├── rho_085_b03.yaml
│   ├── rho_085_b06.yaml
│   ├── rho_085_b12.yaml
│   ├── rho_085_b24.yaml
│   ├── rho_100_b03.yaml
│   ├── rho_100_b06.yaml
│   ├── rho_100_b12.yaml
│   └── rho_100_b24.yaml
└── false_control/
    ├── rho_074_b03.yaml
    ├── rho_074_b06.yaml
    ├── rho_074_b12.yaml
    ├── rho_074_b24.yaml
    ├── rho_085_b03.yaml
    ├── rho_085_b06.yaml
    ├── rho_085_b12.yaml
    ├── rho_085_b24.yaml
    ├── rho_100_b03.yaml
    ├── rho_100_b06.yaml
    ├── rho_100_b12.yaml
    └── rho_100_b24.yaml
```

If the repository has a cleaner native sweep/inheritance mechanism, use it rather than duplicating config contents, but preserve the same logical organization and ensure every one of the 27 cells can be identified unambiguously from frozen config artifacts.

`README.md` must explain the full design.

`study_manifest.yaml` should enumerate every structural cell and replicate count.

---

# 11. Results directory

Use a dedicated study output root, preferably:

```text
results/studies/musr_blackboard_population_01/
```

or the repository-native equivalent.

Do not write into the pilot study directory.

Every episode must record at least:

```text
arm
rho
b or null
replicate
task_id
gold_target
controller_target or null
seed information
```

---

# 12. Preserve the original information/control metrics

This is critical.

The blackboard study must remain compatible with the existing control-efficiency analysis.

Do not replace the original metrics with message statistics.

At round level preserve all records needed for the existing calculations, including:

```text
n_k
x_k = n_k / N
Y_k
sensor identities / sensed vote counts
P(U=1 | Y_k)
U_k
b
n_{k+1}
x_{k+1}
```

For no-control runs, store the appropriate null/disabled controller fields explicitly rather than fabricating controller actions.

Reuse the existing analysis implementation and existing metric names wherever possible.

---

# 13. Core metrics to compute as before

At minimum preserve/recompute the existing hierarchy.

## 13.1 Population outcomes

```text
p_truth(t)
p_target(t)
vote counts
dominant share
vote entropy
consensus / convergence diagnostics
```

For FALSE CONTROL keep both:

```text
p_truth
p_false_target
```

Do not conflate controller target with truth.

---

## 13.2 Sensing information

Use the existing sensing-information definition already used by the project.

In the current theoretical notation this is:

\[
I_{\rm sens} = I(n_k;Y_k)
\]

or its existing implementation-equivalent representation.

Do not silently redefine sensing MI as a different quantity.

Preserve any existing null/permutation/bootstrap procedures.

---

## 13.3 Controller-to-population transfer information

Preserve:

\[
\boxed{
T_\pi
=
I(U_k;n_{k+1}\mid n_k)
}
\]

using the existing discrete CMI estimator / bias controls / bootstrap or permutation machinery.

Where the current code supports conditioning on additional fixed cell parameters, compute it consistently within:

```text
arm
rho
b
```

and in pooled/state-resolved forms as before.

---

## 13.4 Susceptibility / response

Preserve the existing state-local controller-response definition, conceptually:

\[
\chi(x,b)
=
E[x_{k+1}\mid x_k=x,U=1,b]
-
E[x_{k+1}\mid x_k=x,U=0,b].
\]

Use the exact repository implementation/normalization already established for previous studies.

Do not introduce a new susceptibility definition.

Compute it against the **available/observed x states**, consistent with the previous population analyses.

---

## 13.5 Existing information-response efficiency

Preserve the existing:

```text
eta_IR
```

implementation and naming.

Do not redesign it for the blackboard study.

---

## 13.6 Existing thermodynamic/current quantities

Preserve the existing analysis hooks for:

```text
J_c
thermodynamic efficiency / eta_th
signed thermodynamic efficiency if currently used
```

using the same definitions and conventions already in the repository.

Important:

- do not silently change formulas because the actuator is now a blackboard actuator;
- do not delete the metrics;
- clearly label them according to the terminology already used by the code/theory;
- if a current thermodynamic estimator requires assumptions not automatically guaranteed by the new actuator, preserve the raw quantities and mark that limitation in the report rather than inventing a replacement formula.

The objective of this study is explicitly to see whether the previous efficiency structure survives under the new actuation substrate.

---

# 14. State-resolved analysis remains a priority

Do not analyze only global episode averages.

For each persistence slice:

```text
rho = 0.74
rho = 0.85
rho = 1.00
```

prepare the same state-resolved analysis over the actually available population states:

\[
x = n/N.
\]

The main controlled maps should support:

```text
x × b
```

for:

```text
chi
T_pi
eta_IR
J_c
eta_th / signed eta_th where available
```

separately for:

```text
TRUTH CONTROL
FALSE CONTROL
```

Use the observed discrete `x` support directly where statistics permit.

If aggregation/binning is required for sample support, reuse the existing previous-study binning/aggregation rules. Do not invent a new arbitrary x grid.

Also provide rho-aggregated views if the existing analysis already supports them, but keep the three rho slices separately available.

---

# 15. Coarse phase-space summaries

In addition to `x × b | rho`, provide coarse summaries across:

```text
rho × b
```

for the two controlled arms.

Useful summaries include the existing study-level quantities such as:

```text
late/final p_truth
late/final p_target
integrated/mean susceptibility
T_pi
eta_IR
J_c
eta_th
```

Use the same aggregation conventions as the previous studies.

For NO CONTROL, provide the three rho baselines separately; do not manufacture a fake b-axis by rerunning identical conditions.

For plotting only, it is acceptable to repeat the no-control rho baseline visually across b as a reference surface, but clearly mark it as a replicated visualization of the same baseline, not separate simulations.

---

# 16. Blackboard diagnostics are secondary mechanism observables

Keep the new blackboard observables, but treat them as explanatory diagnostics rather than replacements for the core metrics.

At minimum record:

```text
DIRECTIVEs posted
DIRECTIVEs read
unique agents exposed to a DIRECTIVE
eligible DIRECTIVE fraction D_t/M_t
REQUEST count
REPORT count
semantic-only reads
exact evidence acquisitions
refresh events
DIRECTIVE -> REPORT replies
DIRECTIVE-attributed acquisitions
DIRECTIVE-attributed refreshes
mean |K_active|
mean |K_hist|
active latent coverage
historical latent coverage
```

These will help explain why a given `(rho,b)` cell has high or low response/efficiency.

---

# 17. Important control-variable invariants

Before preparing the final configs, verify provider-free that:

```text
n -> Y -> P(U|Y) -> U
```

is unchanged from the frozen controller implementation.

For controlled arms:

```text
U=0 -> zero dawn DIRECTIVEs
U=1 -> exactly b dawn DIRECTIVEs
```

For NO CONTROL:

```text
coordinator absent
no coordinator vote
no controller messages
```

The coordinator must never enter:

```text
N
n
Y sensor pool
p_truth denominator
consensus calculation
```

Add/retain tests for these invariants.

---

# 18. False-control sanity checks

Before provider calls:

1. resolve the exact gold semantic allocation for `task_001`;
2. resolve the exact frozen false semantic allocation;
3. assert they differ;
4. save both to the manifest;
5. verify the controller's public vote/semantic directive target matches the configured arm;
6. verify display-letter shuffling, if any, never changes the semantic target;
7. verify `p_truth` and `p_controller_target` are computed independently.

---

# 19. Preflight / dry run

Before an overnight run:

- load every config;
- validate all 27 structural cells;
- verify 10 repetitions each;
- check there are no accidental duplicate cells;
- check no missing cells;
- estimate total provider calls;
- estimate wall-clock time under the current concurrency/RPM configuration;
- print the planned episode count;
- print resolved task/assignment hashes;
- print resolved truth and false semantic targets.

Do not launch automatically if the user/compute-agent workflow expects preparation only.

Leave exact commands for:

```text
full study run
resume interrupted study
analysis only
report generation
```

using the repository-native CLI.

---

# 20. Analysis outputs

Prepare a reproducible analysis entry point that generates at least:

```text
analysis/blackboard_1_report.md

analysis/tables/cell_summary.csv
analysis/tables/rho_b_summary.csv
analysis/tables/state_resolved_x_b.csv
analysis/tables/sensing_information.csv
analysis/tables/transfer_information.csv
analysis/tables/susceptibility.csv
analysis/tables/efficiencies.csv
analysis/tables/blackboard_diagnostics.csv
```

and figures for:

```text
behavioral rho × b summaries
truth-control x × b maps by rho
false-control x × b maps by rho
chi
T_pi
eta_IR
J_c
eta_th / signed eta_th where implemented
active vs historical information
realized control exposure
```

Reuse the visual conventions of the previous persistence/control-efficiency reports as much as possible.

---

# 21. Report framing

The report should explicitly separate:

## Outcome

```text
Did truth control improve truth?
Did false control move the population toward the false target?
How does persistence alter the outcome?
```

## Control mechanism

```text
Where in x × b space is susceptibility largest?
Where is T_pi largest?
Where does control information fail to become population response?
```

## Efficiency

```text
Where are eta_IR / thermodynamic efficiency highest?
Are the efficiency landscapes non-monotone in b?
Do their maxima differ from the maxima of raw behavioral control?
```

## Blackboard mechanism

```text
How much of nominal b became realized DIRECTIVE exposure?
Did exposure produce replies/evidence propagation?
How did rho alter active-memory retention and refresh?
```

The main narrative remains **control efficiency**, not communication taxonomy.

---

# 22. Hard rules

Do not:

```text
change task_001
regenerate evidence
change the F9 N24 assignment
change N
change q
change q_c
change rounds
add rho values
add b values
change message ontology
change night/day timing
change P(U|Y)
change beta/theta
introduce new controller modes
add more tasks
silently change existing MI/CMI/efficiency definitions
```

If a required current value cannot be verified from the implementation, report it and stop rather than guessing.

---

# 23. Completion summary

When preparation is complete, print:

```text
config root
results root
task id + task hash
assignment path + hash

gold semantic target
false semantic target

rho values
b values
repetitions

no-control cells
truth-control cells
false-control cells
total structural cells
total planned episodes
estimated provider calls
estimated wall time

controller beta
controller theta
q_c
q
N
rounds
tau_B

tests/preflight passed/failed

full-run command
resume command
analysis command
report command
```

The final study design should resolve to:

```text
rho = {0.74, 0.85, 1.00}

NO CONTROL:
    3 cells

TRUTH CONTROL:
    rho × b
    b = {3, 6, 12, 24}
    12 cells

FALSE CONTROL:
    rho × b
    b = {3, 6, 12, 24}
    12 cells

10 repetitions per cell

TOTAL = 27 structural cells
TOTAL = 270 episodes
```

This is `blackboard_1`.
