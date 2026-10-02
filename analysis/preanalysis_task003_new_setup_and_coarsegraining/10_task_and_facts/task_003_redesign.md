# Task 003: controller pool correction and experiment redesign

1 October 2026. Based on `task_003_analysis.md`, its CSV and proof inventory,
current generator/controller code, and independent exact completion calculations.
No model calls or experiment launches were performed.

## What is broken, and what already exists

`selective_design.py` chooses a pool for the false target: exclude the particular
greedy decisive set, require positive individual target lift, then retain facts
consistent with a selected false-winning witness world. The controller's default
`controller_report_pool_mode: frozen` previously returned that pool regardless
of the resolved target. Validation explicitly allowed the truth target to reuse
it. The old test `test_selective_false_pool_can_be_reused_for_exact_truth_counterpart`
encoded this behavior. True statements are not necessarily evidence for a target.

An existing opt-in correction, `target_aligned_v1`, already supplies 23 different
facts to truth control in the six-arm v5 target-aligned configuration directory.
It is distinct from the older studies discussed in the analysis. It requires all
six decisive facts, excludes overlap with the false pool, and boosts decisive
facts in deterministic selection. Those are experimental choices, not general
requirements for correct target alignment.

The narrow code fix rejects a frozen pool whenever its declared target differs
from the resolved target. Existing false-target runs and genuinely truth-designed
frozen pools still work; explicitly configured v1 truth pools still work. Older
truth-control configurations using the default now fail early with migration
guidance. Their historical outputs remain readable, but rerunning those configs
requires an explicit design decision. Do not silently reinterpret or merge old
results with corrected results.

## Independent pool audit

Run `python results/game_analysis/audit_target_pools.py` from the project
environment. The checked-in `task_003_pool_comparison.json` records IDs, sorted-ID
hashes, private overlap, exact posteriors and four-fact proof coverage. These
hashes describe sorted membership, not the ordering hashes of existing studies.

| Pool | Facts | Shared with private evidence | Fails private eligibility | P(A0), P(A1), P(A2) using entire pool | Four-fact proofs |
|---|---:|---:|---:|---|---:|
| Historical false pool, reused in old truth arm | 24 | 8 | 4 | .308, 0, .692 | 0 |
| Existing explicit truth v1 | 23 | 12 | 4 | 1, 0, 0 | 10 |
| All positive-lift A0 facts | 35 | 15 | 6 | 1, 0, 0 | 22 |
| All positive-lift A2 facts | 27 | 11 | 4 | .308, 0, .692 | 0 |
| Private-eligible positive-lift A0 facts | 29 | 15 | 0 | 1, 0, 0 | 0 |
| Private-eligible positive-lift A2 facts | 23 | 11 | 0 | .321, 0, .679 | 0 |
| Positive A0 lift and P(A0) > P(A2) | 23 | 10 | 4 | 1, 0, 0 | 8 |
| Positive A2 lift and P(A2) > P(A0) | 15 | 7 | 2 | .064, 0, .936 | 0 |

The positive-lift pools overlap by 15 facts. Some facts improve both A0 and A2
by weakening A1. Therefore separate target-specific construction does not mean
disjoint pools, and switching targets is not simply flipping a scalar sign.
If the intended intervention is evidence favoring A2 *over A0*, use the
contrastive criterion explicitly. Its smaller false pool is collectively more
persuasive than the full 27-fact pool: adding true facts can reduce P(false).
Individual positive lift does not guarantee positive conditional lift after
other reports or private evidence.

Whole-pool P(A0) < 1 certifies that the historical pool cannot contain a truth
proof of **any** size: a false-winning completion survives every fact. Merely
counting no proofs up to size six would not establish that conclusion. Conversely,
the weak A0 pool proves truth despite containing no four-fact proof. Private
eligibility is not a proof-prevention criterion.

The analysis's 7,694 minimal proofs are exhaustive only through size six; four
disjoint proofs are a packing lower bound. Disjoint fact IDs do not imply
statistically independent evidence. The archived CSV labels also do not preserve
which fact belongs to each particular agent; retain the original assignment
artifact for actual experiments.

## Recommended sequence

1. **Correct the software contract first.** Use the existing audited v1 pools for
   a minimal corrected replication. Freeze new run identifiers and explicitly
   document its 23/24 pool size, 12/8 private overlap and 10/0 shortest-proof
   asymmetries. This answers whether correction changes the prior result; it
   does not isolate evidence-independent directional steerability.
2. **Separate content from activation.** First compare no-control, truth-control
   and false-control under a fixed intervention schedule with the same starting
   population state, private assignment, board rules, model, memory and horizon.
   Use paired initialization artifacts, not just matching integer seeds. Start
   with report-only, canonical wording and fixed posting budgets, e.g. b=3,6,12.
   `advocacy_schedule: always` removes the soft activation gate, but adaptive
   selection, cooldowns and exhaustion can still reduce posting counts: verify
   actual reports and deliveries. Predefine a common feasible replay schedule
   or explicitly analyze the resulting policy as a variable-dose treatment.
3. **Use a common candidate rule for the substantive redesign.** Build pools
   independently from all 49 true facts with positive lift toward each resolved
   target; permit overlap and remove the greedy decisive exclusion in both arms.
   Use the same individual strength gate in both arms. Freeze membership,
   target-specific scores and ordering in versioned artifacts before running.
   The present v1 runtime cannot express this general rule: it forces disjoint
   truth/false pools and retains false-target scores. Implement and test a new
   schema/mode before using these candidate lists as a run configuration.
4. **Add one planned sensitivity at a time.** Contrast positive-lift versus
   contrastive pools, then unrestricted versus private-eligible pools. Keep
   budgets below the smaller pool size. Equal pool size alone does not match
   information: audit lift distributions, private overlap, joint posterior,
   proof access, redundancy and report-sequence conditional effects. If selecting
   matched subsets, freeze their matching objective and re-audit those exact
   subsets; do not infer their properties from the full candidate pools.
5. **Evaluate feedback second.** Repeat the selected fixed-schedule design under
   the soft gate. Measure total closed-loop policy effect separately from the
   fixed-dose effect. The reported 32%/54% firing rates motivate this separation;
   they were not independently re-estimated by this audit. Do not condition only
   on realized firing rounds and call that a causal effect.

The primary endpoints should be final truth share and target share relative to
the paired no-control run, with paired uncertainty over independent episode
replicates. Report consensus accuracy/time, actual activation and delivered
reports, unique evidence exposure/retention, and proof completion in each agent's
active knowledge as mechanisms. Treat unresolved consensus times as censored.
Choose a primary endpoint and horizon before comparing conditions; size the
replicate count using a small corrected pilot and its paired variance.

A truthful false-target controller cannot have a logically sufficient proof of
its false target while the actual world remains possible. Do not try to match
truth/false proof availability to certainty. Study truth-proof access as a
separate factor, or state that the treatment compares correction against
selective persuasion under the world's inherent asymmetry.

For generalization, use multiple worlds, counterbalance which allocation is
correct and which incorrect target is chosen, and stratify by score margin and
minimum proof size. The 12 additional worlds show structural variation, not a
representative estimate of the family. Keep task/world as an analysis unit;
more repetitions of task_003 do not substitute for more worlds.

## Validation and limits

Eleven compatibility regression tests and 17 existing controller-focused tests pass.
The compatibility regression tests are self-contained and exercise truth aliases,
both fact-reporting modes, selection, declared false and truth pools, the existing
explicit correction, missing pools and unsupported targets. The exact audit
reconstructs the world and verifies every CSV singleton lift against the symbolic
engine. Frozen calibration task directories are absent in this local checkout;
the archive-dependent end-to-end tests and a real corrected smoke run therefore
remain execution-site checks. No original analysis, frozen task, study config,
private assignment, or recorded result is changed by this work.
