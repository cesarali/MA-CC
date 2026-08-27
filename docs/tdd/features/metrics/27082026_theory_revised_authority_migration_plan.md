# Migration plan: make `theory_revised.py` the sole theoretical authority

**Date:** 27 August 2026  
**Status:** proposed migration plan; no implementation changes in this document  
**Canonical theory module:** `src/mas_cc/games/relational_reasoning/imitation_round_feedback/theory_revised.py`  
**Scope:** relational imitation-round-feedback analysis, current analysis, standardized study aggregation, run/study configs, reports, manifests, and tests

---

## 1. Outcome

After this migration, every production quantity labelled as theoretical for the
relational round-feedback game must be evaluated by `theory_revised.py`.

The canonical chain will be:

```text
retained round + micro-slot observations
    -> empirical single-affinity calibration and estimators
    -> TheoryParameters reconstructed for one physical cell
    -> theory_revised.single_affinity_reference(...)
    -> empirical/theory comparison table
    -> plots, reports, manifest, and package
```

No production analysis path may obtain a generic `theory_*`, `*_theory`, or
“exact theory” value from the current legacy `theory.py` module.

The existing matched q-voter model may survive only as a deliberately separate
classical null, with an explicit name and output namespace. It must never be
substituted for, merged into, or presented as the revised single-affinity
theory.

This migration changes analysis and reporting only. It must not change the LLM
runtime, scientific cell identities, episode seeds, scheduler topology, or
provider workload.

---

## 2. Why a migration is required

The working tree already contains the correct revised empirical/theory path:

```text
src/mas_cc/analysis/single_affinity.py
    -> theory_revised.py

src/mas_cc/studies/aggregation.py
    -> single_affinity.theory_comparison(...)
    -> analysis/tables/single_affinity_theory_comparison.parquet
```

However, two older production paths still import the legacy theory module:

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/analysis.py
    -> .theory

src/mas_cc/games/relational_reasoning/imitation_round_feedback/current.py
    -> .theory
```

This creates two incompatible meanings of “theory”:

1. the revised isolated single-affinity controlled layer; and
2. the matched finite-`N` controlled q-voter reference.

Both are mathematically meaningful, but they are not interchangeable. The
current generic filenames and field names make it possible for a downstream
reader to compare an updated empirical estimator with the wrong theoretical
reference.

The migration therefore needs to repair authority, naming, configuration, and
provenance together. Merely changing one import is insufficient.

---

## 3. Non-negotiable invariants

### 3.1 One canonical theoretical implementation

`theory_revised.py` is the only production implementation allowed to emit the
single-affinity theoretical values:

```text
chi
T_pi
eta_IR
J_c
I_sens
eta_th
finite-horizon current mean / variance
finite-horizon thermodynamic decomposition
```

Callers may adapt records and serialize results, but they must not reproduce
these formulas.

### 3.2 Theory stays deterministic and provider-free

Do not put trajectory reading, empirical estimation, bootstrapping, provider
calls, plotting, or filesystem writes into `theory_revised.py`.

### 3.3 Empirical estimators remain empirical

`src/mas_cc/analysis/single_affinity.py` remains responsible for estimating
`h`, `gamma`, `chi`, `T_pi`, `J_c`, `I_sens`, `eta_IR`, and `eta_th` from
retained observations. It may call deterministic utilities from
`theory_revised.py`, but it must not fill missing empirical quantities with
theory values.

### 3.4 Physical-cell boundaries are mandatory

Theory parameters and empirical/theory comparisons are resolved per physical
scientific cell. Do not average parameter tuples or construct one theory for a
group spanning different `(N, q_c, b, beta, theta, h, gamma)` values.

### 3.5 No replacement CMI estimator

The established direct-counting MI/CMI, bootstrap, null, and support machinery
remains authoritative. This migration changes the theory source, not the
information estimator.

### 3.6 No provider rerun when retained data suffice

Reaggregation must use canonical Parquet observations. New LLM calls are
justified only if required round or micro-slot fields were never retained.

### 3.7 Old outputs are never silently relabelled

An artifact produced with `theory.py` remains a legacy matched-q-voter artifact.
It must not acquire revised-theory provenance during re-packaging without being
recomputed.

---

## 4. Target architecture

### 4.1 Canonical theory API

All production theory evaluation should enter through the public revised API:

```text
theory_revised.TheoryParameters
theory_revised.theory_parameters_from_record
theory_revised.single_affinity_reference
theory_revised.finite_horizon_thermodynamics
theory_revised.finite_horizon_current_moments
theory_revised.thermodynamic_efficiency
```

If callers currently need a convenience operation absent from this list, add a
pure deterministic helper to `theory_revised.py` and test it there. Do not
restate its mathematics in `analysis.py`, `current.py`, or aggregation.

### 4.2 Empirical/theory facade

`src/mas_cc/analysis/single_affinity.py` should remain the sole facade that
combines empirical observations with the revised reference:

```text
single_affinity_analysis(...)
    -> empirical estimates + intervals + support + provenance

theory_comparison(...)
    -> empirical values beside revised exact values
```

The facade owns comparison semantics such as empirical-occupancy weighting,
calibrated `h`/`gamma`, unavailable reasons, and residuals. Other modules call
the facade rather than constructing a second comparison.

### 4.3 Standardized output contract

The authoritative study-level comparison remains:

```text
analysis/tables/single_affinity_theory_comparison.parquet
```

Each available row should contain at least:

```text
study_id
source_run_id
cell_id
quantity
units
empirical
single_affinity_theory
residual
reference = single_affinity_revised
available
reason
theory_semantics_version
theory_module
theory_api_version
analysis_hash
```

`theory_module` should record the fully qualified canonical module name. The
analysis hash must change when the theory API/semantics version changes.

### 4.4 Classical null isolation

If the old matched q-voter is retained, move or rename its public identity to
something explicit, for example:

```text
matched_qvoter.py
classical_qvoter_reference(...)
matched_qvoter_null.parquet
reference = matched_qvoter_classical_null
```

Forbidden names for that output include:

```text
theory_comparison
theory_state_curves
current_mean_theory
exact_theory
```

The null must be opt-in and must not be emitted merely because relational
configured analysis is enabled.

---

## 5. Configuration contract

### 5.1 Do not accept arbitrary Python module paths in YAML

Configs should select a stable scientific reference name, not import code:

```yaml
analysis:
  options:
    theoretical_reference: single_affinity_revised
```

The loader resolves that name to the repository-owned canonical implementation.
Do not add a free-form `module:` or `callable:` setting; that would make results
dependent on unvalidated code paths and weaken provenance.

### 5.2 Default and allowed values

For relational round feedback:

```text
single_affinity_revised   canonical default
none                      explicitly disable theory comparison
matched_qvoter_null       optional, separately named classical diagnostic
```

If strict interpretation of the project policy disallows the old model
entirely, omit `matched_qvoter_null` from the public config schema and keep it
only in archived tests/notebooks.

### 5.3 Study recipes

Study-level `analysis.yaml` should continue to request scientific quantities,
not implementation modules. Naming any member of the coupled family may still
activate the complete family, but the recipes should list it explicitly for
auditability:

```yaml
estimators:
  - round_target_susceptibility
  - round_target_sensing_mi
  - round_target_actuation_cmi
  - effective_affinity
  - kinetic_compliance

derived:
  - round_target_susceptibility
  - eta_ir
  - target_sensing_information_nats
  - controlled_current
  - affinity_weighted_current_nats
  - thermodynamic_control_expenditure_nats
  - eta_th
```

`experiment.metadata.primary_analysis_family` remains descriptive metadata and
must never trigger computation.

---

## 6. File-by-file migration

### 6.1 `theory_revised.py`

1. Preserve the existing deterministic theory and its current public API.
2. Add an explicit theory API/semantics version constant if one is not already
   present.
3. Confirm that `finite_horizon_current_moments` covers the mean/variance needs
   currently served by legacy `current.py`.
4. Add only minimal deterministic adapters required for parity.
5. Keep the module free of pandas, trajectory readers, bootstrap code, and
   report generation.

### 6.2 `src/mas_cc/analysis/single_affinity.py`

1. Keep all theoretical imports pointed at `theory_revised.py`.
2. Centralize the mapping from empirical `h`/`gamma` plus recorded protocol
   parameters to `TheoryParameters`.
3. Extend comparison provenance with canonical module/API identifiers.
4. Expose any revised finite-horizon current comparison needed by the current
   report so `current.py` does not import theory directly.
5. Continue returning an explicit unavailable result for unidentifiable
   affinity, mixed protocols, or missing controlled rounds.

### 6.3 Relational `analysis.py`

1. Remove all production imports from `.theory`.
2. Delete or retire the embedded generic `theory_comparison`, theory-curve,
   and interpretation implementation that builds the old q-voter reference.
3. Call `mas_cc.analysis.single_affinity.theory_comparison` for the canonical
   comparison.
4. Rename per-run artifacts to identify the reference unambiguously, or write
   the same canonical comparison schema used by study aggregation.
5. Replace `theory_comparison_enabled` with the validated reference selection
   described above, retaining a temporary compatibility shim only if needed.
6. Ensure a run with insufficient data reports `available=false` and a reason;
   it must not fall back to legacy theory.

### 6.4 Relational `current.py`

1. Separate empirical terminal-current calculation from theoretical current
   calculation.
2. Keep `episode_current` and empirical `cell_current` unchanged as behavioral
   observables.
3. Replace legacy theory imports with the revised finite-horizon API, reached
   through the canonical facade where practical.
4. Rename theory fields to make their reference explicit, for example:

   ```text
   current_mean_single_affinity_theory
   current_variance_single_affinity_theory
   ```

5. Do not compare terminal behavioral current with response-based controlled
   `J_c` as if they were the same quantity. Reports must explain the coordinate
   and kernel used by each current.
6. Allow empirical-only callers to bypass all theory computation. Standardized
   aggregation currently needs empirical `cell_current` and should not build a
   discarded theory object.

### 6.5 `src/mas_cc/experiments/configured_analysis.py`

1. Add validation for the stable theoretical-reference option.
2. Pass the resolved reference choice to the relational analyzer.
3. Default new relational configs to `single_affinity_revised`.
4. Decide explicitly how configs lacking the new field are handled:
   - during one compatibility release, resolve them to
     `single_affinity_revised` and record a warning; then
   - require the field or rely on the documented canonical default.
5. Reject unknown references during preflight before any provider call.

### 6.6 `src/mas_cc/studies/aggregation.py`

1. Keep `single_affinity.theory_comparison` as the only canonical theory call.
2. Change `reference` from the ambiguous `single_affinity` to the versioned
   `single_affinity_revised` identifier.
3. Include the theory module/API version in the theory-comparison analysis hash
   and manifest.
4. Call empirical current analysis with theory disabled when producing only
   `episode_current` or `cell_current`.
5. Continue computing theory per physical cell and publishing unavailable
   reasons rather than mixed-cell averages.
6. Preserve the coupled bootstrap and the single-affinity provenance stamp.

### 6.7 Legacy `theory.py`

Choose one of these end states before implementation begins:

**Preferred:** rename it to `matched_qvoter.py`, update explicitly classical
tests/notebooks, and remove `theory.py` after one compatibility window.

**Strict:** archive/remove it from the runtime package immediately and retain
only historical documentation or fixtures.

In either case:

- no production analyzer imports it;
- no generic theory-named output is produced by it;
- it is not a fallback when revised calibration is unavailable;
- legacy artifacts remain identifiable by their old schema/provenance.

### 6.8 Package exports and documentation

1. Export revised theory objects under explicit names if package-level exports
   are needed.
2. Remove documentation that says generic exact quantities “live in theory.”
3. Update analysis reports to distinguish:
   - revised single-affinity theory;
   - empirical single-affinity estimates;
   - behavioral terminal current;
   - optional matched-q-voter classical null.
4. Update config READMEs and command examples after the schema is final.

---

## 7. Config migration matrix

| Config family | Current behavior | Required migration |
|---|---|---|
| Study 01 | analysis disabled | Keep disabled unless an explicit offline recipe is added; no theory output |
| Study 02 | analysis disabled | Same as Study 01 |
| Study 03 | configured per-run analysis; legacy theory emitted automatically | Add corrected empirical estimators and select revised theory explicitly |
| Study 04 | configured per-run analysis; legacy theory emitted automatically | Same as Study 03; update README language |
| Study 05 | mostly analysis disabled; state-matching/affinity role | Preserve retained data and empirical affinity behavior; no implicit theory |
| Study 06 | standardized aggregation; `eta_ir` implicitly activates revised family | List full estimator/derived family explicitly; ensure empirical current bypasses legacy theory |
| Study 07 | same pattern as Study 06 | Same migration as Study 06 |
| Study 08 | explicit updated family already present | Add explicit revised-reference provenance/config once schema exists |
| Root smoke/prompt configs | mostly analysis disabled | Add a tiny revised-theory analysis smoke; keep prompt-only configs theory-free |

The benchmark-only `study06_second_model_validation.yaml` is not an experiment
run config and is outside this theory-source migration.

---

## 8. Artifact and backward-compatibility policy

### 8.1 Version identifiers

Introduce or persist all of:

```text
theory_semantics_version = single_affinity_v1
theory_reference = single_affinity_revised
theory_module = mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised
theory_api_version = <explicit stable version>
```

Legacy matched-q-voter products need a distinct stamp, such as:

```text
theory_reference = matched_qvoter_legacy
```

### 8.2 Existing results

For an existing study result:

1. discover and validate retained canonical rounds and micro-slots;
2. recompute empirical and theoretical tables offline;
3. write a new analysis hash and revised-theory provenance;
4. replace only regenerated analysis products, never source observations;
5. state clearly when missing micro-slot fields prevent finite `h`, `gamma`, or
   `eta_th` estimation.

Do not copy a legacy `theory_comparison.csv` into the standardized comparison
table.

### 8.3 Transitional readers

Readers may recognize old filenames long enough to provide a useful migration
message, but they must not ingest legacy theory columns into revised-theory
fields. Compatibility is allowed at the file-reading boundary, not at the
scientific-semantics boundary.

### 8.4 Output cleanup

Aggregation should remove obsolete analysis products before writing the one
authoritative handoff, while preserving canonical scientific observations.
Any deletion behavior must remain scoped to the analysis output directory and
covered by tests.

---

## 9. Test-driven migration phases

### Phase A — freeze the authority boundary

Add failing tests first:

1. a production-import test that rejects imports of relational `.theory` from
   runtime analysis, current analysis, aggregation, and configured analysis;
2. a provenance test requiring the revised module/reference identifier on
   every canonical theory row;
3. a preflight test rejecting unknown theoretical-reference values;
4. a test proving unidentifiable revised theory does not fall back to the
   q-voter model.

Exit gate: the tests express the intended boundary before code is moved.

### Phase B — migrate current analysis

1. Add parity tests for revised `finite_horizon_current_moments` on the cases
   currently covered by legacy current tests.
2. Split empirical and theoretical current paths.
3. Make standardized aggregation request empirical-only current summaries.
4. Route explicit theoretical current requests to `theory_revised.py`.
5. Rename output fields and report headings.

Exit gate: no production import of `.theory` remains in `current.py`, and
empirical `cell_current` is unchanged on fixed fixtures.

### Phase C — migrate the per-run relational analyzer

1. Replace its embedded q-voter comparison with the canonical facade.
2. Write revised single-affinity comparison rows with the standardized schema.
3. Preserve the established empirical MI/CMI/null/support outputs exactly.
4. Move any retained q-voter calculation behind an explicit classical-null
   request and separate filename.
5. Update Comet mappings so generic `*_theory` keys cannot refer to the old
   model.

Exit gate: enabling relational configured analysis produces revised theory or
an explicit unavailable reason, never an implicit q-voter result.

### Phase D — migrate configs and recipes

1. Update Studies 03 and 04 to request:

   ```text
   round_target_susceptibility
   round_target_sensing_mi
   ```

2. Add the revised-reference selection to analysis-enabled relational configs.
3. Expand Studies 06 and 07 `derived:` sections to list the complete coupled
   family explicitly.
4. Confirm Study 08 remains semantically unchanged except for explicit
   reference/version metadata.
5. Update READMEs and preflight documentation.

Exit gate: every experiment config passes provider-free preflight and every
analysis-enabled relational config resolves to the revised reference.

### Phase E — retire the ambiguous legacy surface

1. Rename/archive legacy q-voter code according to the decision in section
   6.7.
2. Remove compatibility aliases that emit generic theory field names.
3. Add a repository-wide static check for forbidden production imports and
   ambiguous output names.
4. Retain focused classical-null tests only under explicit classical naming.

Exit gate: `rg` finds no production import of the old `.theory` module and no
new generic theory artifact can be produced from it.

### Phase F — reaggregation smoke and handoff

1. Run a tiny mock-provider relational config retaining rounds and micro-slots.
2. Aggregate it through the standardized workflow.
3. Verify revised empirical/theory values and provenance in Parquet.
4. Reaggregate from canonical Parquet after hiding/removing access to the
   source run tree in the fixture.
5. Verify byte-stable scientific values and unchanged analysis identity.
6. Package the handoff and inspect reports/plots for unambiguous labels.

Exit gate: no provider calls are needed for reaggregation and the packaged
theory source is reconstructable from machine-readable provenance.

---

## 10. Required tests

### 10.1 Revised theory invariants

Retain and extend coverage for:

- stochastic kernels summing to one;
- exact susceptibility/kernel-response equality;
- statewise information-response bounds;
- finite-horizon thermodynamic path identity;
- finite-horizon current mean and non-negative variance;
- affinity/compliance calibration;
- deterministic output for fixed parameters.

### 10.2 Empirical/theory consistency

Using data sampled from the revised kernel, verify convergence of:

```text
chi
T_pi
eta_IR
J_c
I_sens
eta_th
finite-horizon current mean / variance
```

The test simulator must draw from the revised theory's own kernel rather than
copying the formulas.

### 10.3 Estimator regression

Verify that migration does not change the established empirical:

```text
round sensing MI
target actuation CMI
null summaries
support diagnostics
episode bootstrap behavior
episode_current
cell_current
effective_affinity
kinetic_compliance
```

Changes to corrected `eta_ir` semantics remain identified by the existing
single-affinity provenance version.

### 10.4 Configuration tests

- every analysis-enabled relational config accepts the revised reference;
- an unknown reference fails preflight;
- `none` produces no theory table but leaves empirical estimates intact;
- an unavailable revised comparison records a reason;
- no-control configs do not manufacture `h`, `gamma`, or `eta_th`;
- benchmark YAML is not misclassified as an experiment config.

### 10.5 Import and output-boundary tests

- no production module imports relational `.theory`;
- every `single_affinity_theory` value originates from the canonical facade;
- old q-voter filenames cannot be mistaken for revised comparison tables;
- theory provenance participates in `analysis_hash`;
- reaggregation cannot reuse a legacy theory table as revised output.

### 10.6 Artifact-profile tests

For both `full` and `results_only`:

- canonical rounds contain the protocol/state fields needed by revised theory;
- canonical micro-slots contain controlled transition fields needed by
  `h`/`gamma`;
- source-tree aggregation and retained-Parquet reaggregation agree;
- missing historical fields yield a clear unavailable reason rather than a
  fallback reference.

---

## 11. Verification commands

Use the repository's existing local Python 3.11+ environment outside Potsdam.
On Potsdam, use the dedicated `MA-CC` Conda environment required by repository
instructions.

Targeted local regression:

```bash
.venv/bin/python -m pytest -q \
  tests/mas_cc/test_single_affinity_consistency.py \
  tests/mas_cc/test_relational_round_feedback_analysis.py \
  tests/mas_cc/test_relational_current_analysis.py \
  tests/mas_cc/test_relational_matched_theory.py \
  tests/mas_cc/test_configured_analysis.py \
  tests/mas_cc/test_studies.py \
  tests/mas_cc/test_results_only_resume.py \
  tests/mas_cc/test_import_safety.py
```

Static authority check after migration:

```bash
rg -n "from \.theory|imitation_round_feedback\.theory import" src/mas_cc
```

Expected result: no production matches.

Provider-free config preflight:

```bash
mas-cc experiment preflight \
  --config <relational-config.yaml> \
  --output-dir <temporary-inspection-directory>
```

Run it for every experiment config changed by the migration. Use study
submission only after all constituent configs pass and only when a real run is
explicitly authorized.

---

## 12. Risks and controls

| Risk | Control |
|---|---|
| Legacy and revised columns share a familiar name | Versioned reference/module fields and renamed classical-null outputs |
| Current mean changes because the kernels describe different systems | Treat it as an intentional reference migration; retain fixed empirical current regression tests |
| Missing microscopic data makes revised calibration unavailable | Report `available=false`; never fall back or smooth silently |
| YAML can select arbitrary code | Use a closed reference-name registry, not module paths |
| Config migration accidentally triggers provider work | Use preflight and offline fixture/reaggregation tests only |
| Study aggregation computes discarded legacy theory | Add empirical-only current mode and an import-boundary test |
| Existing analysis packages are misread as revised | Preserve legacy provenance and force recomputation for revised tables |
| Classical null is scientifically lost | Keep it only under explicit opt-in naming if still required |

---

## 13. Decisions required before implementation

1. **Legacy module end state:** rename `theory.py` to `matched_qvoter.py`, or
   remove/archive it from the runtime package?
2. **Public classical-null support:** should `matched_qvoter_null` remain an
   opt-in analysis reference, or should all runtime access be removed?
3. **Compatibility window:** should old `theory_comparison_enabled` callers
   receive one deprecation release, or should the new reference selection be a
   direct breaking change?
4. **Per-run artifact format:** retain revised CSV outputs for compatibility,
   or adopt the standardized long-format Parquet comparison everywhere?

The recommended choices are: rename and quarantine the legacy module, keep the
classical null opt-in, provide one narrow compatibility shim, and make Parquet
the authoritative format while retaining transitional CSV views if necessary.

---

## 14. Definition of done

- [ ] `theory_revised.py` is the only production source of relational values
      labelled as theory.
- [ ] `analysis.py`, `current.py`, configured analysis, and study aggregation
      contain no production import of legacy `.theory`.
- [ ] The canonical empirical/theory comparison is computed per physical cell
      through `single_affinity.theory_comparison`.
- [ ] Revised current moments come from
      `theory_revised.finite_horizon_current_moments`.
- [ ] Empirical-only `episode_current`/`cell_current` aggregation does not
      construct any theory reference.
- [ ] Every theory-bearing row records revised reference, module, API/semantics
      version, units, and analysis hash.
- [ ] Legacy q-voter products are removed or explicitly labelled as a
      classical null and cannot populate revised-theory fields.
- [ ] Studies 03–08 explicitly request the corrected estimator/reference
      family appropriate to each study.
- [ ] Full and `results_only` artifact profiles support the revised comparison,
      or emit precise missing-field reasons.
- [ ] Existing canonical observations can be reaggregated without provider
      calls.
- [ ] Targeted theory, estimator, config, aggregation, retention, and import
      tests pass.
- [ ] All changed relational configs pass provider-free preflight.
- [ ] Reports, plots, Comet keys, manifests, and packages use unambiguous
      revised-theory naming.
- [ ] No study-specific SLURM job or replacement CMI implementation is added.

---

## 15. Final authority rule

The enforceable repository rule after migration is:

> If a relational analysis artifact calls a value “theory,” its numerical
> value and semantics must be traceable to
> `mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised`.
> Any other mathematical model is a separately named diagnostic or classical
> null, never an alternative source for that field.

