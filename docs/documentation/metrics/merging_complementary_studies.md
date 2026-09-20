# Merge complementary studies before aggregation

Use `mas-cc study merge` when independent runs cover complementary parts of
one scientific design, such as a budget sweep of `[6, 12, 18]` followed by
another sweep of `[24, 30]`. The runs need not have been submitted as an
extension. Finish both runs first: merging requires valid, complete inputs.

```bash
mas-cc study merge \
  --study-dir results/studies/recomm_only_q12_chatoss \
  --study-dir results/studies/recomm_only_q12_chatoss_larger_budgets \
  --output-dir results/studies/recomm_only_q12_chatoss_combined

mas-cc study aggregate \
  --study-dir results/studies/recomm_only_q12_chatoss_combined
```

Pass result directories, rather than `configs/runs/...` folders. Inputs can
be standardized study roots with run trees, study roots with retained
canonical analysis tables, or extracted analysis directories containing
`tables/`, `validation.json`, and `provenance/`. Extract ZIP packages first.
Legacy canonical CSV tables are readable; output uses compressed Parquet.
Saved configs must remain resolvable in the current environment, including
any referenced components. Dataset paths used by the selected analysis recipe
must also be accessible when aggregating.

The merge command creates a new, independent snapshot and leaves both source
studies unchanged. It retains canonical cells, episodes, rounds, micro-slots,
and available canonical diagnostics. It writes one resolved YAML config per
retained cell, so a union of irregular grids does not introduce unobserved
Cartesian-product cells. These configs describe retained observations; the
merged study is an analysis input, not a submission or resume target.

It does not copy or average precomputed estimates, plots, bootstrap draws,
or execution artifacts, and makes no provider calls. Run the existing
aggregator separately to recompute estimates and draw the wider phase diagram.
Scientific cells remain separate estimator units. Any marginalization or
derived pooling follows the selected analysis recipe. Choose complementary
runs with compatible fixed design parameters; merging does not turn different
models or protocols into a single scientific condition.

## Overlapping cells

By default, an overlapping scientific cell fails the merge. Comparison uses
the existing protocol fingerprint with all resolved physical parameters
included. Operational fields, target repetition counts, descriptive study
labels, and experiment tags are excluded. Seed, task, model, control, and
prompt differences remain part of identity.

If both studies include a repeated baseline or budget cell and you intend to
use the first study's entire cell, add:

```text
--overlap keep-first
```

Source order matters. This explicitly discards the later overlapping cell,
even if it contains different observations or more repetitions. It does not
pool repetitions or deduplicate individual episodes. Discarded cells and
their retained source are listed in `merge_manifest.json`. Combining extra
repetitions at the same cell is a separate workflow requiring episode-level
identity and seed checks.

Paired initialization is checked across merged cells when the required state
fields are retained. Inconsistent physical initial states fail the merge.

## Analysis recipe and provenance

Identical source analysis recipes are reused automatically. When they differ,
choose one explicitly:

```text
--analysis-recipe configs/runs/relational_reasoning/blackboard_game/iclr_experiments/recomm_only_q12_chatoss/analysis.yaml
```

The output includes `study_manifest.json`, `submission_manifest.csv`,
`configs/`, `analysis.yaml`, `merge_manifest.json`, and the canonical analysis
inputs with validation metadata. Original source study, cell, and config
indices are retained in `merge_source_*` table columns. Run IDs and canonical
cell/episode keys are qualified to avoid collisions between independent runs.
Aggregation includes `merge_manifest.json` in the final package provenance.

The destination must be new and outside the sources. No files are uploaded
to a cloud bucket by merging or aggregating. Use the upload workflow separately
when you want to publish the finished package.
