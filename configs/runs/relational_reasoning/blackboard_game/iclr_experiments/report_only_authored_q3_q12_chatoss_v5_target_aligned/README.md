# Target-aligned Task 003 report-only extension

This is a new version of the six-arm v5 ICLR study. It preserves q=3/12,
budgets 6/12/18, persistence 0.70/0.85/1.00, 60 repetitions per cell,
provider/model, seeds, and the paired initialization artifact directory.
It has 42 cells and 2,520 planned episodes. The result root is separate from
the original v5 study.

The false controller receives the frozen 24 controller-compatible true facts
ranked for ALLOCATION_2. The truth controller receives 23 other true facts:
six decisive corrective facts and 17 facts that individually raise the
symbolic probability of ALLOCATION_0. Two facts previously labeled neutral
lower that probability and are excluded. The exact ordered truth pool and
individual symbolic lifts are frozen in `truth_pool_audit.json`. The decisive
facts rank ahead of the other truth-supporting facts when deterministic
fallback selects reports; ties rotate reproducibly by episode seed and round. The
LLM-authored controller sees only the arm's eligible pool and chooses grounded
reports from it. No-control has no controller fact pool.

`controller_report_pool_mode: target_aligned_v1` is opt-in. The truth arm
provides its audited IDs in `controller_report_pool_fact_ids`; validation
rejects missing, duplicate, unknown, overlapping, or non-decisive truth pools. The old v5 files
retain their original scientific meaning and must not be resumed as this study.
The frozen source pool hash still identifies the Task 003 artifact; the
`effective_controller_pool_sha256` metadata identifies the fact IDs actually
eligible in each controlled arm. These hashes are calculated from compact
JSON arrays of ordered fact IDs.

The frozen symbolic `DECISIVE` profile gives posterior probability 1.0 to
ALLOCATION_0 when all six decisive facts are combined; this is a task-design
check, not a claim about language-model behavior.

The original Task 003 facts and paired private assignments are unchanged.
The study requires the existing frozen task and initialization artifacts at the
paths in each YAML on the execution site. Validate those paths and run offline
preflight there before any real submission. No provider run is authorized by
these files.
