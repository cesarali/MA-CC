# report_only_authored_q3_q12_chatoss_v5

Version-5 grounded report-only Task-003 study (full 60-repetition study).

## Scientific design

- arms: truth control, false control, and no control
- public controller identity: Agent N+1 (internal controller provenance remains private/auditable)
- participant public actions: REPORT or NONE
- controller public actions: REPORT only (or no post when no intervention is selected)
- citation scope: active memory plus grounded REPORT facts sampled in the current update
- empty citable set: deterministic NONE without discarding the private vote/reason
- q: 3 and 12; controlled arms set both `social_group_size` and `sensor_sample_size` to q
- epistemic persistence: 0.70, 0.85, 1.00
- intervention budgets for controlled arms: [6, 12, 18]
- beta: 4.0; theta/threshold: 0.5
- rounds: 30
- repetitions per cell: 60

## Size

- controlled cells: 36
- no-control cells: 6
- total cells: 42
- total episodes: 2520

All six configs share one paired-initialization directory:

`/shared/home/cesar/work/results/studies/report_only_authored_q3_q12_chatoss_v5_initializations`

Materialize and validate those artifacts before submission. No paid provider run is authorized by these files.
The analysis recipe keeps q in its aggregate grouping and facets line plots by q so q=3 and q=12 are not silently pooled.
