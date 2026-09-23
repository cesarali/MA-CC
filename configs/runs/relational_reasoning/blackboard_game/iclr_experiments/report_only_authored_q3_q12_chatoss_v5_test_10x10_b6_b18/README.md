# report_only_authored_q3_q12_chatoss_v5_test_10x10_b6_b18

Version-5 grounded report-only Task-003 study (10-round, 10-repetition test study).

## Scientific design

- arms: truth control, false control, and no control
- public controller identity: Agent N+1 (internal controller provenance remains private/auditable)
- participant public actions: REPORT or NONE
- controller public actions: REPORT only (or no post when no intervention is selected)
- communication handles: `communication_profile: report_only` and
  `controller_authoring: llm_authored`; ordinary agents are always LLM-authored
- fixed-dose invariant: an inactive control round posts 0 messages; an active
  control round posts exactly the configured intervention budget `b`
- citation scope: active memory plus grounded REPORT facts sampled in the current update
- empty citable set: deterministic NONE without discarding the private vote/reason
- q: 3 and 12; controlled arms set both `social_group_size` and `sensor_sample_size` to q
- epistemic persistence: 0.70, 0.85, 1.00
- intervention budgets for controlled arms: [6, 18]
- beta: 4.0; theta/threshold: 0.5
- rounds: 10
- repetitions per cell: 10

## Size

- controlled cells: 24
- no-control cells: 6
- total cells: 30
- total episodes: 300

All six configs share one paired-initialization directory:

`/shared/home/cesar/work/results/studies/report_only_authored_q3_q12_chatoss_v5_test_10x10_b6_b18_initializations`

Materialize and validate those artifacts before submission. No paid provider run is authorized by these files.
The analysis recipe keeps q in its aggregate grouping and facets line plots by q so q=3 and q=12 are not silently pooled.
