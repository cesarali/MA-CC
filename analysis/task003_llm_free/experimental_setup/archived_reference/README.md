# archived_reference — what was actually run in September

`task003_archived_reference.json` describes the setup behind study
`21-09-2026-full-vs-report-v1`, the only task_003 setup with LLM data.

**Do not build it.** It is kept so the frozen setups can be compared with it.

Its controller pool is César's original 24-fact pool. Facts were admitted by
whether they helped the false target (A2), and the **same** pool was then reused
when the controller targeted the truth (A0). So in that study, a controller
"steering to A0" recommended A0 while quoting evidence picked to undermine it.
Never read its A0-arm numbers as evidence about steering toward the truth.

| | archived | task003-symmetric (frozen) |
|---|---|---|
| agents per allocation A0 / A1 / A2 | 16 / 5 / 3 | 8 / 8 / 8 |
| decisive facts held | 6 of 6 | 6 of 6 |
| proofs the agents can assemble | 4 | 12 |
| controller pool(s) | one, 24 facts, chosen for A2, used for both targets | two, 12 each, one per target |
