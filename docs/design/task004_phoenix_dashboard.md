# Task 004 Phoenix dashboards

The four completed controller suites are imported into the local Phoenix server
at `http://localhost:6006`. Each project contains 40 episode traces: 10 each
for truth/false controller targets at rho 0.75 and 1.0.

| Budget | Communication | Phoenix project |
| --- | --- | --- |
| Episode b=90 | Report only | `task004-episode-b90-report-only` |
| Episode b=90 | Full communication | `task004-episode-b90-full-comms` |
| Round b=3, sigmoid gate | Report only | `task004-round-b3-report-only` |
| Round b=3, sigmoid gate | Full communication | `task004-round-b3-full-comms` |

Open `http://localhost:6006/redirects/projects/<project-name>` to view a
project. Episode traces are tagged with `mas_cc.source_arm`,
`mas_cc.epistemic_persistence`, `mas_cc.communication_profile`, and budget
settings. Expand a round, then an LLM span to inspect its `input.value` prompt
and `output.value` response. Controller spans are named `controller
communication rN`; agent spans carry the agent ID and decision stage.

These are retrospective imports of completed Cesar artifacts, so their Phoenix
timestamps are import times. Controller prompts are present on every controller
span except 34 older attempts in the episode-b90 report-only suite, whose
records kept only summaries. Agent prompts were sampled by the run logger
(typically 200 per episode); spans lacking a retained prompt have
`mas_cc.prompt_retained=false`. The archived round board reconstructed from
agent decisions omits controller posts; the controller's full observed board
is present inside its retained prompt.

The repeatable importer is `scripts/local/phoenix_import.py`. The read-only
Cesar export recipe is `scripts/local/phoenix_export_task004.py`.
