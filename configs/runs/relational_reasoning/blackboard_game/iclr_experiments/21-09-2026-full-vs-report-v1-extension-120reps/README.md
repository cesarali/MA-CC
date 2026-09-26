# 21-09-2026 full-vs-report V1 extension to 120 repetitions

This directory is the complete 120-repetition target for extending
`21-09-2026-full-vs-report-v1`. It preserves the original V1 scientific
conditions and operational settings; only the repetition target is raised
from 60 to 120.

The executable target is `execution.repetitions: 120`. The duplicated legacy
`experiment.metadata.repetitions` annotation intentionally remains `60`
because it participates in the original scientific protocol fingerprint;
changing it would create incompatible cells instead of extending them.
The controlled-arm operational safety caps are increased only enough to cover
the conservative preflight estimate for the complete 120-repetition target;
budget caps are excluded from scientific identity.
Both arms use `execution.parallelism: 16`, matching the prepared
`b6-b9-b15` study's episode parallelism. Parallelism is also excluded from
scientific identity.
The later `controller_allow_mixed_message_types` option is intentionally
omitted. Its resolved default is `false`, preserving V1's historical
single-controller-message-type-per-dawn behavior in full-communication cells.

The extension must be planned against the existing study result root:

`/shared/home/cesar/work/results/studies/21-09-2026-full-vs-report-v1`

The target contains 28 cells and 3,360 canonical episodes. The existing
partial study has 1,382 validated completed episodes available for reuse, so
the expected delta is 1,978 episodes: the 298 holes below repetition 60 plus
1,680 new episodes for repetitions 60 through 119.

Do not submit this directory as an unrelated fresh study. Use the study
extension workflow so retained episode identities and lineage are recorded.
The 60 initialization artifacts for repetitions 60 through 119 must also be
materialized before a real extension submission.
