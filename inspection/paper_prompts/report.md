# Paper prompt example report

- Status: **PASS**
- No LLM was called.
- Social-conventions source: `pdfs/Emergence of social conventions supplementary.pdf`, section **Prompting → Example Prompt**.
- HiddenBench source: `pdfs/Systematic Failures in Collective Reasoning under Distributed Information in.pdf`, Appendix **A.4 Prompts and Communication Templates**.
- HiddenBench fixture: `/home/cesarali/LanguageGames/MA-CC/scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/scaled/exact_replication/N_32.json`, task `1`, agent `0`.
- Token counts are dependency-free estimates from `mas_cc_regex_v1_estimate`.

## What is adapted

- The social-conventions wording, F/J actions, simultaneous choice, +100/−50 payoffs, bounded memory, answer-first response, and final user request follow the supplementary example. The concrete score and three memory rows are an inspection fixture.
- HiddenBench uses the downloaded scenario, shared facts, and the selected agent's private fact. Fact order is deterministically shuffled. The two public transcript lines are inspection fixtures constructed from other agents' private packets.
- The HiddenBench `correct_answer` audit field is never copied into the prompt context or request.
- Lego blocks are merged by consecutive role so the final transmission is exactly one `system` message followed by one `user` message, matching both papers' request shape.

## Readable requests

- [`all_requests.md`](all_requests.md) — all five complete requests in one document.
- [`social_conventions/request.md`](social_conventions/request.md) — one convention decision.
- [`hiddenbench_first_speaker/request.md`](hiddenbench_first_speaker/request.md) — the first discussion turn.
- [`hiddenbench_discussion/request.md`](hiddenbench_discussion/request.md) — a later discussion turn with public transcript.
- [`hiddenbench_pre_vote/request.md`](hiddenbench_pre_vote/request.md) — a vote before discussion.
- [`hiddenbench_post_vote/request.md`](hiddenbench_post_vote/request.md) — one post-discussion vote.

Each example directory also contains the source prompt config, context, rendered blocks, and compiled JSON messages for machine inspection.
