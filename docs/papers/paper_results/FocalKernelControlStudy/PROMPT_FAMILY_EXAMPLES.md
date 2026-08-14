# Prompt-family definitions and matched examples

The six families differ only in the framing of social identity/reliability. The
linked files are the exact frozen prompts for the same base state (`state_0001`,
task 1), making them counterfactual twins.

1. **Anonymous** — participant identities are hidden and no reliability
   information is supplied. [Exact prompt](prompt_examples/bucket_01_anonymous__state_0001.md)
2. **Persistent identity** — stable agent identities are shown, without a
   reliability claim. [Exact prompt](prompt_examples/bucket_02_persistent_identity__state_0001.md)
3. **Positive reputation** — the focal agent is described as having previously
   provided useful information or recommendations. [Exact prompt](prompt_examples/bucket_03_positive_reputation__state_0001.md)
4. **Negative reputation** — the focal agent is described as having previously
   provided misleading or incorrect information or recommendations.
   [Exact prompt](prompt_examples/bucket_04_negative_reputation__state_0001.md)
5. **Social reputation** — a third party previously described the focal agent as
   reliable, while the decision maker is told that this assessment is
   unverified. [Exact prompt](prompt_examples/bucket_05_social_reputation__state_0001.md)
6. **Strategic uncertainty** — participants may possess different information
   and objectives, and some recommendations may be strategic, without naming
   which participants differ. [Exact prompt](prompt_examples/bucket_06_strategic_uncertainty__state_0001.md)

The prompt manifest records the SHA-256 of every exact prompt, including these
examples, under `provenance/PROMPT_MANIFEST.jsonl`.
