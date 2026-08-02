# Remaining legacy prompt references

The remaining Version 1 symbols are limited to:

- `src/mas_cc/prompts/compatibility.py`, `context.py`, `composer.py`, and
  `examples.py`;
- renderer-only files `plugins/ashery_2025.py`, `basic_binary_choice.py`,
  `hidden_profile_paper.py`, and `social_conventions_paper.py`;
- `tests/mas_cc/test_prompts.py`, which freezes historical behavior through an
  explicit import from `mas_cc.prompts.compatibility`.

They are unregistered by default. No game, planning, runtime, CLI, provider
adapter, benchmark script, or notebook imports them. The primary
`mas_cc.prompts` API does not export their symbols.

Deletion is scheduled after the historical Version 1 fixture suite is retired;
no later phase may add a new dependency on this compatibility surface.
