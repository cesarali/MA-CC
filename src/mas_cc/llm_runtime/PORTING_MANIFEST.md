# PORTING_MANIFEST

## Canonical component path

```
src/mas_cc/llm_runtime/
```

This is the one directory to copy as a unit into another repository.

```
src/mas_cc/llm_runtime/
├── __init__.py            # docstring only — providers/ and prompts/ are independent
│                            siblings (neither imports the other) and are imported directly
├── config.py               # LLMProviderConfig, PromptConfig, ProviderConfig
├── messages.py              # Message, MessageRole — LLM chat-message vocabulary
├── exceptions.py            # MasCCError, ValidationError, ConfigurationError
├── validation.py            # ValidationIssue, ValidationResult
├── secrets.py               # assert_secret_free (credential-shaped field guard)
├── PORTING_MANIFEST.md      # this file
├── SOURCE_VERSION
├── providers/
│   ├── __init__.py
│   ├── errors.py              # ProviderError
│   ├── protocols.py           # LLMProvider
│   ├── requests.py            # CompletionRequest
│   ├── responses.py           # CompletionResponse, ProviderUsage, redact_raw_response
│   ├── capabilities.py        # ProviderCapabilities
│   ├── registry.py            # ProviderRegistry, create_default_provider_registry, create_llm_provider
│   ├── budget.py              # BudgetCeiling, BudgetLimits, RuntimeBudgetGuard, BudgetGuardedProvider
│   ├── model_profiles.py      # ModelProfile, TemperatureRule, catalogue registry/fallbacks
│   ├── model_profiles.json    # checked-in per-model probe catalogue (package data)
│   ├── profiles.py            # request-normalizing provider decorator
│   ├── pricing.py             # ModelPricing, PricingCatalog, PricingSource impls, MonetaryAmount, ...
│   └── adapters/
│       ├── __init__.py
│       ├── mock.py                 # MockLLMProvider
│       ├── openai.py               # OpenAIProvider
│       ├── university.py           # UniversityProvider
│       ├── gemma_local.py          # GemmaLocalProvider
│       ├── _openai_compatible.py   # OpenAICompatibleProvider (shared base for openai/university)
│       └── _potsdam_network.py     # Windows VPN-bridge helper used only by UniversityProvider
└── prompts/
    ├── __init__.py
    ├── blocks.py          # PromptBlock, RenderedPromptBlock, UNBOUND, Unbound
    ├── contracts.py       # ResponseContract
    ├── compiled.py        # CompiledPrompt
    ├── full_prompt.py     # FullPrompt, CompilablePrompt
    ├── registry.py         # PromptRegistry, create_default_prompt_registry
    ├── versions.py         # PromptVersion
    ├── fingerprints.py     # canonical_json, fingerprint
    ├── tokenization.py     # TokenCounter, RegexTokenCounter
    ├── reporting.py        # PromptMarkdownLogger, render_prompt_request_markdown
    └── _values.py          # freeze, thaw (private value normalization helpers)
```

There are no compatibility re-export shims anywhere in the source repo: this
consolidation rewrote every internal caller (~40 files across `games/`,
`cli/`, `planning/`, `experiments/`, `runtime/`, plus one script under
`scripts/`) to import `mas_cc.llm_runtime.providers`/`mas_cc.llm_runtime.prompts`
directly. The former `src/mas_cc/llm_providers/` and the generic-mechanism
parts of `src/mas_cc/prompts/` no longer exist as separate packages.

## What moved here from where

- `src/mas_cc/llm_providers/` → `src/mas_cc/llm_runtime/providers/` (directory removed after rewiring callers)
- `src/mas_cc/prompts/` (minus `plugins/` and `examples.py`, see Excluded below) → `src/mas_cc/llm_runtime/prompts/` (directory removed after rewiring callers)
- `LLMProviderConfig` / `PromptConfig` (previously defined inline in `src/mas_cc/config/models.py`, alongside the much larger repository-wide run-config schema) → `src/mas_cc/llm_runtime/config.py`; `config/models.py` still re-exports them (`RunConfig` composes them alongside the sections it owns), since that module is not part of the portable component
- `ProviderError` (previously in `src/mas_cc/core/exceptions.py`) → `src/mas_cc/llm_runtime/providers/errors.py`. It subclasses `MasCCError`, which — see below — also ended up moving into `llm_runtime/exceptions.py`, so this is now an intra-component dependency, not a reach back into `core`.
- A ~35-line credential-field guard (previously the repo-wide `mas_cc.config.assert_secret_free`, which audits an entire `RunConfig` tree) was narrowed and duplicated into `llm_runtime/secrets.py`, scoped to just what `prompts/reporting.py` needs before writing a Markdown log.
- `Message`/`MessageRole` (previously in `src/mas_cc/core/records.py`) → `src/mas_cc/llm_runtime/messages.py`. `MessageRole` is literally `SYSTEM`/`USER`/`ASSISTANT`/`TOOL`, so this is LLM chat-message vocabulary, not a general framework primitive. `Message`'s `message_id`/`created_at` fields (typed against `core.ids.MessageId`/`core.records.Timestamp`) were dropped in the same move: nothing anywhere in the repository ever set them, so rather than pull those two core types along too, the dead fields were removed.
- `MasCCError`/`ValidationError`/`ConfigurationError` (previously `src/mas_cc/core/exceptions.py`) → `src/mas_cc/llm_runtime/exceptions.py`, and `ValidationIssue`/`ValidationResult` (previously `src/mas_cc/core/validation.py`) → `src/mas_cc/llm_runtime/validation.py`. `core/exceptions.py` and `core/validation.py` are now deleted (nothing else in `core/` needed them).

Every caller across the repo (`games/`, `config/`, `control/`, `cli/`,
`experiments/`, `metrics/`, tests, one script) that used to import any of
these six names from `mas_cc.core` now imports them from
`mas_cc.llm_runtime` instead. **`mas_cc.llm_runtime` has zero import-time
dependency on `mas_cc.core`** — verified directly (see "Copying to another
repository" below) rather than merely documented.

## Concrete provider classes

- `MockLLMProvider` (`providers/adapters/mock.py`) — deterministic, no network, used in tests
- `OpenAIProvider` (`providers/adapters/openai.py`) — official OpenAI chat-completions API
- `UniversityProvider` (`providers/adapters/university.py`) — University of Potsdam OpenAI-compatible proxy
- `GemmaLocalProvider` (`providers/adapters/gemma_local.py`) — local Transformers/accelerate model
- `OpenAICompatibleProvider` (`providers/adapters/_openai_compatible.py`) — shared HTTP base class for `OpenAIProvider`/`UniversityProvider`; private, not part of the public surface

All four are registered lazily by string module path in
`create_default_provider_registry()`; importing `llm_runtime.providers` (or
`llm_runtime.providers.adapters`) never imports `openai`, `requests`,
`torch`, or `transformers` — those are imported lazily inside adapter
methods, only when a call is actually made.

`create_llm_provider()` wraps the OpenAI-compatible providers in
`ProfiledLLMProvider`. The decorator looks up the concrete provider/model pair
in the packaged `model_profiles.json`, applies known temperature/seed rules,
and warns once for each automatic rule applied. Unknown entries remain usable
through a profile explicitly marked `probe_source="inferred"`. The adapter's
`discover_models()` method is shared with the opt-in exploratory probe so model
listing and normal university endpoint discovery cannot drift apart.

## Prompt subsystem classes/functions

Kernel (generic mechanism, ships with no game/paper content):
`PromptBlock`, `RenderedPromptBlock`, `UNBOUND`/`Unbound`, `ResponseContract`,
`CompiledPrompt`, `FullPrompt`, `CompilablePrompt`, `PromptRegistry`,
`create_default_prompt_registry` (returns an **empty**
registry in this component — see Portability limitations),
`PromptVersion`, `canonical_json`, `fingerprint`, `TokenCounter`,
`RegexTokenCounter`, `PromptMarkdownLogger`, `render_prompt_request_markdown`,
`freeze`/`thaw`.

Config models: `LLMProviderConfig`, `PromptConfig`, `ProviderConfig` (alias
of `LLMProviderConfig`). Message vocabulary (`messages.py`, shared by both
`providers` and `prompts`): `Message`, `MessageRole`. Exceptions
(`exceptions.py`): `MasCCError`, `ValidationError`, `ConfigurationError`.
Validation (`validation.py`): `ValidationIssue`, `ValidationResult`.

`providers` and `prompts` are independent siblings: importing one never
imports the other (enforced by
`tests/mas_cc/test_llm_runtime.py::test_llm_runtime_providers_and_prompts_are_independent_siblings`),
so `llm_runtime/__init__.py` itself re-exports nothing — import the two
subpackages directly.

## External dependencies, grouped by provider

- **Core (always required):** standard library only. `llm_runtime` has no
  dependency on `mas_cc.core` or any other part of this repository — see
  "Copying to another repository" below.
- **mock:** none.
- **openai / university** (`OpenAICompatibleProvider`): `requests` (imported
  lazily inside `_get_session`).
- **gemma_local:** `torch`, `transformers`, `accelerate`, `safetensors`,
  `huggingface-hub` (all imported lazily inside `GemmaLocalProvider`'s model
  loading path — matches this repo's `gemma4` optional extra in
  `pyproject.toml`).
- **university** additionally: on Windows only, `_potsdam_network.py` may
  shell out to a local VPN-bridge helper process; this is a narrow,
  network-environment-specific concern, not a pip dependency.

The setuptools package-data declaration for this component must include
`mas_cc.llm_runtime.providers/model_profiles.json`; otherwise exact profiles
are unavailable from an installed wheel.

Importing `mas_cc.llm_runtime.providers` or `mas_cc.llm_runtime.prompts`
never requires `openai`, `requests`, `torch`, or `transformers` to be
installed — verified by
`tests/mas_cc/test_llm_runtime.py::test_llm_runtime_imports_without_optional_provider_dependencies`.

## Environment variables

- `OPENAI_API_KEY` — default credentials env var name for `OpenAIProvider` (overridable via `LLMProviderConfig.credentials_env`)
- `POTSDAM_API_KEY` — default credentials env var name for `UniversityProvider`
- `BASE_POTSDAM_LLM_URL` — default base-URL env var name for `UniversityProvider`
- `HF_HOME` — optional; if unset, `GemmaLocalProvider` leaves the Hugging Face cache location to its default resolution
- None of the above are read at import time. `LLMProviderConfig` only ever
  carries the *names* of environment variables (`credentials_env`,
  `base_url_env`); it never carries a secret value itself, and
  `llm_runtime/secrets.py` guards against that shape slipping into a log.

## Repository-specific integrations intentionally excluded

- `mas_cc.games.*`, `mas_cc.experiments.*`, `mas_cc.observability.*`
  (Comet/audit), `mas_cc.storage.*`, `mas_cc.control.*`, `mas_cc.cli.*`,
  `mas_cc.planning.*` — nothing in `llm_runtime` imports any of these
  (enforced by `test_llm_runtime_does_not_import_games_experiments_or_cli`).
- `mas_cc.config` (the repository-wide `RunConfig`/`GameConfig`/`GridSpec`/
  YAML-loading system) — `llm_runtime` depends only on the two config
  dataclasses it owns itself (`config.py`), not on the larger package.
- Game- and paper-specific prompt *content* (game rules, "Player 1/2"
  phrasing, specific experiment wording) built on top of the generic
  kernel, not moved into `llm_runtime`:
  - `src/mas_cc/games/prompt_library/basic_choice_v3.py` and
    `hidden_profile_v3.py` — V3 kernel-based fixtures, live under `games/`.
  This location is wired into this repository's own
  `mas_cc.games.registry.create_default_prompt_registry()`, which is
  *not* part of the portable component.
- `src/naming_game/` — the legacy pre-reorganization implementation; not
  touched or inventoried as part of this bundle by explicit decision (the
  active line is `mas_cc`).

## Copying to another repository

`llm_runtime` has **zero** dependency on `mas_cc.core`, or on any other part
of this repository. This was verified, not just claimed — an earlier
revision of this bundle did depend on `mas_cc.core` for a handful of types;
each was resolved by moving the type in (if it was genuinely LLM-specific)
or removing the need for it (if it was dead weight), never by leaving a
cross-package dependency in place:

- `Message`/`MessageRole` moved into `llm_runtime/messages.py` outright —
  they're LLM chat-message vocabulary (`MessageRole` is literally
  `SYSTEM`/`USER`/`ASSISTANT`/`TOOL`). Every caller across the repo (not
  just `llm_runtime`) was rewired to import them from
  `mas_cc.llm_runtime.messages`, which is what made the move safe: had
  `Message` been *duplicated* in two places instead of moved, a game
  building a `Message` from one definition and handing it to a provider
  expecting the other would fail an `isinstance` check.
- `MasCCError`/`ValidationError`/`ConfigurationError` and
  `ValidationIssue`/`ValidationResult` moved into `llm_runtime/exceptions.py`
  and `llm_runtime/validation.py` the same way, for the same reason — they
  turned out to be used throughout `games/`, `control/`, `config/`, `cli/`
  for validating `GameConfig`/`ControlConfig`/etc., nothing LLM-specific
  about that, but a single canonical location beats two. Every caller
  repo-wide was rewired; `core/exceptions.py` and `core/validation.py` are
  now deleted, since nothing else needed them.
- `Message.message_id`/`created_at` (typed against `core.ids.MessageId`/
  `core.records.Timestamp`) were simply dropped: nothing anywhere in the
  repository ever constructed a `Message` with either field set, so rather
  than pull two more core types into the component, the dead fields went
  away. `Timestamp` and `MessageId` still exist in `mas_cc.core` for
  whatever (non-`llm_runtime`) code wants a general-purpose timestamp or
  message identifier.

Verified by
`tests/mas_cc/test_llm_runtime.py::test_llm_runtime_imports_without_optional_provider_dependencies`
and a plain `grep -rn "mas_cc\.core" src/mas_cc/llm_runtime/` (empty).

**To copy:**

```
src/<destination_package>/llm_runtime/    # this directory, verbatim — nothing else needed
```

Update the one absolute-import prefix (`mas_cc.llm_runtime` →
`<destination_package>.llm_runtime`) throughout after copying — all
internal imports are absolute (`from mas_cc.llm_runtime...`), not
relative-across-package, so this is a mechanical rename.

## Running the tests

```
python -m pytest tests/mas_cc/test_llm_runtime.py -v
python -m pytest tests/mas_cc/test_llm_providers.py \
    tests/mas_cc/test_prompts_v3.py tests/mas_cc/test_provider_economics.py \
    tests/mas_cc/test_import_safety.py -v
```

The second line's test files kept their original names but now import
`mas_cc.llm_runtime.providers`/`mas_cc.llm_runtime.prompts` directly (there
is no compatibility layer). No test makes a real paid API call; `mock` is
the only adapter exercised.

Result at the time this manifest was written: all of the above pass. The
full repository suite (`python -m pytest -q`) has two pre-existing,
unrelated failures (`test_phase_6_inspection_contract_audit_and_determinism`,
`test_phase_6_standard_inspection_cli`) caused by a missing baseline fixture
file (`inspection/realignment_v3/baseline/phase_06/selected_audit_traces.jsonl`,
not checked into the repository) — unrelated to this bundling work and
present before it.

## Known portability limitations

- `create_default_prompt_registry()` in this component returns an **empty**
  registry (no built-in `FullPrompt` registrations). This repository's own
  `mas_cc.games.registry.create_default_prompt_registry` wraps it and
  additionally registers `games/prompt_library/` content; that wrapper is
  intentionally not part of the portable component. A destination
  repository registers its own content via `PromptRegistry.register(...)`.
- `UniversityProvider`'s Windows VPN-bridge helper
  (`adapters/_potsdam_network.py`) assumes a specific institutional network
  setup (University of Potsdam); it is harmless (a no-op) outside that
  environment but is not generically useful — a destination repository
  without that network can ignore it or delete it, since `UniversityProvider`
  is otherwise a thin `OpenAICompatibleProvider` subclass.
