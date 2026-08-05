# Agent Prompt 1 — Prepare the Source Project’s LLM and Prompt Component

## Role

You are working in the **source repository**, which already contains the functioning implementation. Your task is to identify, consolidate, and prepare the complete LLM-provider and prompt subsystem so that its canonical directory can later be copied into another repository.

Do not turn this into a separately published Python library. Do not redesign the entire repository. Preserve current behavior and make the smallest coherent refactor required to produce a self-contained, copyable component.

## Objective

Create one canonical, self-contained Python component containing:

1. **All LLM-provider logic**
   - the abstract base class, protocol, or common provider interface;
   - all concrete child/provider implementations;
   - specifically the university API provider, OpenAI provider, and locally running model provider;
   - common request and response models;
   - provider configuration models;
   - provider-specific exceptions and shared error handling;
   - provider registry, factory, or construction logic;
   - provider-owned retry, timeout, token-usage, model-capability, and response-normalization logic;
   - any small utility required for the providers to function.

2. **All prompt logic**
   - prompt data models and message types;
   - system, user, assistant, and tool-message construction;
   - prompt templates and reusable prompt fragments;
   - prompt rendering, interpolation, formatting, and validation;
   - prompt builders or factories;
   - prompt registries or lookup logic;
   - provider-specific message conversion, when this logically belongs to the prompt/provider boundary;
   - context assembly, truncation, or token-budget logic, when currently part of the prompt subsystem;
   - tests and fixtures required to verify prompt construction.

The resulting component must be usable without importing application-specific game, agent, experiment, logging, or domain logic.

---

## Small Execution Plan

### Phase 1 — Inventory

Inspect the repository before changing anything.

1. Locate every file that participates in:
   - provider abstraction;
   - concrete provider implementations;
   - provider configuration and construction;
   - request/response normalization;
   - prompt definitions, templates, builders, rendering, and message conversion.
2. Search imports and call sites so that indirectly required files are not missed.
3. Produce a short dependency map distinguishing:
   - code that belongs inside the portable component;
   - repository-specific code that must remain outside;
   - external Python dependencies;
   - environment variables and runtime assumptions.
4. Identify tests that already cover this behavior.

Do not assume that files with unrelated names are irrelevant. Follow actual imports and usages.

### Phase 2 — Consolidate the Canonical Component

Refactor the relevant code into one coherent canonical directory, following the repository's existing package conventions.

A reasonable target shape is:

```text
src/<source_package>/llm_runtime/
├── __init__.py
├── contracts.py
├── models.py
├── config.py
├── errors.py
├── registry.py
├── providers/
│   ├── __init__.py
│   ├── base.py
│   ├── university.py
│   ├── openai.py
│   └── local.py
└── prompts/
    ├── __init__.py
    ├── models.py
    ├── templates.py
    ├── builders.py
    ├── rendering.py
    └── conversion.py
```

This exact file layout is not mandatory. Preserve sensible existing modules rather than mechanically forcing every suggested file. The important requirement is that there is one clearly bounded directory containing the complete subsystem.

Requirements:

- Keep one canonical implementation; do not leave two independently editable copies.
- Preserve public behavior and existing configuration semantics.
- Avoid circular imports.
- Avoid importing optional provider dependencies at package-import time.
- Keep base contracts and common models importable even when OpenAI, Transformers, or another optional backend is not installed.
- Do not import game logic, agent orchestration, experiment tracking, Comet, dataset code, or repository-specific CLI code from the component.
- Replace repository-specific dependencies with:
  - explicit constructor arguments;
  - small configuration objects;
  - dependency injection;
  - or narrowly scoped callbacks/interfaces.
- Do not embed API keys, tokens, endpoints, or credentials.
- Preserve asynchronous behavior where it already exists.
- Preserve provider-specific behavior unless a change is required for portability.
- Keep prompts as data or templates rather than coupling them directly to a particular experiment.

### Phase 3 — Preserve Compatibility

Existing source-project imports and call sites must continue to work.

Use one of these approaches:

- update call sites to the new canonical imports; or
- add thin compatibility re-exports from old module paths.

Do not maintain duplicated implementations behind the old imports.

Where practical, expose a stable public surface from the component's `__init__.py`, for example:

```python
from .contracts import LLMProvider
from .models import LLMRequest, LLMResponse
from .registry import create_provider

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "create_provider",
]
```

Do not eagerly import every concrete provider from the package root if doing so would make optional dependencies mandatory.

### Phase 4 — Tests and Handoff Metadata

Add or adapt tests that verify:

1. the base provider contract;
2. construction of each concrete provider using mocked clients or transports;
3. request normalization;
4. response normalization;
5. exception mapping;
6. prompt rendering and interpolation;
7. construction of message sequences;
8. provider-specific prompt/message conversion;
9. package import without optional provider dependencies;
10. at least one end-to-end mocked call from prompt construction through provider response.

Do not make real paid API calls in tests.

Create a file inside the canonical component named:

```text
PORTING_MANIFEST.md
```

It must contain:

- canonical component path;
- list of included modules;
- list of concrete provider classes;
- list of prompt-related classes/functions;
- external dependencies grouped by provider;
- required and optional environment variables;
- repository-specific integrations intentionally excluded;
- recommended destination path in another repository;
- commands for running the relevant tests;
- any known portability limitations.

Also create:

```text
SOURCE_VERSION
```

containing the current Git commit hash. If the working tree is dirty, include the commit hash and state that uncommitted changes were present.

---

## Scope Decisions

### Include

Include logic whose primary responsibility is calling an LLM, configuring an LLM backend, normalizing LLM inputs/outputs, or constructing prompts/messages for those calls.

Examples:

- `LLMProvider` abstract class or protocol;
- all subclasses;
- OpenAI client wrapper;
- university API wrapper;
- local Transformers/vLLM/llama.cpp wrapper;
- provider factory;
- provider configuration;
- shared response and usage structures;
- prompt templates;
- message builders;
- prompt context objects;
- prompt rendering and validation;
- prompt-to-provider conversion;
- narrowly scoped retry and timeout behavior used by providers.

### Exclude

Keep the following outside unless they are truly generic and required by the subsystem:

- games and environments;
- agent policies;
- experiment execution;
- metrics and logging;
- Comet integration;
- result persistence;
- dataset logic;
- pharmacometrics or other domain-specific code;
- application-specific orchestration;
- repository-specific command-line interfaces.

If an excluded subsystem currently constructs prompts, move only the generic prompt mechanism into the portable component. Keep the domain-specific prompt content or experiment policy outside when it is not reusable.

---

## Implementation Constraints

- Prefer typed dataclasses, Pydantic models, or the repository's existing typed structures.
- Preserve type annotations.
- Use explicit exceptions rather than returning ambiguous sentinel values.
- Do not silently swallow provider errors.
- Keep raw provider responses optional and clearly marked.
- Avoid global mutable provider clients.
- Make configuration serializable where practical.
- Do not introduce a package publication or release workflow.
- Do not add Git submodules or symlinks.
- Do not perform unrelated cleanup.
- Do not change scientific or application behavior.

---

## Required Final Report

After implementing the changes, report:

1. the canonical directory that should be copied;
2. every provider implementation included;
3. every prompt subsystem module included;
4. old import paths changed or re-exported;
5. tests added or updated;
6. test command and result;
7. external dependencies required by each provider;
8. any remaining coupling that could not safely be removed;
9. the source Git commit recorded in `SOURCE_VERSION`;
10. exact copy instructions for the receiving repository.

---

## Definition of Done

The task is complete only when:

- there is one canonical directory containing all provider abstractions, all concrete provider children, and all generic prompt logic;
- the source repository still works;
- old duplicate implementations have been removed or reduced to compatibility re-exports;
- optional provider dependencies do not break unrelated imports;
- tests pass without real API calls;
- `PORTING_MANIFEST.md` and `SOURCE_VERSION` exist;
- the agent can identify one directory that can be copied as a unit into another repository.
