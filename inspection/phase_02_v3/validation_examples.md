# Phase 2 validation examples

These intentionally invalid in-memory examples are not run configurations.
Secret values are neither read nor included.

## Invalid request concurrency

```text
llm_provider.request_concurrency: must be at least 1
```

## Forbidden inline secret field

```text
llm_provider.api_key: inline secret fields are forbidden; use an *_env variable-name field
llm_provider.api_key: unknown field
```
