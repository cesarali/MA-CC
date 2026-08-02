# Provider boundary diff

- Compiled messages equal wire messages: **true**
- Prompt family, versions, hashes, block values, and response contracts remain local request metadata and are excluded from `wire_messages()`.
- Provider adapters receive only `CompletionRequest`.
