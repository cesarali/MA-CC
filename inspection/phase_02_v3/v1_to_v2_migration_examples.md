# Prompt component migration: Version 1 to Version 2

Version 1 remains readable as a temporary migration input. Version 2 selects a
registered concrete `FullPrompt`; its Python class owns the authoritative block
order.

## Version 1 input

```yaml
schema_version: 1
prompt_family: basic_choice
prompt_version: 1
blocks: [task, rules, private_state, recent_memory, current_interaction]
response_contract:
  type: choice_only
  allowed_values: [A, B]
options:
  message_mode: merge_consecutive_roles
  block_separator: "\n\n"
```

## Version 2 equivalent

```yaml
schema_version: 2
prompt_family: basic_choice
prompt_version: 1
message_mode: merge_consecutive_roles
block_separator: "\n\n"
response_contract:
  type: choice_only
  allowed_values: [A, B]
```

Diagnostics: remove `blocks`; move `message_mode` and `block_separator` from
`options` to the component top level. The resolved export records the registered
block manifest and definition hash without binding dynamic private values.
