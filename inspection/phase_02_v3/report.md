# Phase 2 inspection report

- Status: **PASS**
- Command: `mas-cc inspect phase 2 --config configs/runs/provider_smoke_test_v3.yaml --output-dir inspection/phase_02_v3`
- Code paths exercised: YAML loading, relative component lookup, recursive overrides, non-secret environment defaults, schema validation, immutable model construction, resolved export, secret audit, and invalid-example diagnostics.
- Input: `/home/cesarali/LanguageGames/MA-CC/configs/runs/provider_smoke_test_v3.yaml`
- Expected behavior: all component references are expanded; defaults are explicit; credential fields contain environment-variable names only; repeated loading produces identical values and YAML.
- Deviations or warnings: none.

## Results

- Deterministic repeated load: passed
- Invalid examples rejected with exact fields: passed
- Resolved output secret-marker audit: passed

## Files to inspect manually

- `input_config.yaml` — unresolved run composition supplied to the command.
- `resolved_config.yaml` — component references and defaults fully expanded.
- `config_schema.json` — machine-readable resolved run schema.
- `prompt_schema_v2.json` — standalone prompt component Version 2 schema.
- `v1_to_v2_migration_examples.md` — exact migration shape and diagnostics.
- `resolved_prompt_component.yaml` — registered order and definition fingerprint,
  without dynamic block values.
- `secret_scan.json` — machine-readable credential and secret-value audit.
- `validation_examples.md` — exact field diagnostics for intentional failures.
- `manifest.json` — artifact hashes and machine-readable pass/fail checks.
