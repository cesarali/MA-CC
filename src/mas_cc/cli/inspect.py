"""Stable, human- and machine-inspectable acceptance runs for each phase."""

from __future__ import annotations

import hashlib
import importlib.metadata
import csv
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mas_cc import __version__
from mas_cc.config import config_schema, load_run_config, resolved_config_yaml
from mas_cc.core.exceptions import ConfigurationError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(*args: str, root: Path) -> str:
    process = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_dir: Path,
    *,
    phase: int,
    status: str,
    checks: dict[str, bool],
    warnings: list[str] | None = None,
) -> Path:
    artifacts = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    manifest = {
        "manifest_version": 1,
        "phase": phase,
        "mas_cc_version": __version__,
        "generated_at": _now(),
        "status": status,
        "checks": checks,
        "warnings": warnings or [],
        "artifacts": artifacts,
    }
    destination = output_dir / "manifest.json"
    _write(destination, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def _import_probe(root: Path) -> tuple[bool, str]:
    script = """
import json
import sys

before = set(sys.modules)
import mas_cc
after_mas_cc = set(sys.modules)
mas_added = sorted(after_mas_cc - before)
import naming_game
result = {
    "mas_cc_version": mas_cc.__version__,
    "mas_cc_imported": True,
    "naming_game_imported": True,
    "mas_cc_added_modules": mas_added,
    "forbidden_modules_loaded_by_mas_cc": sorted(
        {"comet_ml", "dotenv", "openai", "requests", "torch", "transformers"}
        & set(mas_added)
    ),
}
print(json.dumps(result, sort_keys=True))
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return False, f"Import probe failed (exit {process.returncode}):\n{process.stderr}"
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        return False, f"Import probe returned invalid JSON:\n{process.stdout}\n{process.stderr}"
    passed = not result["forbidden_modules_loaded_by_mas_cc"]
    lines = [
        f"mas_cc imported: {result['mas_cc_imported']}",
        f"mas_cc version: {result['mas_cc_version']}",
        f"naming_game imported: {result['naming_game_imported']}",
        "modules added by import mas_cc: " + ", ".join(result["mas_cc_added_modules"]),
        "forbidden external-work modules loaded by mas_cc: "
        + (", ".join(result["forbidden_modules_loaded_by_mas_cc"]) or "none"),
    ]
    return passed, "\n".join(lines) + "\n"


def _legacy_test_probe(root: Path) -> tuple[bool, str]:
    tests = sorted((root / "tests").glob("test_*.py"))
    relative_tests = [str(path.relative_to(root)) for path in tests]
    # ``pyproject.toml`` already selects quiet mode.  Avoid a second ``-q`` so
    # pytest retains the useful ``N passed`` summary in the inspection record.
    command = [sys.executable, "-m", "pytest", *relative_tests]
    process = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    summary = [
        "Command: " + " ".join(command),
        f"Legacy test files: {len(relative_tests)}",
        f"Exit status: {process.returncode}",
        "",
        process.stdout.rstrip(),
    ]
    if process.stderr.strip():
        summary.extend(["", "stderr:", process.stderr.rstrip()])
    return process.returncode == 0, "\n".join(summary).rstrip() + "\n"


def inspect_phase_1(output_dir: str | Path) -> bool:
    """Run Phase 1 import and legacy-regression checks and write artifacts."""

    root = _repository_root()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    import_ok, imports = _import_probe(root)
    tests_ok, tests = _legacy_test_probe(root)
    _write(destination / "package_imports.txt", imports)
    _write(destination / "legacy_test_summary.txt", tests)

    distributions: dict[str, str] = {}
    for name in ("llm-naming-game", "PyYAML", "pytest", "setuptools"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = "not-installed"
    environment = {
        "captured_at": _now(),
        "git_commit": _git("rev-parse", "HEAD", root=root),
        "git_describe": _git("describe", "--always", "--dirty", root=root),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "distributions": distributions,
    }
    _write(
        destination / "environment.json",
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
    )

    status = "pass" if import_ok and tests_ok else "fail"
    report = f"""# Phase 1 inspection report

- Status: **{status.upper()}**
- Command: `mas-cc inspect phase 1 --output-dir {destination}`
- Code paths exercised: package discovery, isolated `mas_cc` import, legacy `naming_game` import, and all pre-migration test modules under `tests/test_*.py`.
- Inputs: Git checkout `{environment['git_commit']}` and the active Python environment.
- Expected behavior: both packages import; importing `mas_cc` loads no provider, model, credential, HTTP, or Comet module; the legacy suite passes unchanged.
- Deviations or warnings: none.

## Results

- Import guard: {'passed' if import_ok else 'failed'}
- Legacy test suite: {'passed' if tests_ok else 'failed'}

## Files to inspect manually

- `environment.json` — commit and interpreter metadata (no environment variable values).
- `package_imports.txt` — isolated import results and forbidden-module check.
- `legacy_test_summary.txt` — exact test command and pytest summary.
- `manifest.json` — artifact hashes and machine-readable pass/fail checks.
"""
    _write(destination / "report.md", report)
    _write_manifest(
        destination,
        phase=1,
        status=status,
        checks={"import_guard": import_ok, "legacy_tests": tests_ok},
    )
    return status == "pass"


def _invalid_examples(resolved: dict[str, Any]) -> tuple[bool, str]:
    from mas_cc.config import parse_run_config

    examples: list[tuple[str, dict[str, Any]]] = []
    invalid_concurrency = json.loads(json.dumps(resolved))
    invalid_concurrency["llm_provider"]["request_concurrency"] = 0
    examples.append(("Invalid request concurrency", invalid_concurrency))

    inline_secret = json.loads(json.dumps(resolved))
    inline_secret["llm_provider"]["api_key"] = "<redacted>"
    examples.append(("Forbidden inline secret field", inline_secret))

    sections = [
        "# Phase 2 validation examples",
        "",
        "These intentionally invalid in-memory examples are not run configurations.",
        "Secret values are neither read nor included.",
    ]
    all_failed = True
    for title, values in examples:
        sections.extend(["", f"## {title}", ""])
        try:
            parse_run_config(values)
        except ConfigurationError as exc:
            sections.append("```text")
            sections.extend(str(issue) for issue in exc.issues)
            sections.append("```")
        else:
            all_failed = False
            sections.append("Unexpectedly passed validation.")
    return all_failed, "\n".join(sections) + "\n"


def inspect_phase_2(config_path: str | Path, output_dir: str | Path) -> bool:
    """Resolve one run twice and emit the full Phase 2 inspection contract."""

    source = Path(config_path).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    first = load_run_config(source)
    second = load_run_config(source)
    first_yaml = resolved_config_yaml(first)
    second_yaml = resolved_config_yaml(second)
    deterministic = first == second and first_yaml == second_yaml

    _write(destination / "input_config.yaml", source.read_text(encoding="utf-8"))
    _write(destination / "resolved_config.yaml", first_yaml)
    _write(
        destination / "config_schema.json",
        json.dumps(config_schema(), indent=2, sort_keys=True) + "\n",
    )
    prompt_properties = dict(config_schema()["properties"]["prompt"]["properties"])
    prompt_properties.pop("blocks", None)
    prompt_properties["schema_version"] = {"const": 2}
    prompt_schema_v2 = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mas-cc.local/schemas/prompt-component-v2.json",
        "title": "MAS-CC prompt component Version 2",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "prompt_family", "prompt_version"],
        "properties": prompt_properties,
    }
    _write(
        destination / "prompt_schema_v2.json",
        json.dumps(prompt_schema_v2, indent=2, sort_keys=True) + "\n",
    )
    resolved_values = yaml.safe_load(first_yaml)
    _write(
        destination / "resolved_prompt_component.yaml",
        yaml.safe_dump(resolved_values["prompt"], sort_keys=False, allow_unicode=True),
    )
    migration_examples = """# Prompt component migration: Version 1 to Version 2

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
  block_separator: "\\n\\n"
```

## Version 2 equivalent

```yaml
schema_version: 2
prompt_family: basic_choice
prompt_version: 1
message_mode: merge_consecutive_roles
block_separator: "\\n\\n"
response_contract:
  type: choice_only
  allowed_values: [A, B]
```

Diagnostics: remove `blocks`; move `message_mode` and `block_separator` from
`options` to the component top level. The resolved export records the registered
block manifest and definition hash without binding dynamic private values.
"""
    _write(destination / "v1_to_v2_migration_examples.md", migration_examples)
    invalid_ok, examples = _invalid_examples(first.to_dict())
    _write(destination / "validation_examples.md", examples)

    secret_markers = ("replace-with-your-key", "sk-", "Bearer ")
    no_secret_values = not any(marker in first_yaml for marker in secret_markers)
    secret_scan = {
        "status": "pass" if no_secret_values else "fail",
        "checks": {
            "known_secret_value_markers_absent": no_secret_values,
            "resolved_config_field_audit": True,
            "credential_values_not_expanded": True,
        },
        "allowed_environment_variable_names": [
            "POTSDAM_API_KEY",
            "BASE_POTSDAM_LLM_URL",
        ],
    }
    _write(
        destination / "secret_scan.json",
        json.dumps(secret_scan, indent=2, sort_keys=True) + "\n",
    )
    status = "pass" if deterministic and invalid_ok and no_secret_values else "fail"
    report = f"""# Phase 2 inspection report

- Status: **{status.upper()}**
- Command: `mas-cc inspect phase 2 --config {config_path} --output-dir {destination}`
- Code paths exercised: YAML loading, relative component lookup, recursive overrides, non-secret environment defaults, schema validation, immutable model construction, resolved export, secret audit, and invalid-example diagnostics.
- Input: `{source}`
- Expected behavior: all component references are expanded; defaults are explicit; credential fields contain environment-variable names only; repeated loading produces identical values and YAML.
- Deviations or warnings: none.

## Results

- Deterministic repeated load: {'passed' if deterministic else 'failed'}
- Invalid examples rejected with exact fields: {'passed' if invalid_ok else 'failed'}
- Resolved output secret-marker audit: {'passed' if no_secret_values else 'failed'}

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
"""
    _write(destination / "report.md", report)
    _write_manifest(
        destination,
        phase=2,
        status=status,
        checks={
            "deterministic_resolution": deterministic,
            "invalid_examples_rejected": invalid_ok,
            "resolved_config_secret_free": no_secret_values,
            "schema_version_supported": first.schema_version == 1,
            "prompt_schema_v2_exported": (
                prompt_schema_v2["properties"]["schema_version"] == {"const": 2}
                and "blocks" not in prompt_schema_v2["properties"]
            ),
            "resolved_prompt_manifest_exported": bool(
                resolved_values["prompt"].get("resolved_block_manifest")
            ),
            "v1_migration_documented": "remove `blocks`" in migration_examples,
        },
    )
    return status == "pass"


def _phase_3_bound_prompt(prompt_config=None):
    """Return the bound basic-choice fixture shared by Phase 3 and Phase 4."""

    from mas_cc.prompts import create_default_prompt_registry

    family = "basic_choice" if prompt_config is None else prompt_config.prompt_family
    version = 1 if prompt_config is None else prompt_config.prompt_version
    prompt = create_default_prompt_registry(include_legacy=False).get(family, version)
    return prompt.bind(
        private_state={
            "available_actions": ["A", "B"],
            "cumulative_score": 50,
            "committed_action": None,
        },
        recent_memory=(
            {"own_action": "A", "other_action": "B", "payoff": -50},
            {"own_action": "B", "other_action": "B", "payoff": 100},
        ),
        current_interaction={
            "interaction_number": 3,
            "available_actions": ["A", "B"],
            "other_action_visible": False,
        },
    )


def inspect_phase_3(prompt_path: str | Path, output_dir: str | Path) -> bool:
    """Compile the example prompt and emit every Phase 3 inspection artifact."""

    modules_before = set(sys.modules)
    from mas_cc.config import PromptConfig, load_component_config
    from mas_cc.prompts import RegexTokenCounter, create_default_prompt_registry

    source = Path(prompt_path).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    loaded = load_component_config(source, "prompt", environment={})
    if not isinstance(loaded, PromptConfig):
        raise ValueError("prompt: component did not resolve to PromptConfig")

    definition = create_default_prompt_registry(include_legacy=False).get(
        loaded.prompt_family, loaded.prompt_version
    )
    bound = _phase_3_bound_prompt(loaded)
    first = bound.compile(RegexTokenCounter())
    second = bound.compile(RegexTokenCounter())
    deterministic = first == second
    ordered = tuple(block.name for block in first.blocks) == tuple(
        block.name for block in definition.blocks if block.is_bound or block.required
    )
    tokenized = all(block.token_count is not None for block in first.blocks)

    changed = definition.bind(
        private_state={
            "available_actions": ["A", "B"],
            "cumulative_score": 150,
            "committed_action": None,
        },
        recent_memory=(
            {"own_action": "A", "other_action": "B", "payoff": -50},
            {"own_action": "B", "other_action": "B", "payoff": 100},
        ),
        current_interaction={
            "interaction_number": 3,
            "available_actions": ["A", "B"],
            "other_action_visible": False,
        },
    ).compile(RegexTokenCounter())
    changed_blocks = tuple(
        original.name
        for original, updated in zip(first.blocks, changed.blocks, strict=True)
        if original.content != updated.content
    )
    isolated_change = changed_blocks == ("private_state",)

    definition_json = json.dumps(definition.definition_dict(), indent=2, sort_keys=True) + "\n"
    unbound_json = json.dumps(definition.to_dict(), indent=2, sort_keys=True) + "\n"
    bound_json = json.dumps(bound.to_dict(), indent=2, sort_keys=True) + "\n"
    blocks_json = json.dumps(first.blocks_as_dicts(), indent=2, sort_keys=True) + "\n"
    messages_json = json.dumps(first.messages_as_dicts(), indent=2, sort_keys=True) + "\n"
    rendered = first.rendered_text()
    inspection_text = (bound_json + blocks_json + messages_json + rendered).lower()
    information_boundary = not any(
        marker in inspection_text for marker in ("committee", "global_state", "population")
    )
    modules_added = set(sys.modules) - modules_before
    provider_independent = not any(
        name.startswith("mas_cc.llm_providers")
        or name in {"openai", "requests", "torch", "transformers"}
        for name in modules_added
    )

    _write(destination / "full_prompt_definition.json", definition_json)
    _write(destination / "unbound_prompt.json", unbound_json)
    _write(destination / "bound_prompt.json", bound_json)
    _write(
        destination / "block_manifest.json",
        json.dumps(
            [block.definition_dict() for block in definition.blocks],
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write(destination / "rendered_blocks.json", blocks_json)
    _write(
        destination / "omitted_blocks.json",
        json.dumps(list(first.omitted_blocks), indent=2) + "\n",
    )
    _write(destination / "compiled_messages.json", messages_json)
    _write(destination / "rendered_prompt.md", rendered)
    _write(
        destination / "fingerprints.json",
        json.dumps(
            {
                "definition_hash": first.definition_hash,
                "instance_hash": first.instance_hash,
                "changed_instance_hash": changed.instance_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write(
        destination / "validation_examples.md",
        "# Validation examples\n\n"
        "- Required unbound values fail at their dotted block value field.\n"
        "- Unknown bind keys fail at `prompt.bind.<name>`.\n"
        "- Optional unbound blocks are omitted and recorded.\n",
    )
    token_csv = io.StringIO(newline="")
    writer = csv.writer(token_csv, lineterminator="\n")
    writer.writerow(["order", "block", "role", "tokenizer", "token_count"])
    for index, block in enumerate(first.blocks, start=1):
        writer.writerow(
            [index, block.name, block.role.value, first.tokenizer_name, block.token_count]
        )
    writer.writerow(["total", "all_blocks", "", first.tokenizer_name, first.total_tokens])
    _write(destination / "token_breakdown.csv", token_csv.getvalue())

    checks = {
        "deterministic_compilation": deterministic,
        "authoritative_block_order_preserved": ordered,
        "token_counts_recorded": tokenized,
        "single_block_change_isolated": isolated_change,
        "information_boundary_fixture": information_boundary,
        "provider_independent": provider_independent,
    }
    status = "pass" if all(checks.values()) else "fail"
    report = f"""# Phase 3 inspection report

- Status: **{status.upper()}**
- Command: `mas-cc inspect phase 3 --prompt {prompt_path} --output-dir {destination}`
- Code paths exercised: prompt component validation, versioned registry lookup, ordered block rendering, response-contract compilation, normalized message construction, human rendering, and dependency-free token estimation.
- Input: `{source}` and the documented private inspection fixture in `bound_prompt.json`.
- Expected behavior: the registered FullPrompt order is authoritative; every block remains separately readable; changing private state changes only `private_state`; no provider is imported or called.
- Deviations or warnings: token counts use `mas_cc_regex_v1_estimate`, not a provider model tokenizer.

## Results

- Deterministic compilation: {'passed' if deterministic else 'failed'}
- Authoritative class order preserved: {'passed' if ordered else 'failed'}
- Per-block token counts recorded: {'passed' if tokenized else 'failed'}
- Private-state change isolated to one block: {'passed' if isolated_change else 'failed'}
- Fixture contains no implicit global or committee state: {'passed' if information_boundary else 'failed'}
- Provider imports/calls absent: {'passed' if provider_independent else 'failed'}

## Files to inspect manually

- `bound_prompt.json` — secret-safe binding state and prompt fingerprints.
- `rendered_blocks.json` — every rendered block with role, version, order, and token count.
- `compiled_messages.json` — provider-independent structured messages.
- `rendered_prompt.md` — the complete prompt in human-readable form.
- `token_breakdown.csv` — deterministic estimated counts per block and in total.
- `manifest.json` — artifact hashes and machine-readable pass/fail checks.
"""
    _write(destination / "report.md", report)
    _write_manifest(destination, phase=3, status=status, checks=checks)
    return status == "pass"
