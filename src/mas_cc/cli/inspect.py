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
    invalid_ok, examples = _invalid_examples(first.to_dict())
    _write(destination / "validation_examples.md", examples)

    secret_markers = ("replace-with-your-key", "sk-", "Bearer ")
    no_secret_values = not any(marker in first_yaml for marker in secret_markers)
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
- `config_schema.json` — machine-readable schema version 1.
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
        },
    )
    return status == "pass"


def _phase_3_context():
    from mas_cc.prompts import PromptContext

    return PromptContext(
        task_description=(
            "Coordinate with another player by choosing one of the two available actions."
        ),
        game_rules=(
            "Choose exactly one action on every interaction.",
            "Both players receive a positive payoff when their actions match.",
            "Both players receive a negative payoff when their actions differ.",
            "The other player's current choice is not visible before you decide.",
        ),
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
        decision_instruction="Select the action that you will play in this interaction.",
        metadata={"fixture": "phase_03_inspection_v1"},
    )


def inspect_phase_3(prompt_path: str | Path, output_dir: str | Path) -> bool:
    """Compile the example prompt and emit every Phase 3 inspection artifact."""

    modules_before = set(sys.modules)
    from dataclasses import replace

    from mas_cc.config import PromptConfig, load_component_config
    from mas_cc.prompts import (
        PromptComposer,
        RegexTokenCounter,
        create_default_prompt_registry,
    )

    source = Path(prompt_path).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    loaded = load_component_config(source, "prompt", environment={})
    if not isinstance(loaded, PromptConfig):
        raise ValueError("prompt: component did not resolve to PromptConfig")

    context = _phase_3_context()
    composer = PromptComposer(create_default_prompt_registry(), RegexTokenCounter())
    first = composer.compose(loaded, context)
    second = composer.compose(loaded, context)
    deterministic = first == second
    ordered = tuple(block.name for block in first.blocks) == loaded.blocks
    tokenized = all(block.token_count is not None for block in first.blocks)

    changed_context = replace(
        context,
        private_state={**context.to_dict()["private_state"], "cumulative_score": 150},
    )
    changed = composer.compose(loaded, changed_context)
    changed_blocks = tuple(
        original.name
        for original, updated in zip(first.blocks, changed.blocks, strict=True)
        if original.content != updated.content
    )
    isolated_change = changed_blocks == ("private_state",)

    context_json = json.dumps(context.to_dict(), indent=2, sort_keys=True) + "\n"
    blocks_json = json.dumps(first.blocks_as_dicts(), indent=2, sort_keys=True) + "\n"
    messages_json = json.dumps(first.messages_as_dicts(), indent=2, sort_keys=True) + "\n"
    rendered = first.rendered_text()
    inspection_text = (context_json + blocks_json + messages_json + rendered).lower()
    information_boundary = not any(
        marker in inspection_text for marker in ("committee", "global_state", "population")
    )
    modules_added = set(sys.modules) - modules_before
    provider_independent = not any(
        name.startswith("mas_cc.llm_providers")
        or name in {"openai", "requests", "torch", "transformers"}
        for name in modules_added
    )

    _write(destination / "prompt_context.json", context_json)
    _write(destination / "prompt_blocks.json", blocks_json)
    _write(destination / "compiled_messages.json", messages_json)
    _write(destination / "rendered_prompt.md", rendered)

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
        "yaml_block_order_preserved": ordered,
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
- Input: `{source}` and the documented private inspection fixture in `prompt_context.json`.
- Expected behavior: the seven YAML blocks compile in order; every block remains separately readable; changing private state changes only `private_state`; no provider is imported or called.
- Deviations or warnings: token counts use `mas_cc_regex_v1_estimate`, not a provider model tokenizer.

## Results

- Deterministic compilation: {'passed' if deterministic else 'failed'}
- YAML order preserved: {'passed' if ordered else 'failed'}
- Per-block token counts recorded: {'passed' if tokenized else 'failed'}
- Private-state change isolated to one block: {'passed' if isolated_change else 'failed'}
- Fixture contains no implicit global or committee state: {'passed' if information_boundary else 'failed'}
- Provider imports/calls absent: {'passed' if provider_independent else 'failed'}

## Files to inspect manually

- `prompt_context.json` — the exact information available to the example agent.
- `prompt_blocks.json` — every rendered block with role, version, order, and token count.
- `compiled_messages.json` — provider-independent structured messages.
- `rendered_prompt.md` — the complete prompt in human-readable form.
- `token_breakdown.csv` — deterministic estimated counts per block and in total.
- `manifest.json` — artifact hashes and machine-readable pass/fail checks.
"""
    _write(destination / "report.md", report)
    _write_manifest(destination, phase=3, status=status, checks=checks)
    return status == "pass"
