"""CLI handlers for provider-backed diagnostic probes.

A probe is not a game and not an experiment: it has no population loop, so it
gets its own command rather than a `Game` type that would have to no-op most of
the interface.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import yaml

from mas_cc.probes.controller_retention.config import load_probe_config
from mas_cc.probes.controller_retention.runner import run_probe


def run_controller_retention_probe(
    config_path: Path,
    output_dir: Path | None = None,
    *,
    mode: str = "run",
    stream: Any = sys.stderr,
) -> tuple[bool, Path, str]:
    """``preflight`` sends nothing, ``analyze`` re-reads a finished run, ``run``
    does the whole thing.  All three write the same artifact tree."""

    config = load_probe_config(config_path)
    result = run_probe(
        config,
        output_dir=output_dir,
        preflight_only=mode == "preflight",
        analyze_only=mode == "analyze",
        stream=stream,
    )
    if mode == "preflight":
        return (
            result.preflight.passed,
            result.output_dir,
            "Probe preflight " + ("passed" if result.preflight.passed else "FAILED"),
        )
    if mode == "run" and not result.completed_successfully:
        return (
            False,
            result.report_path or result.output_dir,
            "Controller-retention probe finished with missing or invalid calls",
        )
    return (
        result.preflight.passed,
        result.report_path or result.output_dir,
        "Controller-retention probe report written",
    )


def run_configured_probe(
    config_path: Path,
    output_dir: Path | None = None,
    *,
    mode: str = "run",
    approve_preflight: Path | None = None,
    stream: Any = sys.stderr,
) -> tuple[bool, Path, str]:
    """Dispatch one probe config while preserving the legacy default."""

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    probe_name = (
        raw.get("probe", "controller_retention")
        if isinstance(raw, dict)
        else "controller_retention"
    )
    if probe_name == "musr_truthful_selective":
        from mas_cc.probes.musr_truthful_selective import (
            analyze,
            load_config,
            prepare,
            run,
        )

        config = load_config(config_path)
        if mode == "preflight":
            root, payload = prepare(config, output_dir)
            return (
                bool(payload["passed"]),
                root / "preflight/report.md",
                "MuSR truthful-selective preflight written",
            )
        if mode == "analyze":
            result = analyze(config, output_dir)
            return (
                True,
                Path(result["report"]),
                "MuSR truthful-selective report written",
            )
        result = __import__("asyncio").run(
            run(config, output_dir, approve_preflight=approve_preflight)
        )
        ok = result["execution"]["successful"] == result["execution"]["scheduled"]
        return (
            ok,
            Path(result["report"]),
            "MuSR truthful-selective calibration completed",
        )
    if probe_name != "musr_local_evidence":
        if probe_name == "musr_blackboard_prompt_validation":
            from mas_cc.probes.musr_blackboard_prompt_validation import (
                analyze,
                load_config,
                prepare,
                run,
            )

            config = load_config(config_path)
            if mode == "preflight":
                root, payload = prepare(config, output_dir)
                return (
                    bool(payload["passed"]),
                    root / f"preflight/{config.mode}_report.md",
                    f"MuSR blackboard {config.mode} preflight written",
                )
            if mode == "analyze":
                result = analyze(config, output_dir)
                return (
                    True,
                    Path(result["report"]),
                    "MuSR blackboard validation report written",
                )
            result = __import__("asyncio").run(
                run(config, output_dir, approve_preflight=approve_preflight)
            )
            ok = result["execution"]["successful"] == result["execution"]["scheduled"]
            return ok, Path(result["report"]), "MuSR blackboard validation completed"
        if probe_name == "musr_symbolic_ambiguity_replication":
            from mas_cc.probes.musr_symbolic_ambiguity_replication import (
                analyze,
                load_config,
                prepare,
                run,
            )

            config = load_config(config_path)
            if mode == "preflight":
                root, payload = prepare(config, output_dir)
                return (
                    bool(payload["passed"]),
                    root / "preflight/report.md",
                    "MuSR symbolic-ambiguity replication preflight written",
                )
            if mode == "analyze":
                result = analyze(config, output_dir)
                return (
                    True,
                    Path(result["report"]),
                    "MuSR symbolic-ambiguity replication report written",
                )
            result = __import__("asyncio").run(
                run(config, output_dir, approve_preflight=approve_preflight)
            )
            ok = (
                result["execution"].get("completed") == result["execution"]["scheduled"]
            )
            destination = (
                Path(result["report"])
                if result.get("report")
                else Path(result["output"])
            )
            return ok, destination, "MuSR symbolic-ambiguity replication completed"
        if probe_name == "musr_symbolic_ambiguity":
            from mas_cc.probes.musr_symbolic_ambiguity import (
                analyze,
                load_config,
                prepare,
                run,
            )

            config = load_config(config_path)
            if mode == "preflight":
                root, payload = prepare(config, output_dir)
                return (
                    bool(payload["passed"]),
                    root / "preflight/report.md",
                    "MuSR symbolic-ambiguity preflight written",
                )
            if mode == "analyze":
                result = analyze(config, output_dir)
                return (
                    True,
                    Path(result["report"]),
                    "MuSR symbolic-ambiguity report written",
                )
            result = __import__("asyncio").run(
                run(config, output_dir, approve_preflight=approve_preflight)
            )
            ok = result["execution"]["successful"] == result["execution"]["scheduled"]
            return (
                ok,
                Path(result["report"]),
                "MuSR symbolic-ambiguity calibration completed",
            )
        if probe_name == "musr_private_redistribution":
            from mas_cc.probes.musr_private_redistribution import (
                analyze,
                load_config,
                prepare,
                run,
            )

            config = load_config(config_path)
            if mode == "preflight":
                root, _, _, _, _ = prepare(config, output_dir)
                payload = yaml.safe_load(
                    (root / "preflight/preflight.json").read_text(encoding="utf-8")
                )
                return (
                    bool(payload["passed"]),
                    root / "preflight/report.md",
                    "MuSR private-redistribution preflight written",
                )
            if mode == "analyze":
                result = analyze(config, output_dir)
                return (
                    True,
                    Path(result["report"]),
                    "MuSR private-redistribution report written",
                )
            result = __import__("asyncio").run(
                run(config, output_dir, approve_preflight=approve_preflight)
            )
            ok = result["execution"]["successful"] == result["execution"]["scheduled"]
            return (
                ok,
                Path(result["report"]),
                "MuSR private-redistribution calibration completed",
            )
        if probe_name == "musr_prompt_solvability":
            from mas_cc.probes.musr_prompt_solvability import (
                analyze,
                load_config,
                prepare,
                run,
            )

            config = load_config(config_path)
            if mode == "preflight":
                root, _, payload = prepare(config, output_dir)
                return (
                    payload["passed"],
                    root / "preflight/report.md",
                    "MuSR prompt-solvability preflight written",
                )
            if mode == "analyze":
                result = analyze(config, output_dir)
                return (
                    True,
                    Path(result["report"]),
                    "MuSR prompt-solvability report written",
                )
            result = __import__("asyncio").run(
                run(config, output_dir, approve_preflight=approve_preflight)
            )
            ok = result["execution"]["successful"] == result["execution"]["scheduled"]
            return (
                ok,
                Path(result["report"]),
                "MuSR prompt-solvability calibration completed",
            )
        return run_controller_retention_probe(
            config_path, output_dir, mode=mode, stream=stream
        )
    from mas_cc.probes.musr_local_evidence.config import load_probe_config as load_local
    from mas_cc.probes.musr_local_evidence.runner import analyze, prepare, run

    config = load_local(config_path)
    if mode == "preflight":
        root, plan, payload = prepare(config, output_dir)
        return (
            payload["passed"],
            root / "preflight/report.md",
            "MuSR local-evidence preflight written",
        )
    if mode == "analyze":
        from mas_cc.probes.musr_local_evidence.preflight import build_plan

        root = Path(output_dir or config.output_dir)
        plan = build_plan(config)
        result = analyze(root, plan)
        return True, Path(result["report"]), "MuSR local-evidence report written"
    result = __import__("asyncio").run(
        run(config, output_dir, approve_preflight=approve_preflight)
    )
    ok = result["execution"]["successful"] == config.nominal_calls
    return ok, Path(result["report"]), "MuSR local-evidence probe completed"


__all__ = ["run_configured_probe", "run_controller_retention_probe"]
