from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NERSC = ROOT / "scripts" / "nersc"


def _run(*arguments: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(argument) for argument in arguments],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_prepared_study(study: Path, *, shard_count: int) -> list[Path]:
    study.mkdir()
    outputs = [study / "runs" / str(index) for index in range(shard_count)]
    with (study / "execution_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "array_index",
                "config_index",
                "config_path",
                "cell_index",
                "cell_id",
                "output_dir",
            )
        )
        for index, output in enumerate(outputs):
            writer.writerow(
                (index, 0, "/config.yaml", index, f"cell-{index:04d}", output)
            )
    (study / "preparation.json").write_text(
        json.dumps(
            {
                "status": "prepared",
                "execution_site": "nersc",
                "array": f"0-{shard_count - 1}%{shard_count}",
            }
        ),
        encoding="utf-8",
    )
    (study / "execution_plan.json").write_text(
        json.dumps(
            {
                "shard_count": shard_count,
                "array_throttle": shard_count,
                "cpus_per_task": 8,
                "memory": "8G",
                "time_limit": "04:00:00",
                "partition": "all",
                "qos": "normal",
            }
        ),
        encoding="utf-8",
    )
    return outputs


def _write_fake_scheduler(
    tmp_path: Path, *, first_exit: int, delay_seconds: float = 0
) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    counter = tmp_path / "salloc-count"
    fake_salloc = fake_bin / "salloc"
    fake_salloc.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
count=0
if [[ -f \"${{FAKE_COUNTER_FILE}}\" ]]; then
  read -r count <\"${{FAKE_COUNTER_FILE}}\"
fi
count=$((count + 1))
printf '%s\\n' \"${{count}}\" >\"${{FAKE_COUNTER_FILE}}\"
sleep {delay_seconds}
index=$((count - 1))
printf -v cell_id 'cell-%04d' \"${{index}}\"
destination=\"${{FAKE_STUDY_DIR}}/runs/${{index}}/nested/cells/${{cell_id}}\"
mkdir -p \"${{destination}}\"
printf '{{\"status\":\"completed\",\"episode_row_counts\":{{\"%s-0000\":1}}}}\\n' \\
  \"${{cell_id}}\" >\"${{destination}}/cell_complete.json\"
if (( count == 1 )); then
  exit {first_exit}
fi
""",
        encoding="utf-8",
    )
    fake_salloc.chmod(0o755)
    fake_squeue = fake_bin / "squeue"
    fake_squeue.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_squeue.chmod(0o755)
    return fake_bin, counter


def test_nersc_shell_scripts_are_valid_and_executable():
    scripts = sorted(NERSC.glob("*.sh"))
    assert scripts
    for script in scripts:
        assert os.access(script, os.X_OK), script
        completed = _run("bash", "-n", script)
        assert completed.returncode == 0, completed.stderr


def test_allocation_dry_run_hard_codes_interactive_cpu_policy():
    completed = _run(
        NERSC / "allocate_cpu.sh",
        "--account",
        "m1234",
        "--nodes",
        "4",
        "--time",
        "04:00:00",
        "--dry-run",
    )
    assert completed.returncode == 0, completed.stderr
    assert "--qos=interactive" in completed.stdout
    assert "--constraint=cpu" in completed.stdout
    assert "--nodes=4" in completed.stdout
    assert "--time=04:00:00" in completed.stdout
    assert "sbatch" not in completed.stdout


def test_nersc_launchers_reject_qos_override_and_resource_overflow():
    qos = _run(NERSC / "allocate_cpu.sh", "--account", "m1234", "--qos=regular")
    assert qos.returncode == 2
    assert "fixed to interactive" in qos.stderr

    nodes = _run(
        NERSC / "allocate_cpu.sh", "--account", "m1234", "--nodes", "5", "--dry-run"
    )
    assert nodes.returncode == 2
    assert "at most 4 nodes" in nodes.stderr

    walltime = _run(
        NERSC / "allocate_cpu.sh",
        "--account",
        "m1234",
        "--time",
        "04:00:01",
        "--dry-run",
    )
    assert walltime.returncode == 2
    assert "cannot exceed 04:00:00" in walltime.stderr


def test_single_command_dry_run_uses_salloc_then_srun():
    completed = _run(
        NERSC / "run_command.sh",
        "--account",
        "m1234",
        "--dry-run",
        "--",
        "python",
        "-V",
    )
    assert completed.returncode == 0, completed.stderr
    words = completed.stdout.split()
    assert words[0] == "salloc"
    assert "--qos=interactive" in words
    assert "--constraint=cpu" in words
    assert "srun" in words
    assert "--cpus-per-task=256" in words


def test_prepared_study_plan_caps_workers_by_science_and_node_resources(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    manifest = study / "execution_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("array_index", "config_index", "config_path", "cell_index", "cell_id", "output_dir")
        )
        for index in range(40):
            writer.writerow(
                (
                    index,
                    0,
                    "/config.yaml",
                    index,
                    f"cell-{index:04d}",
                    study / "runs" / str(index),
                )
            )
    (study / "preparation.json").write_text(
        json.dumps(
            {"status": "prepared", "execution_site": "nersc", "array": "0-39%18"}
        ),
        encoding="utf-8",
    )
    (study / "execution_plan.json").write_text(
        json.dumps(
            {
                "shard_count": 40,
                "array_throttle": 18,
                "cpus_per_task": 8,
                "memory": "8G",
                "time_limit": "04:00:00",
                "partition": "all",
                "qos": "normal",
            }
        ),
        encoding="utf-8",
    )

    completed = _run(
        sys.executable,
        NERSC / "study_plan.py",
        "--study-dir",
        study,
        "--nodes",
        "4",
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["nersc_qos"] == "interactive"
    assert plan["nersc_constraint"] == "cpu"
    assert plan["source_qos"] == "normal"
    assert plan["workers_per_node_ceiling"] == 16
    assert plan["total_workers"] == 18

    (study / "preparation.json").write_text(
        json.dumps(
            {"status": "prepared", "execution_site": "potsdam", "array": "0-39%18"}
        ),
        encoding="utf-8",
    )
    rejected = _run(
        sys.executable,
        NERSC / "study_plan.py",
        "--study-dir",
        study,
        "--nodes",
        "4",
    )
    assert rejected.returncode != 0
    assert "not prepared for the NERSC scheduler adapter" in rejected.stderr


def test_study_progress_requires_every_seal_and_reports_failed_resume(tmp_path):
    study = tmp_path / "study"
    outputs = [study / "runs" / f"cell-{index:04d}" for index in range(2)]
    for output in outputs:
        output.mkdir(parents=True)
    with (study / "execution_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("array_index", "config_index", "config_path", "cell_index", "cell_id", "output_dir")
        )
        for index, output in enumerate(outputs):
            writer.writerow((index, 0, "/config.yaml", index, f"cell-{index:04d}", output))
    completed_cell = outputs[0] / "nested" / "cells" / "cell-0000"
    completed_cell.mkdir(parents=True)
    (completed_cell / "cell_complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "episode_row_counts": {"cell-0000-0000": 24},
            }
        ),
        encoding="utf-8",
    )
    failed = outputs[1] / "nested" / "cells" / "cell-0001" / ".resume" / "cell-0001-0000"
    failed.mkdir(parents=True)
    (failed / "manifest.json").write_text(
        json.dumps({"episode_id": "cell-0001-0000", "status": "failed"}),
        encoding="utf-8",
    )

    completed = _run(
        sys.executable,
        NERSC / "study_progress.py",
        "--study-dir",
        study,
    )
    assert completed.returncode == 0, completed.stderr
    progress = json.loads(completed.stdout)
    assert progress["expected_cells"] == 2
    assert progress["sealed_cells"] == 1
    assert progress["completed_episodes"] == 1
    assert progress["failed_episodes"] == 1
    assert progress["complete"] is False


def test_study_progress_counts_completed_resume_before_cell_seal(tmp_path):
    study = tmp_path / "study"
    (output,) = _write_prepared_study(study, shard_count=1)
    resume = output / "nested" / "cells" / "cell-0000" / ".resume" / "cell-0000-0000"
    resume.mkdir(parents=True)
    (resume / "manifest.json").write_text(
        json.dumps({"episode_id": "cell-0000-0000", "status": "completed"}),
        encoding="utf-8",
    )

    completed = _run(
        sys.executable,
        NERSC / "study_progress.py",
        "--study-dir",
        study,
    )
    assert completed.returncode == 0, completed.stderr
    progress = json.loads(completed.stdout)
    assert progress["expected_cells"] == 1
    assert progress["sealed_cells"] == 0
    assert progress["completed_episodes"] == 1
    assert progress["failed_episodes"] == 0
    assert progress["complete"] is False


def test_allocation_boundary_clears_only_orphaned_provider_leases(tmp_path):
    study = tmp_path / "study"
    _write_prepared_study(study, shard_count=1)
    control = study / "runtime" / "provider-control"
    control.mkdir(parents=True)
    state = {
        "schema_version": 1,
        "limit": 3,
        "leases": {
            "old-a": {"worker": "old:1", "expires_at": 9999999999.0},
            "old-b": {"worker": "old:2", "expires_at": 9999999999.0},
        },
        "events": [{"at": 123.0, "success": True}],
        "global_pause_until": 456.0,
        "updated_at": 789.0,
    }
    (control / "state.json").write_text(json.dumps(state), encoding="utf-8")

    completed = _run(
        sys.executable,
        NERSC / "reset_provider_control_leases.py",
        "--study-dir",
        study,
    )
    assert completed.returncode == 0, completed.stderr
    assert "cleared_orphaned_provider_leases=2" in completed.stdout
    reset = json.loads((control / "state.json").read_text(encoding="utf-8"))
    assert reset["leases"] == {}
    assert reset["limit"] == 3
    assert reset["events"] == state["events"]
    assert reset["global_pause_until"] == 456.0
    assert reset["updated_at"] == 789.0
    assert reset["last_allocation_lease_reset_count"] == 2
    assert reset["last_allocation_lease_reset_at"] > 0


def test_aggregation_progress_requires_valid_manifest_and_archive(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    pending = _run(
        sys.executable,
        NERSC / "aggregation_progress.py",
        "--study-dir",
        study,
    )
    assert pending.returncode == 0, pending.stderr
    assert json.loads(pending.stdout)["status"] == "pending"

    analysis = study / "analysis"
    analysis.mkdir()
    (analysis / "validation.json").write_text(
        json.dumps({"valid": False, "errors": ["missing episode"]}),
        encoding="utf-8",
    )
    failed = _run(
        sys.executable,
        NERSC / "aggregation_progress.py",
        "--study-dir",
        study,
    )
    assert failed.returncode == 0, failed.stderr
    assert json.loads(failed.stdout)["status"] == "failed"

    (analysis / "validation.json").write_text(
        json.dumps({"valid": True}), encoding="utf-8"
    )
    (analysis / "analysis_manifest.json").write_text(
        json.dumps({"study_id": "test-study", "status": "complete"}),
        encoding="utf-8",
    )
    archive = analysis / "test-study_analysis.zip"
    archive.write_bytes(b"zip")
    complete = _run(
        sys.executable,
        NERSC / "aggregation_progress.py",
        "--study-dir",
        study,
    )
    assert complete.returncode == 0, complete.stderr
    progress = json.loads(complete.stdout)
    assert progress["status"] == "complete"
    assert progress["detail"] == str(archive)


def test_detached_supervisor_dry_run_is_idempotent_and_interactive_only(tmp_path):
    if os.environ.get("NERSC_HOST") != "perlmutter":
        pytest.skip("NERSC Conda environment is only available on Perlmutter")
    study = tmp_path / "study"
    _write_prepared_study(study, shard_count=1)
    environment = os.environ.copy()
    environment["NERSC_RESULTS_ROOT"] = str(tmp_path)

    completed = _run(
        NERSC / "start_study_supervisor.sh",
        "--account",
        "m1234",
        "--nodes",
        "1",
        "--study-dir",
        study,
        "--time",
        "02:00:00",
        "--aggregate",
        "--ensure",
        "--dry-run",
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert "nohup setsid" in completed.stdout
    assert "run_study_until_complete.sh" in completed.stdout
    assert "--time 02:00:00" in completed.stdout
    assert "--aggregate" in completed.stdout
    assert "regular" not in completed.stdout
    assert "sbatch" not in completed.stdout


def test_rollover_rechecks_progress_and_requests_fresh_allocation(tmp_path):
    if os.environ.get("NERSC_HOST") != "perlmutter":
        pytest.skip("NERSC module and Conda environment are only available on Perlmutter")
    study = tmp_path / "study"
    _write_prepared_study(study, shard_count=2)
    fake_bin, counter = _write_fake_scheduler(tmp_path, first_exit=124)

    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_COUNTER_FILE": str(counter),
            "FAKE_STUDY_DIR": str(study),
            "NERSC_RESULTS_ROOT": str(tmp_path),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    completed = _run(
        NERSC / "run_study_until_complete.sh",
        "--account",
        "m1234",
        "--nodes",
        "1",
        "--study-dir",
        study,
        "--time",
        "02:00:00",
        "--retry-delay",
        "0",
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert counter.read_text(encoding="utf-8").strip() == "2"
    assert "requesting interactive CPU allocation 1" in completed.stdout
    assert "allocation=1 exit=124 cells=1/2" in completed.stdout
    assert "requesting interactive CPU allocation 2" in completed.stdout
    assert "all 2 cell seals are complete" in completed.stdout


def test_direct_study_launcher_requires_explicit_opt_in_for_shorter_walltime(tmp_path):
    if os.environ.get("NERSC_HOST") != "perlmutter":
        pytest.skip("NERSC module and Conda environment are only available on Perlmutter")
    study = tmp_path / "study"
    _write_prepared_study(study, shard_count=1)
    environment = os.environ.copy()
    environment["NERSC_RESULTS_ROOT"] = str(tmp_path)
    command = (
        NERSC / "run_study.sh",
        "--account",
        "m1234",
        "--nodes",
        "1",
        "--study-dir",
        study,
        "--time",
        "02:00:00",
        "--dry-run",
    )

    rejected = _run(*command, env=environment)
    assert rejected.returncode == 2
    assert "shorter than planned shard time" in rejected.stderr

    allowed = _run(*command, "--allow-shorter-than-plan", env=environment)
    assert allowed.returncode == 0, allowed.stderr
    assert "--time=02:00:00" in allowed.stdout
    assert "resumable allocation time 02:00:00" in allowed.stderr


def test_detached_supervisor_survives_launcher_exit_and_ensure_is_idempotent(tmp_path):
    if os.environ.get("NERSC_HOST") != "perlmutter":
        pytest.skip("NERSC module and Conda environment are only available on Perlmutter")
    study = tmp_path / "study"
    _write_prepared_study(study, shard_count=1)
    fake_bin, counter = _write_fake_scheduler(
        tmp_path, first_exit=0, delay_seconds=1.5
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_COUNTER_FILE": str(counter),
            "FAKE_STUDY_DIR": str(study),
            "NERSC_RESULTS_ROOT": str(tmp_path),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    command = (
        NERSC / "start_study_supervisor.sh",
        "--account",
        "m1234",
        "--nodes",
        "1",
        "--study-dir",
        study,
        "--retry-delay",
        "0",
    )
    started = _run(*command, env=environment)
    assert started.returncode == 0, started.stderr
    assert "started pid=" in started.stdout

    ensured = _run(*command, "--ensure", env=environment)
    assert ensured.returncode == 0, ensured.stderr
    assert "supervisor already active" in ensured.stdout

    log = study / "runtime" / "nersc-rollover.log"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if log.is_file() and "all 1 cell seals are complete" in log.read_text(
            encoding="utf-8"
        ):
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"detached supervisor did not complete; log={log.read_text()}")
    assert counter.read_text(encoding="utf-8").strip() == "1"


def test_rollover_retries_aggregation_allocation_until_archive_exists(tmp_path):
    if os.environ.get("NERSC_HOST") != "perlmutter":
        pytest.skip("NERSC module and Conda environment are only available on Perlmutter")
    study = tmp_path / "study"
    (output,) = _write_prepared_study(study, shard_count=1)
    cell = output / "nested" / "cells" / "cell-0000"
    cell.mkdir(parents=True)
    (cell / "cell_complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "episode_row_counts": {"cell-0000-0000": 1},
            }
        ),
        encoding="utf-8",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    counter = tmp_path / "aggregation-count"
    fake_salloc = fake_bin / "salloc"
    fake_salloc.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
count=0
if [[ -f "${FAKE_COUNTER_FILE}" ]]; then
  read -r count <"${FAKE_COUNTER_FILE}"
fi
count=$((count + 1))
printf '%s\\n' "${count}" >"${FAKE_COUNTER_FILE}"
if (( count == 1 )); then
  exit 124
fi
mkdir -p "${FAKE_STUDY_DIR}/analysis"
printf '{"valid":true}\\n' >"${FAKE_STUDY_DIR}/analysis/validation.json"
printf '{"study_id":"test-study","status":"complete"}\\n' \
  >"${FAKE_STUDY_DIR}/analysis/analysis_manifest.json"
printf 'zip\\n' >"${FAKE_STUDY_DIR}/analysis/test-study_analysis.zip"
""",
        encoding="utf-8",
    )
    fake_salloc.chmod(0o755)
    fake_squeue = fake_bin / "squeue"
    fake_squeue.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_squeue.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_COUNTER_FILE": str(counter),
            "FAKE_STUDY_DIR": str(study),
            "NERSC_RESULTS_ROOT": str(tmp_path),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    completed = _run(
        NERSC / "run_study_until_complete.sh",
        "--account",
        "m1234",
        "--nodes",
        "1",
        "--study-dir",
        study,
        "--retry-delay",
        "0",
        "--aggregate",
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert counter.read_text(encoding="utf-8").strip() == "2"
    assert "aggregation attempt=1 exit=124 status=pending" in completed.stdout
    assert "strict aggregation complete:" in completed.stdout


def test_study_launcher_dry_run_ignores_source_qos_and_uses_interactive(tmp_path):
    if os.environ.get("NERSC_HOST") != "perlmutter":
        pytest.skip("NERSC module and Conda environment are only available on Perlmutter")
    study = tmp_path / "study"
    study.mkdir()
    with (study / "execution_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "array_index",
                "config_index",
                "config_path",
                "cell_index",
                "cell_id",
                "output_dir",
            )
        )
        for index in range(4):
            writer.writerow(
                (
                    index,
                    0,
                    "/config.yaml",
                    index,
                    f"cell-{index:04d}",
                    study / "runs" / str(index),
                )
            )
    (study / "preparation.json").write_text(
        json.dumps(
            {"status": "prepared", "execution_site": "nersc", "array": "0-3%4"}
        ),
        encoding="utf-8",
    )
    (study / "execution_plan.json").write_text(
        json.dumps(
            {
                "shard_count": 4,
                "array_throttle": 4,
                "cpus_per_task": 8,
                "memory": "8G",
                "time_limit": "00:10:00",
                "partition": "all",
                "qos": "normal",
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["NERSC_RESULTS_ROOT"] = str(tmp_path)
    completed = _run(
        NERSC / "run_study.sh",
        "--account",
        "m1234",
        "--nodes",
        "1",
        "--study-dir",
        study,
        "--dry-run",
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--qos=interactive" in completed.stdout
    assert "--constraint=cpu" in completed.stdout
    assert "--qos=normal" not in completed.stdout
    assert "srun" in completed.stdout
    assert "run_study_rank.py" in completed.stdout


def test_rank_worker_refuses_regular_qos_before_reading_manifest(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_QOS": "regular",
            "SLURM_JOB_CONSTRAINTS": "cpu",
            "SLURM_PROCID": "0",
            "SLURM_NTASKS": "1",
        }
    )
    completed = _run(
        sys.executable,
        NERSC / "run_study_rank.py",
        "--study-dir",
        tmp_path,
        "--manifest",
        tmp_path / "missing.csv",
        "--worker-kind",
        "cell",
        "--total-workers",
        "1",
        env=environment,
    )
    assert completed.returncode == 2
    assert "refusing non-interactive NERSC QoS: regular" in completed.stderr


def test_nersc_executable_scripts_never_call_sbatch():
    for script in sorted((*NERSC.glob("*.sh"), *NERSC.glob("*.py"))):
        text = script.read_text(encoding="utf-8")
        assert "sbatch" not in text, script
        assert "scripts/Potsdam" not in text, script


def test_generic_potsdam_launchers_have_no_nersc_scheduler_or_path_dependency():
    launchers = (
        ROOT / "scripts/Potsdam/SLURM/run_config_array.job",
        ROOT / "scripts/Potsdam/SLURM/run_study_cell_array.job",
    )
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "MAS_CC_EXECUTION_SITE=potsdam" in text
        assert "/pscratch/" not in text
        assert "salloc" not in text
        assert "scripts/nersc" not in text
