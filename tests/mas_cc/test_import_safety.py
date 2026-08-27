import json
import subprocess
import sys
from pathlib import Path


def test_import_mas_cc_has_no_heavy_or_external_side_effect_imports():
    script = """
import json
import sys
before = set(sys.modules)
import mas_cc
added = set(sys.modules) - before
print(json.dumps({
    "version": mas_cc.__version__,
    "forbidden": sorted(
        added & {"comet_ml", "dotenv", "openai", "requests", "torch", "transformers"}
    ),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    probe = json.loads(result.stdout)
    assert probe == {"version": "0.1.0", "forbidden": []}


def test_import_runtime_standalone_before_games_does_not_cycle():
    """mas_cc.runtime.loop_runtime is imported by mas_cc.games (via runner.py
    and naming_convention/runtime.py); importing mas_cc.runtime by itself
    first, before anything pulls in mas_cc.games, must not deadlock the
    partially-initialized module."""

    script = "import mas_cc.runtime\nprint('ok')"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "ok"


def test_import_prompt_kernel_does_not_import_games_or_providers():
    script = """
import json
import sys
import mas_cc.llm_runtime.prompts
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith('mas_cc.games') or name.startswith('mas_cc.llm_runtime.providers')
)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []


def test_production_analysis_never_imports_legacy_relational_theory():
    root = Path("src/mas_cc")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from .theory" in text or "imitation_round_feedback.theory import" in text:
            offenders.append(str(path))
    assert offenders == []
