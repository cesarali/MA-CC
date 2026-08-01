import json
import subprocess
import sys


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
