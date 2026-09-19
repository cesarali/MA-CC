"""Run the local finalizer on a study, publishing into a side directory.

Used to gate performance changes: finalize a study with the candidate code into
a scratch directory, then compare every published table against the package the
reference code produced (compare_dirs.py). Everything sits under a main guard
because the finalizer's spawn pools re-import __main__.

usage: finalize_to.py STUDY_DIR OUTPUT_DIR
"""
import json
import sys
import time
from pathlib import Path


def main() -> int:
    from mas_cc.studies.aggregation import _aggregate_study_local

    study, out = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.time()
    result = _aggregate_study_local(study, analysis_output_dir=out)
    print(json.dumps({"seconds": round(time.time() - started, 1), "complete": result.get("complete"), "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
