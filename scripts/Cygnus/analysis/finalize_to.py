"""Run the local finalizer on a study, publishing into a side directory.

Used to gate performance changes: finalize a study with the candidate code into
a scratch directory, then compare every published table against the package the
reference code produced (compare_dirs.py). Everything sits under a main guard
because the finalizer's spawn pools re-import __main__.

usage: finalize_to.py STUDY_DIR OUTPUT_DIR [--allow-incomplete]
"""
import json
import resource
import sys
import time
from pathlib import Path


def main() -> int:
    from mas_cc.studies.aggregation import _aggregate_study_local

    study, out = Path(sys.argv[1]), Path(sys.argv[2])
    allow_incomplete = "--allow-incomplete" in sys.argv[3:]
    started = time.time()
    result = _aggregate_study_local(study, analysis_output_dir=out, allow_incomplete=allow_incomplete)
    # ru_maxrss is KiB on Linux; children = the spawn pool workers (max over any one child).
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(json.dumps({"seconds": round(time.time() - started, 1), "complete": result.get("complete"), "output": str(out),
                      "peak_rss_mib_self": round(own), "peak_rss_mib_largest_child": round(child)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
