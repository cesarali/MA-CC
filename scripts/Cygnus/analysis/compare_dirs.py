"""Compare two published analysis packages table by table.

usage: compare_dirs.py REFERENCE_TABLES CANDIDATE_TABLES [--ignore col,col]

Every Parquet file in REFERENCE must exist in CANDIDATE with an equal frame
(pandas equality, NaN == NaN). Columns named in --ignore (defaults cover the
run-time provenance stamps) are dropped before comparing. Prints one line per
table and a final RESULT line; exit status 1 on any difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_IGNORE = {"created_at", "generated_at", "published_at", "analysis_created_at", "timestamp"}


def main(argv: list[str]) -> int:
    ref, cand = Path(argv[1]), Path(argv[2])
    ignore = set(DEFAULT_IGNORE)
    if len(argv) > 4 and argv[3] == "--ignore":
        ignore |= set(argv[4].split(","))
    problems = 0
    ref_files = sorted(ref.glob("*.parquet"))
    cand_names = {p.name for p in cand.glob("*.parquet")}
    for path in ref_files:
        other = cand / path.name
        if not other.is_file():
            print(f"MISSING   {path.name}")
            problems += 1
            continue
        a = pd.read_parquet(path)
        b = pd.read_parquet(other)
        dropped = [c for c in ignore if c in a.columns or c in b.columns]
        a = a.drop(columns=[c for c in dropped if c in a.columns])
        b = b.drop(columns=[c for c in dropped if c in b.columns])
        try:
            pd.testing.assert_frame_equal(a, b, check_like=False)
            print(f"IDENTICAL {path.name} rows={len(a)}" + (f" (ignored {','.join(dropped)})" if dropped else ""))
        except AssertionError as exc:
            problems += 1
            detail = str(exc).splitlines()[0][:160]
            diff_cols = [c for c in a.columns if c in b.columns and len(a) == len(b) and not a[c].equals(b[c])]
            print(f"DIFFERENT {path.name} rows={len(a)}/{len(b)} cols={diff_cols[:8]} :: {detail}")
    extra = sorted(cand_names - {p.name for p in ref_files})
    for name in extra:
        print(f"EXTRA     {name}")
    print(f"RESULT: {'byte-equivalent tables' if problems == 0 else f'{problems} table(s) differ'}; extra={len(extra)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
