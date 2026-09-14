"""Verify every retained scientific table and archive member against a baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import pandas as pd

from mas_cc.studies.table_io import read_scientific_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline analysis directory")
    parser.add_argument("candidate", type=Path, help="Candidate analysis directory")
    args = parser.parse_args()
    names = sorted(path.name for path in (args.baseline / "tables").glob("*.parquet"))
    assert names and names == sorted(path.name for path in (args.candidate / "tables").glob("*.parquet"))
    counts = {}
    for name in names:
        expected = read_scientific_table(args.baseline / "tables" / name)
        actual = read_scientific_table(args.candidate / "tables" / name)
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-12, obj=name)
        counts[name] = len(actual)
        print(json.dumps({"table": name, "rows": len(actual), "matches": True}), flush=True)
    baseline_zip, = args.baseline.glob("*_analysis.zip")
    candidate_zip, = args.candidate.glob("*_analysis.zip")
    with zipfile.ZipFile(baseline_zip) as expected, zipfile.ZipFile(candidate_zip) as actual:
        assert sorted(actual.namelist()) == sorted(expected.namelist())
        assert actual.testzip() is None
        for name in actual.namelist():
            assert not ({".work", "cache", "logs", "runs"} & set(Path(name).parts))
    print(json.dumps({"all_tables_match": True, "table_rows": counts,
                      "package_members_match": True}), flush=True)


if __name__ == "__main__":
    main()
