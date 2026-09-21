from __future__ import annotations

import pandas as pd

from mas_cc.studies.canonical import _harmonise_target_semantics
from mas_cc.studies.table_io import single_kind_semantics, write_scientific_table


def _rows(*values):
    return [{"cell_id": f"cell-{i:04d}", "target_semantics": value} for i, value in enumerate(values)]


def test_single_spelling_is_left_exactly_as_it_was():
    booleans, strings = _rows(False, False), _rows("truth", "truth")
    frame = {"c": pd.DataFrame(_rows(False))}
    assert _harmonise_target_semantics([booleans], frame) is False
    assert _harmonise_target_semantics([strings], {}) is False
    assert [r["target_semantics"] for r in booleans] == [False, False]
    assert frame["c"]["target_semantics"].tolist() == [False]


def test_mixed_yaml_booleans_and_strings_become_canonical_labels_and_write(tmp_path):
    rows = _rows("none", False, True, None)
    frames = {"c": pd.DataFrame(_rows(True, "none"))}
    # The raw failure this exists for: pyarrow cannot hold bool and str in one column.
    import pyarrow as pa
    import pytest

    with pytest.raises(pa.ArrowTypeError, match="target_semantics"):
        pd.DataFrame(rows).to_parquet(tmp_path / "raw.parquet", engine="pyarrow")
    assert _harmonise_target_semantics([rows], frames) is True
    assert [r["target_semantics"] for r in rows] == ["none", "false", "truth", None]
    assert frames["c"]["target_semantics"].tolist() == ["truth", "none"]
    write_scientific_table(tmp_path / "after", "cells", pd.DataFrame(rows))


def test_report_tables_concatenated_from_differently_spelled_sources_are_writable(tmp_path):
    derived = pd.DataFrame({"target_semantics": ["false", "truth"], "estimate": [1.0, 2.0]})
    legacy = pd.DataFrame({"target_semantics": [False, True], "estimate": [3.0, 4.0]})
    table = pd.concat([derived, legacy], ignore_index=True)
    path = write_scientific_table(tmp_path, "report", table)
    assert pd.read_parquet(path)["target_semantics"].tolist() == ["false", "truth", "false", "truth"]


def test_single_kind_tables_are_the_same_object():
    for values in ([False, True], ["false", "none"], [None, None]):
        frame = pd.DataFrame({"target_semantics": values})
        assert single_kind_semantics(frame) is frame
    assert single_kind_semantics(pd.DataFrame({"x": [1]}))["x"].tolist() == [1]
