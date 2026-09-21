from __future__ import annotations

import pandas as pd

from mas_cc.studies.canonical import _harmonise_target_semantics
from mas_cc.studies.table_io import write_scientific_table


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
    try:
        write_scientific_table(tmp_path / "before", "cells", pd.DataFrame(rows))
    except Exception as error:  # the failure this fix exists for
        assert "target_semantics" in str(error)
    else:
        raise AssertionError("a mixed bool/str column was expected to be unwritable")
    assert _harmonise_target_semantics([rows], frames) is True
    assert [r["target_semantics"] for r in rows] == ["none", "false", "truth", None]
    assert frames["c"]["target_semantics"].tolist() == ["truth", "none"]
    write_scientific_table(tmp_path / "after", "cells", pd.DataFrame(rows))
