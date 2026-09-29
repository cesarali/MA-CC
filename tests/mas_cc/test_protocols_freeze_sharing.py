"""_freeze must reuse already-frozen parts, without changing any value.

Game states are built once per turn and kept for the whole episode, and each
one extends the previous state's frozen history by one entry. When _freeze
rebuilt already-frozen data, every state carried a private deep copy of the
full history, and memory grew quadratically (about 2 GB per 30-round
task_004 episode, killing SLURM job 19687 at round ~16 under 8 GB).
"""

from collections import namedtuple
from types import MappingProxyType

import pytest

from mas_cc.games.protocols import _freeze


def _reference_freeze(value):
    # The original implementation, kept as the definition of the right values.
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return MappingProxyType({str(k): _reference_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_reference_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_reference_freeze(v) for v in value)
    return value


def _deep_equal(a, b):
    if isinstance(a, MappingProxyType):
        return isinstance(b, MappingProxyType) and a.keys() == b.keys() and all(
            _deep_equal(a[k], b[k]) for k in a
        )
    if isinstance(a, tuple):
        return type(b) is tuple and len(a) == len(b) and all(map(_deep_equal, a, b))
    return type(a) is type(b) and a == b


def test_extending_a_frozen_history_reuses_the_old_entries():
    history = _freeze([{"round": r, "votes": [1, 2, 3], "tags": {"a"}} for r in range(50)])
    extended = _freeze([*history, {"round": 50, "votes": [3, 2, 1], "tags": set()}])
    assert len(extended) == 51
    assert all(extended[i] is history[i] for i in range(50))


def test_already_frozen_values_come_back_unchanged():
    frozen = _freeze({"history": [{"x": 1}], "n": 3})
    assert _freeze(frozen) is frozen


@pytest.mark.parametrize(
    "value",
    [
        {"a": [1, {"b": [2, 3]}], 7: {"c"}},
        [({"k": [1]},), {"z"}, "s", None, 1.5],
        MappingProxyType({"a": [1, 2]}),      # proxy hiding a list: must be rebuilt
        MappingProxyType({1: "non-str key"}),  # non-string key: must be rebuilt
        namedtuple("Pair", "x y")(1, [2]),     # namedtuple still becomes a plain tuple
    ],
)
def test_values_match_the_original_implementation(value):
    assert _deep_equal(_freeze(value), _reference_freeze(value))


def test_result_is_still_read_only():
    frozen = _freeze({"history": [{"x": [1]}]})
    with pytest.raises(TypeError):
        frozen["history"] = ()
    assert type(frozen["history"][0]["x"]) is tuple
