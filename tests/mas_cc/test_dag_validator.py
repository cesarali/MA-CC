from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("validate_dag", ROOT / "scripts/Cygnus/dag/validate_dag.py")
validate_dag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_dag)


def _node(identifier, depends_on=(), authority="auto"):
    return {"id": identifier, "title": identifier, "does": "work", "gate": "checked",
            "authority": authority, "depends_on": list(depends_on)}


def test_committed_dag_is_structurally_valid_and_ordered():
    dag = validate_dag.load(ROOT / "docs/handoff/observatory-dag.yaml")
    assert validate_dag.structure(dag) == []
    order = validate_dag.topological(dag["nodes"])
    position = {identifier: index for index, identifier in enumerate(order)}
    assert all(position[d] < position[n["id"]] for n in dag["nodes"] for d in n["depends_on"] if d in position)


def test_structure_reports_cycles_unknown_dependencies_and_missing_gates():
    cyclic = {"nodes": [_node("a", ["c"]), _node("b", ["a"]), _node("c", ["b"])]}
    assert any(problem.startswith("cycle: ") for problem in validate_dag.structure(cyclic))
    assert "b: unknown dependency zz" in validate_dag.structure({"nodes": [_node("a"), _node("b", ["zz"])]})
    broken = _node("a", authority="someone")
    broken["gate"] = ""
    problems = validate_dag.structure({"nodes": [broken]})
    assert "a: missing gate" in problems and any("authority must be" in p for p in problems)


def test_next_separates_runnable_from_operator_gated_and_honours_done():
    dag = {"done": [{"id": "z"}], "nodes": [_node("a", ["z"]), _node("b", ["a"]), _node("c", authority="operator"),
                                            _node("d", ["c"])]}
    first = validate_dag.next_nodes(dag, set())
    assert first["runnable"] == ["a"] and first["waiting_on_operator"] == ["c"] and first["blocked"] == ["b", "d"]
    assert validate_dag.next_nodes(dag, {"a", "c"})["runnable"] == ["b", "d"]


def test_ancestors_is_the_transitive_closure():
    closure = validate_dag.ancestors([_node("a"), _node("b", ["a"]), _node("c", ["b"])])
    assert closure == {"a": set(), "b": {"a"}, "c": {"a", "b"}}
