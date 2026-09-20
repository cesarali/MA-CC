"""Validate a work DAG and pick what can run next.

Two layers. The structural layer is deterministic: unique ids, known dependencies, no cycles,
a gate on every node, a declared authority. The judgment layer asks System One (Jev) typed
questions and never edits the DAG: it reports

* missing edges  - for each unordered pair with no path between them, P(b cannot start
  correctly until a is finished), both directions;
* weak edges     - for each declared edge, the same probability (low = the edge may be
  unnecessary and is serialising work for nothing);
* authority      - P(the node performs an irreversible, outward-facing, paid or secret-handling
  action that needs a human sign-off), compared with the declared ``authority``;
* readiness/risk - P(the node is specified well enough to execute without a further decision)
  and an ordered risk score.

    python scripts/Cygnus/dag/validate_dag.py docs/handoff/observatory-dag.yaml --structure-only
    python scripts/Cygnus/dag/validate_dag.py docs/handoff/observatory-dag.yaml --out /tmp/dag-report
    python scripts/Cygnus/dag/validate_dag.py docs/handoff/observatory-dag.yaml next --state state.json

``next`` is the automation entry point: it prints the nodes whose dependencies are done, split
into ``runnable`` (authority auto) and ``waiting_on_operator``. The state file is
``{"done": [ids]}``; ids under the DAG's own ``done:`` list count as done too.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

AUTHORITIES = {"auto", "operator"}
RISK_LEVELS = ["no risk to existing behaviour or data", "low risk, easily reverted", "moderate risk, needs a careful gate",
               "high risk, could lose data, spend money or break a live service"]
MISSING_EDGE = 0.70
WEAK_EDGE = 0.30
AUTHORITY_DISAGREE = 0.70


def load(path: Path) -> dict[str, Any]:
    dag = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(dag, dict) or not isinstance(dag.get("nodes"), list):
        raise SystemExit(f"{path}: expected a mapping with a 'nodes' list")
    return dag


def structure(dag: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    nodes = dag["nodes"]
    ids = [node.get("id") for node in nodes]
    done = {item["id"] for item in dag.get("done", [])}
    for identifier in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"duplicate node id {identifier}")
    for identifier in done & set(ids):
        problems.append(f"{identifier} is listed both as done and as a node")
    for node in nodes:
        name = node.get("id", "<no id>")
        for key in ("id", "title", "does", "gate", "authority"):
            if not str(node.get(key, "")).strip():
                problems.append(f"{name}: missing {key}")
        if node.get("authority") not in AUTHORITIES:
            problems.append(f"{name}: authority must be one of {sorted(AUTHORITIES)}")
        for dependency in node.get("depends_on", []) or []:
            if dependency not in ids and dependency not in done:
                problems.append(f"{name}: unknown dependency {dependency}")
            if dependency == name:
                problems.append(f"{name}: depends on itself")
    if not problems and (cycle := _cycle(nodes)):
        problems.append("cycle: " + " -> ".join(cycle))
    return problems


def _edges(nodes: list[dict[str, Any]]) -> dict[str, set[str]]:
    known = {node["id"] for node in nodes}
    return {node["id"]: {d for d in (node.get("depends_on") or []) if d in known} for node in nodes}


def _cycle(nodes: list[dict[str, Any]]) -> list[str] | None:
    edges, state, trail = _edges(nodes), {}, []

    def visit(identifier: str) -> list[str] | None:
        state[identifier] = 1
        trail.append(identifier)
        for dependency in sorted(edges[identifier]):
            if state.get(dependency) == 1:
                return trail[trail.index(dependency):] + [dependency]
            if dependency not in state and (found := visit(dependency)):
                return found
        state[identifier] = 2
        trail.pop()
        return None

    for identifier in sorted(edges):
        if identifier not in state and (found := visit(identifier)):
            return found
    return None


def ancestors(nodes: list[dict[str, Any]]) -> dict[str, set[str]]:
    edges = _edges(nodes)
    closure: dict[str, set[str]] = {}

    def collect(identifier: str) -> set[str]:
        if identifier not in closure:
            closure[identifier] = set()
            for dependency in edges[identifier]:
                closure[identifier] |= {dependency} | collect(dependency)
        return closure[identifier]

    for identifier in edges:
        collect(identifier)
    return closure


def topological(nodes: list[dict[str, Any]]) -> list[str]:
    edges, order, placed = _edges(nodes), [], set()
    while len(order) < len(edges):
        layer = sorted(i for i in edges if i not in placed and edges[i] <= placed)
        order += layer
        placed |= set(layer)
    return order


def next_nodes(dag: dict[str, Any], done: set[str]) -> dict[str, list[str]]:
    done = done | {item["id"] for item in dag.get("done", [])}
    ready = [n for n in dag["nodes"] if n["id"] not in done and set(n.get("depends_on") or []) <= done]
    blocked = [n["id"] for n in dag["nodes"] if n["id"] not in done and n not in ready]
    return {"runnable": [n["id"] for n in ready if n["authority"] == "auto"],
            "waiting_on_operator": [n["id"] for n in ready if n["authority"] == "operator"],
            "blocked": blocked, "done": sorted(done)}


def _card(node: dict[str, Any]) -> dict[str, Any]:
    return {"id": node["id"], "title": node["title"], "does": " ".join(str(node["does"]).split()),
            "gate": node["gate"], "touches": node.get("touches", [])}


def judge(dag: dict[str, Any], out_dir: Path, workers: int) -> dict[str, Any]:
    from mas_cc.analysis.systemone import Budget, SystemOneClient, noul, score

    nodes = dag["nodes"]
    by_id = {node["id"]: node for node in nodes}
    closure = ancestors(nodes)
    context = {"project": dag.get("name"), "finished_already": [f"{d['id']}: {d['title']}" for d in dag.get("done", [])]}
    client = SystemOneClient(cache_dir=out_dir / "systemone_cache", budget=Budget(max_usd=0.50), workers=workers)

    node_items = [({**context, "work_item": _card(node)}, {
        "ready": noul("The work item is specified concretely enough that an engineer could carry it out and check its gate "
                      "without making a further design decision."),
        "needs_signoff": noul("Carrying out this work item performs an action that is irreversible, outward-facing to other "
                              "people, spends money, or writes or moves a secret, and therefore needs a human sign-off first. "
                              "Opening a pull request without merging it does not count."),
        "risk": score("How risky is carrying out this work item?", RISK_LEVELS)}) for node in nodes]

    def order_question(first: str, second: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return ({**context, "earlier_candidate": _card(by_id[first]), "later_candidate": _card(by_id[second])},
                {"must_precede": noul("The later candidate cannot be started correctly until the earlier candidate is "
                                      "finished, because it needs something the earlier candidate produces.")})

    unordered = [(a, b) for a, b in combinations(sorted(by_id), 2) if a not in closure[b] and b not in closure[a]]
    pair_keys = [pair for a, b in unordered for pair in ((a, b), (b, a))]
    edge_keys = [(dependency, node["id"]) for node in nodes for dependency in sorted(_edges(nodes)[node["id"]])]
    records = client.ask_many(node_items + [order_question(*k) for k in pair_keys] + [order_question(*k) for k in edge_keys])
    node_records = records[:len(nodes)]
    pair_records = records[len(nodes):len(nodes) + len(pair_keys)]
    edge_records = records[len(nodes) + len(pair_keys):]

    per_node = []
    for node, record in zip(nodes, node_records):
        answers = record["answers"]
        signoff = float(answers["needs_signoff"]["noul"])
        declared = node["authority"]
        disagreement = (declared == "auto" and signoff >= AUTHORITY_DISAGREE) or (declared == "operator" and signoff <= 1 - AUTHORITY_DISAGREE)
        per_node.append({"id": node["id"], "authority": declared, "p_needs_signoff": round(signoff, 3),
                         "authority_disagreement": disagreement, "p_ready": round(float(answers["ready"]["noul"]), 3),
                         "risk": round(float(answers["risk"]["score"]), 2)})
    missing = sorted(({"earlier": a, "later": b, "p": round(float(r["answers"]["must_precede"]["noul"]), 3)}
                      for (a, b), r in zip(pair_keys, pair_records)
                      if float(r["answers"]["must_precede"]["noul"]) >= MISSING_EDGE), key=lambda row: -row["p"])
    declared_edges = [{"earlier": a, "later": b, "p": round(float(r["answers"]["must_precede"]["noul"]), 3)}
                      for (a, b), r in zip(edge_keys, edge_records)]
    return {"model": client.model, "usage": client.usage.as_dict(), "nodes": per_node, "missing_edge_candidates": missing,
            "declared_edges": declared_edges, "weak_edges": [e for e in declared_edges if e["p"] <= WEAK_EDGE],
            "pairs_checked": len(pair_keys), "thresholds": {"missing_edge": MISSING_EDGE, "weak_edge": WEAK_EDGE,
                                                            "authority": AUTHORITY_DISAGREE}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dag", type=Path)
    parser.add_argument("command", nargs="?", choices=["validate", "next"], default="validate")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("dag-report"))
    parser.add_argument("--state", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    dag = load(args.dag)
    problems = structure(dag)
    if problems:
        print(json.dumps({"structure": "invalid", "problems": problems}, indent=2))
        return 1
    if args.command == "next":
        done = set(json.loads(args.state.read_text())["done"]) if args.state and args.state.is_file() else set()
        print(json.dumps(next_nodes(dag, done), indent=2))
        return 0
    report: dict[str, Any] = {"structure": "valid", "nodes": len(dag["nodes"]), "order": topological(dag["nodes"]),
                              "next": next_nodes(dag, set())}
    if not args.structure_only:
        args.out.mkdir(parents=True, exist_ok=True)
        report["judgment"] = judge(dag, args.out, args.workers)
        (args.out / "dag_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
