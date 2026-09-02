"""Small MuSR-inspired entailment-tree representation with full round trips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_ALLOWED_KINDS = frozenset({"hidden", "deduced", "explicit", "commonsense"})


@dataclass(frozen=True, slots=True)
class ReasoningNode:
    node_id: str
    text: str
    kind: str
    children: tuple["ReasoningNode", ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id or not self.text.strip():
            raise ValueError("reasoning nodes require non-empty IDs and text")
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"unsupported reasoning node kind: {self.kind}")

    def walk(self) -> tuple["ReasoningNode", ...]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return tuple(nodes)

    def explicit_leaves(self) -> tuple["ReasoningNode", ...]:
        return tuple(
            node
            for node in self.walk()
            if node.kind == "explicit" and not node.children
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "kind": self.kind,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReasoningNode":
        children = value.get("children", [])
        if not isinstance(children, list):
            raise ValueError("reasoning node children must be a list")
        return cls(
            node_id=str(value["node_id"]),
            text=str(value["text"]),
            kind=str(value["kind"]),
            children=tuple(cls.from_dict(child) for child in children),
        )


@dataclass(frozen=True, slots=True)
class ReasoningTree:
    latent_fact_id: str
    branch_id: str
    root: ReasoningNode

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_fact_id": self.latent_fact_id,
            "branch_id": self.branch_id,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReasoningTree":
        root = value.get("root")
        if not isinstance(root, Mapping):
            raise ValueError("reasoning tree root must be an object")
        return cls(
            latent_fact_id=str(value["latent_fact_id"]),
            branch_id=str(value["branch_id"]),
            root=ReasoningNode.from_dict(root),
        )


def build_reasoning_tree(
    *,
    latent_fact_id: str,
    branch_id: str,
    hidden_claim: str,
    intermediate_claims: tuple[str, ...],
    statements: tuple[str, ...],
    commonsense_bridges: tuple[str, ...],
) -> ReasoningTree:
    leaves = tuple(
        ReasoningNode(f"{branch_id}_explicit_{index}", text, "explicit")
        for index, text in enumerate(statements)
    ) + tuple(
        ReasoningNode(f"{branch_id}_commonsense_{index}", text, "commonsense")
        for index, text in enumerate(commonsense_bridges)
    )
    children = leaves
    for index, text in reversed(tuple(enumerate(intermediate_claims))):
        children = (
            ReasoningNode(f"{branch_id}_deduced_{index}", text, "deduced", children),
        )
    root = ReasoningNode(f"{branch_id}_root", hidden_claim, "hidden", children)
    return ReasoningTree(latent_fact_id, branch_id, root)
