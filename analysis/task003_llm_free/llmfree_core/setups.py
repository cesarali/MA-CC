"""Load a frozen setup: its agents, and the controller pool for a given target."""
from __future__ import annotations
import hashlib, json, pathlib
from dataclasses import dataclass

from .world import World

TARGETS = {"truth": 0, "false": 2}          # arm -> target allocation index


@dataclass(frozen=True)
class Setup:
    name: str
    agent_ids: tuple[str, ...]               # sorted
    agent_facts: tuple[int, ...]             # one fact index per agent, same order
    pools: dict                              # target index -> tuple of fact indices, sorted by fact ID
    file_hashes: dict                        # file name -> sha256, for the run manifest


def _read(path: pathlib.Path, hashes: dict) -> dict:
    raw = path.read_bytes()
    hashes[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def load_setup(name: str, setups_dir: str | pathlib.Path, world: World) -> Setup:
    folder = pathlib.Path(setups_dir) / name
    hashes: dict = {}
    agents = _read(folder / "agents.json", hashes)
    ids = tuple(sorted(agents["agent_assignments"]))
    facts = []
    for a in ids:
        assigned = agents["agent_assignments"][a]
        assert len(assigned) == 1, f"{a} holds {len(assigned)} facts; the simulator expects 1"
        facts.append(world.index[assigned[0]])

    def pool(file):
        ids_ = _read(folder / file, hashes)["fact_ids"]
        return tuple(sorted((world.index[f] for f in ids_), key=lambda i: world.fact_ids[i]))

    if name == "task003_symmetric":
        pools = {0: pool("controller_pool_A0.json"), 2: pool("controller_pool_A2.json")}
    elif name == "task003_nosolution":
        p = pool("controller_pool.json")
        pools = {0: p, 2: p}
    else:
        raise ValueError(f"unknown setup {name!r}")
    return Setup(name, ids, tuple(facts), pools, hashes)
