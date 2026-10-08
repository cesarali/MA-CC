"""Export task_003's possible worlds and facts to world_task003.json, once (shared by all simulators).

This is the ONLY part of the simulator that uses MA-CC code (through
../task_and_facts/engine.py). Everything else reads the exported file, so the
simulator runs without MA-CC installed.

What is exported:
  - the 14,388 possible worlds that have a unique winning allocation, as bit
    positions (world i is bit i);
  - for each allocation, a bitmask of the worlds it wins;
  - for each of the 49 true facts, its ID, text, and a bitmask of the worlds in
    which it holds.

The posterior given a set of facts is then: AND the facts' masks, count the
surviving worlds won by each allocation, divide by the number surviving.

Run from the repository root:
    .venv/bin/python analysis/task003_llm_free/llmfree_core/export_world.py
"""
from __future__ import annotations
import csv, hashlib, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "task_and_facts"))
from engine import World  # noqa: E402

OUT = HERE / "world_task003.json"
VECTOR = (3, 1, 1, 2, 2, 1, 1, 1, 2)

w = World(VECTOR)
text = {r["fact"]: r["text"] for r in csv.DictReader(open(ROOT / "task_and_facts" / "task_003_facts.csv"))}
assert set(text) == set(w.ids), "facts CSV and engine disagree"

world = {
    "task": "task_003",
    "hidden_vector": list(VECTOR),
    "truth": f"ALLOCATION_{w.truth}",
    "n_worlds": w.n_worlds,
    "all_mask": format(w.all_mask, "x"),
    "win_masks": {f"ALLOCATION_{k}": format(w.win_mask[k], "x") for k in range(3)},
    "facts": [{"id": f, "text": text[f], "mask": format(w.fact_mask[i], "x")} for i, f in enumerate(w.ids)],
}

# self-check: the exported masks reproduce the engine's posteriors
for i, f in enumerate(w.ids):
    m = int(world["facts"][i]["mask"], 16) & int(world["all_mask"], 16)
    n = m.bit_count()
    got = tuple((m & int(world["win_masks"][f"ALLOCATION_{k}"], 16)).bit_count() / n for k in range(3))
    assert got == w.posterior([i]), f
payload = json.dumps(world, indent=1, ensure_ascii=False)
OUT.write_text(payload + "\n")
print(f"wrote {OUT.name}: {w.n_worlds} worlds, {len(w.ids)} facts, "
      f"sha256 {hashlib.sha256(payload.encode()).hexdigest()[:16]}")
