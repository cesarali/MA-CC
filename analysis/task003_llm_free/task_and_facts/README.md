# task_and_facts — what task_003 actually is

Read [`task_003_analysis.md`](task_003_analysis.md) first. It defines the task
from the nine hidden numbers up, explains every classification applied to the 49
facts, and reports who can prove the answer.

| file | contents |
|---|---|
| `task_003_analysis.md` | **main report** — the task, the 99-proposition catalog, the 49 true facts, the four classifications, proof enumeration, the full fact table |
| `task_003_facts.csv` | all 49 facts with ΔP(A0/A1/A2), entropy, maxP, eligibility, decisiveness, allocation |
| `task_003_proofs.json` | minimal solving sets, per-fact appearance counts, coverage, disjoint packing |
| `generality.json` | the same measures on 12 other random worlds |

## Scripts

| script | produces |
|---|---|
| `engine.py` | the world + exact posterior and proof engine. **Imported by both setups' `build.py` and the simulator; do not move it** |
| `build_table.py` | `task_003_facts.csv`, with consistency assertions |
| `proofs.py` | `task_003_proofs.json` |
| `generality.py` | `generality.json` |

## Key numbers

The world numbers hold for both frozen setups. The rows from "controller pool"
to "assemblable" describe the **archived** setup (César's 24-fact pool and the
archived agents), not the frozen ones; see `../experimental_setup/` for those.

| | |
|---|---:|
| hidden world | `(3,1,1,2,2,1,1,1,2)` |
| allocation scores | 8 / 4 / 6 — `ALLOCATION_0` correct |
| propositions in the catalog | 99 |
| true in this world | **49** |
| private-eligible (may go in an agent packet) | 41 |
| decisive (the greedy proof set) | 6 — a subset of the 41 |
| controller pool | 24 |
| agent packets | 21 distinct, 1 per agent |
| **in neither** | **12** |
| minimum proof size | **4 facts** (24 such proofs) |
| minimal proofs up to size 6 | 7,694 |
| assemblable by agents / by controller | **4** / **0** |
| disjoint proofs (lower bound) | 4 |
