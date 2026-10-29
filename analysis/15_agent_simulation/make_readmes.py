"""Write the README.md of each 15-agent setup folder from its JSON files, so the numbers cannot drift from the data.
Usage, from the repository root (after build_setups.py):
    .venv/bin/python analysis/15_agent_simulation/make_readmes.py
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
WORLD = {"task003": "task003 (allocation puzzle: Alice, Bruno, Chandra; A0 is correct, A2 is the false target)",
         "object_placement": "MuSR Object Placements (where Casey looks for the keys; A0 = drawer is correct, A2 = desk "
                             "is the false target)"}
KIND = {
    "symmetric": ("Agents can prove the answer; two controller pools",
                  "The 15 agents' facts together prove A0 (P(A0) = 1), but no single fact or pair does. They are a "
                  "subset of the 18 distinct agent facts of the original 24-agent setup, chosen to keep the proof, "
                  "minimise overlap with the controller pools and balance single-fact leanings. The two controller "
                  "pools are the original ones, unchanged: the A0 pool is used when the controller targets A0 (truth "
                  "arm), the A2 pool when it targets A2 (false arm)."),
    "no_solution": ("No proof; agents and the single controller pool are each approximately uniform",
                    "The 15 agents' facts together give approximately (1/3, 1/3, 1/3): they favour no answer and prove "
                    "nothing. The single 12-fact controller pool, used for both targets, is also approximately "
                    "uniform. No fact is in both. Agents and pool together prove nothing. Found by a seeded joint "
                    "search (`../../build_setups.py`)."),
}


def fmt(p):
    return "(" + ", ".join(f"{x:.3f}" for x in p) + ")"


def main():
    for task in ("task003", "object_placement"):
        for kind in ("symmetric", "no_solution"):
            folder = HERE / task / kind
            agents = json.loads((folder / "agents.json").read_text())
            pa = agents["properties"]
            pools = sorted(folder.glob("controller_pool*.json"))
            title, para = KIND[kind]
            lines = [f"# {task} / {kind} (15 agents): {title}", "",
                     f"**Built 29 October 2026** by [`../../build_setups.py`](../../build_setups.py) (deterministic; "
                     "rebuild rather than edit). World: " + WORLD[task] + ".", "", para, "",
                     "## Files", "", "| file | what it is |", "|---|---|",
                     "| `agents.json` | `agent_assignments` (15 agents, one fact each), every fact's text and strength, "
                     "and checked properties |"]
            for p in pools:
                d = json.loads(p.read_text())
                lines.append(f"| `{p.name}` | the 12-fact pool used when the controller targets "
                             f"{' or '.join(d['used_when_controller_targets'])} (`fact_ids` is the part code uses) |")
            lines += ["", "## Properties", "", "| | |", "|---|---|",
                      f"| agents, facts per agent, distinct facts | {pa['agents']}, {pa['facts_per_agent']}, "
                      f"{pa['distinct_facts']} |",
                      f"| agents' joint posterior (A0, A1, A2) | {fmt(pa['joint_posterior_A0_A1_A2'])}"
                      + (f": proves {pa['proves']}" if pa["proves"] else f"; max deviation from uniform "
                         f"{pa['max_deviation_from_uniform']}") + " |",
                      f"| agents' single-fact leanings A0 / A1 / A2 (neutral facts excluded) | "
                      f"{' / '.join(map(str, pa['single_fact_leanings_A0_A1_A2']))} |",
                      f"| neutral agent facts (no answer above 1/3 + 0.01 when read alone) | {pa['neutral_single_facts']} |",
                      f"| mean single-fact belief A0 / A1 / A2 | {fmt(pa['mean_individual_belief_A0_A1_A2'])} |",
                      f"| agent facts also in a controller pool | {', '.join(pa['facts_also_in_a_controller_pool']) or 'none'} |"]
            for p in pools:
                pp = json.loads(p.read_text())["properties"]
                lines += [f"| `{p.name}`: joint posterior | {fmt(pp['joint_posterior_A0_A1_A2'])}"
                          + ("" if not pp["proves_any_allocation"] else " (proves its target)") + " |",
                          f"| `{p.name}`: leanings A0 / A1 / A2, neutral facts | "
                          f"{' / '.join(map(str, pp['single_fact_leanings_A0_A1_A2']))}, {pp['neutral_single_facts']} |",
                          f"| `{p.name}`: new facts for the agents | {pp['new_facts_for_the_agents']} of {pp['size']} |",
                          f"| agents + whole `{p.name}` | {fmt(pp['agents_plus_whole_pool_posterior_A0_A1_A2'])}"
                          + (f": proves {pp['agents_plus_whole_pool_proves']}" if pp["agents_plus_whole_pool_proves"]
                             else ": proves nothing") + " |"]
            if task == "object_placement" and kind == "no_solution":
                lines += ["", "**Constraint of this world.** Object Placements has 34 true facts, 18 of them neutral "
                          "when read alone, and only 2 informative facts leaning A1. Since 27 distinct facts are needed, "
                          "at least 11 must be neutral; the search splits them so that agents and pool carry the same "
                          "neutral fraction. Here both A1-leaning facts ended in the pool."]
            (folder / "README.md").write_text("\n".join(lines) + "\n")
            print("wrote", folder / "README.md")


if __name__ == "__main__":
    main()
