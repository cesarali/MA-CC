"""Build the K-projects worlds and setups (fact scaling, decided 22 October 2026).

For K in (2, 4) writes:
  llmfree_core/world_task003_K<K>.json             {"type": "k_projects", "K": K, "base": "world_task003.json"}
  experimental_setup/task003_symmetric_K<K>/       agents.json (24K agents: the 24 base agents once per project,
                                                    each holding its base fact for that project), controller pools
                                                    (the base pools once per project: 12K facts)
  experimental_setup/task003_nosolution_K<K>/      the same for task003_nosolution
Fact IDs are "k<p>:<base id>" (llmfree_core/kworld.py). The frozen base setups are only read.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/finite_size/build_k_setups.py
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]          # analysis/task003_llm_free
SETUPS = ROOT / "experimental_setup"


def main():
    for K in (2, 4):
        (ROOT / "llmfree_core" / f"world_task003_K{K}.json").write_text(json.dumps(
            {"type": "k_projects", "K": K, "base": "world_task003.json",
             "what_this_is": f"{K} independent task_003-type projects with one split (llmfree_core/kworld.py)"},
            indent=1) + "\n")
        for name, pools in (("task003_symmetric", ("controller_pool_A0.json", "controller_pool_A2.json")),
                            ("task003_nosolution", ("controller_pool.json",))):
            src, dst = SETUPS / name, SETUPS / f"{name}_K{K}"
            dst.mkdir(exist_ok=True)
            a = json.loads((src / "agents.json").read_text())
            assign = {f"p{p}_{agent}": [f"k{p}:{fid}" for fid in facts]
                      for p in range(K) for agent, facts in a["agent_assignments"].items()}
            out = {k: v for k, v in a.items() if k not in ("agent_assignments", "status", "built_by")}
            out.update(setup=f"{a['setup']}-K{K}", status=f"BUILT 2026-10-22 from the frozen {name}",
                       built_by="key_phenomena_hypothesis/finite_size/build_k_setups.py", K=K,
                       what_this_is=f"{K} projects with one split; per project the agents and facts of {name}.",
                       agent_assignments=assign)
            (dst / "agents.json").write_text(json.dumps(out, indent=1) + "\n")
            for f in pools:
                d = json.loads((src / f).read_text())
                d["fact_ids"] = [f"k{p}:{fid}" for p in range(K) for fid in d["fact_ids"]]
                d.update(status=f"BUILT 2026-10-22 from the frozen {name}", K=K)
                (dst / f).write_text(json.dumps(d, indent=1) + "\n")
            (dst / "README.md").write_text(
                f"# {name}_K{K}\n\n{K} independent projects with one split (same person alone in every project; "
                f"scores summed). Per project: the agents and controller pools of the frozen `{name}`, with fact IDs "
                f"prefixed `k<p>:`. World: `llmfree_core/world_task003_K{K}.json`. Built by "
                f"`simulation_2/key_phenomena_hypothesis/finite_size/build_k_setups.py`; rebuild rather than edit.\n")
            print("->", dst)


if __name__ == "__main__":
    main()
