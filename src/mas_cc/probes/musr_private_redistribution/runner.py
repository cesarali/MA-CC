"""Prepare, run, analyze, and seal the redistribution calibration."""

from __future__ import annotations
import asyncio,csv,json,shutil,subprocess
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any,Mapping
import yaml
from mas_cc.games.relational_reasoning.data import RelationalTask,load_musr_team_allocation_task
from mas_cc.llm_runtime.providers import UniversityPricingSource
from mas_cc.musr_team_allocation_generator.io_utils import sha256_file,sha256_object,write_json_atomic
from mas_cc.probes.musr_prompt_solvability.execution import execute
from mas_cc.probes.musr_prompt_solvability.prompting import render
from .analysis import observation_rows,plots,report,select_regime,summarize,terminal,write_csv
from .assignment import REGIMES,build_assignment,structural_summary
from .config import RedistributionConfig
from .design import assignments,call_plan


def _git()->dict[str,Any]:
    try:return {"commit":subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(),"dirty":bool(subprocess.run(["git","status","--porcelain"],check=True,capture_output=True,text=True).stdout.strip())}
    except (OSError,subprocess.CalledProcessError):return {"commit":None,"dirty":None}

def load_tasks(config:RedistributionConfig)->dict[str,RelationalTask]: return {task_id:load_musr_team_allocation_task(config.task_dir,task_id,population_size=config.population_size) for task_id in config.tasks}

def _assignment_checks(tasks:Mapping[str,RelationalTask],amap:Mapping[tuple[str,str],Mapping])->list[dict[str,Any]]:
    summaries=[structural_summary(value) for value in amap.values()]; expected={"R2":sorted([3]*6+[2]*3),"R3":[4]*9,"R4":sorted([6]*3+[5]*6)}
    return [
        {"check":"all_tasks","passed":len(tasks)==3 and all(len(t.fact_order)==27 for t in tasks.values())},
        {"check":"collective_completeness","passed":all(row["global_9_of_9"] for row in summaries)},
        {"check":"six_cards","passed":all(row["min_cards"]==row["max_cards"]==6 for row in summaries if row["regime"]!="NAT")},
        {"check":"exact_latent_targets","passed":all(row["min_latent_values"]==row["max_latent_values"]==int(row["regime"][1]) for row in summaries if row["regime"]!="NAT")},
        {"check":"holder_profiles","passed":all(sorted(amap[(task_id,regime)]["latent_holder_counts"].values())==expected[regime] for task_id in tasks for regime in ("R2","R3","R4"))},
        {"check":"r4_no_candidate_completeness","passed":all(row["agents_fully_scoring_any_candidate"]==0 for row in summaries if row["regime"]=="R4")},
        {"check":"all_cards_used","passed":all(set(value["card_holder_counts"])==set(tasks[task_id].fact_order) for (task_id,_),value in amap.items())},
    ]

def prepare(config:RedistributionConfig,output_dir:Path|None=None)->tuple[Path,dict[str,RelationalTask],dict,tuple,dict]:
    root=Path(output_dir or config.output_dir); root.mkdir(parents=True,exist_ok=True); tasks=load_tasks(config); amap=assignments(tasks,config.seed); specs=call_plan(tasks,amap,config.private_repetitions,config.endpoint_repetitions,config.seed); rendered={s.call_id:render(tasks[s.task_id],s) for s in specs}; checks=_assignment_checks(tasks,amap)
    forbidden=("skill_matrix","cooperation_matrix","candidate_scores","gold_answer","candidate_score_terms"); checks += [{"check":"call_count","passed":len(specs)==492},{"check":"hidden_metadata_absent","passed":all(all(term not in "\n".join(m.content for m in p.messages) for term in forbidden) for p in rendered.values())},{"check":"budgets","passed":len(specs)<=config.max_requests and sum(p.token_estimate for p in rendered.values())<=config.max_input_tokens and len(specs)*config.provider.max_output_tokens<=config.max_output_tokens_total}]
    quote=UniversityPricingSource(config.provider).fetch(config.provider.type,config.provider.model); payload={"passed":all(c["passed"] for c in checks),"checks":checks,"calls":{"total":len(specs),"private":432,"zero_f9":60},"estimated_input_tokens":sum(p.token_estimate for p in rendered.values()),"maximum_output_tokens":len(specs)*config.provider.max_output_tokens,"workers":config.workers,"estimated_wall_seconds":len(specs)/config.workers*float(config.provider.options.get("estimated_latency_seconds",10)),"conservative_wall_seconds":len(specs)/config.workers*config.provider.timeout_seconds,"assignment_sha256":sha256_object({f"{k[0]}:{k[1]}":v["assignment_sha256"] for k,v in amap.items()}),"call_plan_sha256":sha256_object([s.to_dict() for s in specs]),"prompt_sha256":sha256_object({k:v.to_dict() for k,v in rendered.items()}),"pricing":quote.to_dict(),"conservative_cost":quote.pricing.cost(config.max_input_tokens,min(config.max_output_tokens_total,len(specs)*config.provider.max_output_tokens)).to_dict() if quote.pricing else None}
    pre=root/"preflight"; pre.mkdir(parents=True,exist_ok=True); write_json_atomic(pre/"preflight.json",payload); write_json_atomic(pre/"call_plan.json",[s.to_dict() for s in specs]); write_json_atomic(pre/"pricing_snapshot.json",quote.to_dict()); approval=sha256_object({"config":config.to_dict(),"preflight":payload}); (pre/"preflight_id.txt").write_text(approval+"\n"); (pre/"report.md").write_text(f"# Private redistribution preflight\n\n- Passed: **{payload['passed']}**\n- Calls: {len(specs)}\n- Estimated input tokens: {payload['estimated_input_tokens']:,}\n- Maximum output tokens: {payload['maximum_output_tokens']:,}\n- Expected wall time: {payload['estimated_wall_seconds']/60:.1f} minutes\n- Conservative wall time: {payload['conservative_wall_seconds']/3600:.2f} hours\n",encoding="utf-8")
    assign_dir=root/"assignments"; audit=root/"structural_audit"; assign_dir.mkdir(parents=True,exist_ok=True); audit.mkdir(parents=True,exist_ok=True); summary=[]; diagnostics=[]; holders=[]
    for (task_id,regime),value in amap.items(): write_json_atomic(assign_dir/f"{task_id}_{regime}_assignments.json",value); summary.append(structural_summary(value)); diagnostics.extend({**row,"regime":regime} for row in value["diagnostics"]); holders.extend({"task_id":task_id,"regime":regime,"latent_id":latent,"holders":count} for latent,count in value["latent_holder_counts"].items())
    write_csv(assign_dir/"assignment_summary.csv",summary); write_csv(audit/"current_private_coverage.csv",[r for r in diagnostics if r["regime"]=="NAT"]); write_csv(audit/"redistributed_private_coverage.csv",[r for r in diagnostics if r["regime"]!="NAT"]); write_csv(audit/"candidate_score_coverage.csv",diagnostics); write_csv(audit/"latent_holder_counts.csv",holders)
    (root/"config.yaml").write_text(yaml.safe_dump(config.to_dict(),sort_keys=False)); (root/"README.md").write_text("# MuSR private redistribution calibration 01\n\nSee `analysis/private_redistribution_calibration_report.md`.\n"); manifest={"schema_version":1,"probe":"musr_private_redistribution","status":"planned","provider":config.provider.type,"model":config.provider.model,"config_sha256":sha256_object(config.to_dict()),"assignment_sha256":payload["assignment_sha256"],"call_plan_sha256":payload["call_plan_sha256"],"prompt_sha256":payload["prompt_sha256"],"prior_calibration_manifest_sha256":sha256_file(Path("results/studies/musr_prompt_solvability_calibration_01/manifest.json")),"mas_cc_git":_git()}; write_json_atomic(root/"manifest.json",manifest)
    return root,tasks,amap,specs,rendered

def _approval(root:Path,value:Path|str|None)->None:
    actual=Path(value).read_text().strip() if value is not None and Path(str(value)).is_file() else str(value or "").strip(); expected=(root/"preflight/preflight_id.txt").read_text().strip()
    if actual!=expected: raise RuntimeError("probe run requires matching preflight approval")

def _outputs(config:RedistributionConfig,root:Path,tasks:Mapping[str,RelationalTask],amap:Mapping,specs:tuple)->dict[str,Any]:
    raw=terminal(root/"behavioral/raw_calls.jsonl"); observations=observation_rows(raw,tasks); write_csv(root/"behavioral/observation_level_results.csv",observations); pooled=summarize(observations,("regime",)); per_task=summarize(observations,("task_id","regime")); write_csv(root/"behavioral/summary_by_regime.csv",pooled); write_csv(root/"behavioral/summary_by_task_regime.csv",per_task)
    dev=summarize([r for r in observations if r["task_id"] in config.development],("regime",)); selected=select_regime(dev); heldout=summarize([r for r in observations if r["task_id"] in config.heldout],("regime",)); write_json_atomic(root/"behavioral/selection.json",{"selected_private_regime":selected,"rule":"largest of R4/R3/R2 with development truth <= 0.50"})
    tables=root/"analysis/tables"; write_csv(tables/"regime_truth_rates.csv",pooled); write_csv(tables/"truth_by_latent_coverage.csv",summarize(observations,("num_latent_values",))); write_csv(tables/"truth_by_card_count.csv",summarize(observations,("num_cards",))); write_csv(tables/"candidate_score_coverage_results.csv",summarize(observations,("num_fully_scoreable_allocations",))); zero=[r for r in observations if r["regime"]=="Zero"]; zero_rows=[{"task_id":task,"n":len([r for r in zero if r["task_id"]==task]),**{key:sum(r["parsed_semantic_answer"]==key for r in zero if r["task_id"]==task) for key in ("ALLOCATION_0","ALLOCATION_1","ALLOCATION_2")}} for task in config.tasks]; write_csv(tables/"zero_semantic_preferences.csv",zero_rows)
    structural=list(csv.DictReader((root/"assignments/assignment_summary.csv").open())); converted=[{k:(float(v) if k.startswith(("mean_","std_")) else int(v) if k.startswith(("min_","max_","agents_","fully_")) else v=="True" if k=="global_9_of_9" else v) for k,v in row.items()} for row in structural]; gold=Counter(task.correct_relation for task in tasks.values()); plots(pooled,observations,converted,root/"analysis/figures"); text=report(converted,pooled,per_task,selected,heldout,gold); report_path=root/"analysis/private_redistribution_calibration_report.md"; report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(text)
    held={r["regime"]:r for r in heldout}; passed=selected is not None and held.get(selected,{}).get("truth_rate",1)<=.5 and held.get("F9",{}).get("truth_rate",0)>=.8 and held.get("Zero",{}).get("truth_rate",1)<=.5; execution={"scheduled":len(specs),"terminal":len(raw),"successful":sum(r.get("parse_success") is True for r in raw),"failed":sum(r.get("parse_success") is not True for r in raw)}; artifacts={str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="manifest.json"}; manifest=json.loads((root/"manifest.json").read_text()); manifest.update({"status":"complete" if execution["successful"]==len(specs) else "incomplete","acceptance_decision":"PASS" if passed else "FAIL","selected_private_regime":selected,"execution":execution,"observed_usage":{"requests":len(raw),"input_tokens":sum(int((r.get("usage") or {}).get("input_tokens") or 0) for r in raw),"output_tokens":sum(int((r.get("usage") or {}).get("output_tokens") or 0) for r in raw),"transport_retries":sum(int(r.get("transport_retries") or 0) for r in raw)},"artifact_hashes":artifacts}); manifest.pop("manifest_content_sha256",None); manifest["manifest_content_sha256"]=sha256_object(manifest); write_json_atomic(root/"manifest.json",manifest); return {"output":str(root),"report":str(report_path),"selected":selected,"decision":"PASS" if passed else "FAIL","pooled":pooled,"heldout":heldout,"execution":execution}

async def run(config:RedistributionConfig,output_dir:Path|None=None,*,approve_preflight:Path|str|None=None)->dict[str,Any]:
    root=Path(output_dir or config.output_dir); _approval(root,approve_preflight); tasks=load_tasks(config); amap=assignments(tasks,config.seed); specs=call_plan(tasks,amap,config.private_repetitions,config.endpoint_repetitions,config.seed); rendered={s.call_id:render(tasks[s.task_id],s) for s in specs}; await execute(config,tasks,specs,rendered,root/"behavioral/raw_calls.jsonl",probe_name="musr_private_redistribution"); return _outputs(config,root,tasks,amap,specs)

def analyze(config:RedistributionConfig,output_dir:Path|None=None)->dict[str,Any]:
    root=Path(output_dir or config.output_dir); tasks=load_tasks(config); amap=assignments(tasks,config.seed); specs=call_plan(tasks,amap,config.private_repetitions,config.endpoint_repetitions,config.seed); return _outputs(config,root,tasks,amap,specs)

__all__=["analyze","load_tasks","prepare","run"]
