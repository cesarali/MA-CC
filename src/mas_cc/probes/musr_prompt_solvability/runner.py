"""Staged development selection, held-out validation, and finalization."""

from __future__ import annotations

import csv,json,shutil,subprocess
from pathlib import Path
from typing import Any,Mapping

import yaml

from mas_cc.games.relational_reasoning.data import RelationalTask,load_musr_team_allocation_task
from mas_cc.llm_runtime.providers import UniversityPricingSource
from mas_cc.musr_team_allocation_generator.io_utils import sha256_file,sha256_object,write_json_atomic

from .analysis import render_plots,report,rows,select_packet,select_prompt,summarize,write_csv
from .config import SolvabilityConfig
from .design import CallSpec,packet_definitions,phase_a,phase_b,phase_c
from .execution import execute
from .prompting import RenderedCall,render


def _git()->dict[str,Any]:
    try:
        return {"commit":subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(),"dirty":bool(subprocess.run(["git","status","--porcelain"],check=True,capture_output=True,text=True).stdout.strip())}
    except (OSError,subprocess.CalledProcessError): return {"commit":None,"dirty":None}


def load_tasks(config:SolvabilityConfig)->dict[str,RelationalTask]:
    return {task_id:load_musr_team_allocation_task(config.task_dir,task_id,population_size=config.population_size) for task_id in (*config.development_tasks,*config.heldout_tasks)}


def rendered(tasks:Mapping[str,RelationalTask],specs:tuple[CallSpec,...])->dict[str,RenderedCall]: return {spec.call_id:render(tasks[spec.task_id],spec) for spec in specs}


def _checks(config:SolvabilityConfig,tasks:Mapping[str,RelationalTask],specs:tuple[CallSpec,...],prompts:Mapping[str,RenderedCall])->list[dict[str,Any]]:
    forbidden=("skill_matrix","cooperation_matrix","candidate_scores","hidden_claim","gold_answer")
    return [
        {"check":"provider_model","passed":config.provider.type=="university" and config.provider.model=="gwdg/openai-gpt-oss-120b"},
        {"check":"development_heldout_split","passed":len(config.development_tasks)>=2 and bool(config.heldout_tasks) and not set(config.development_tasks)&set(config.heldout_tasks)},
        {"check":"task_contract","passed":all(len(task.fact_order)==27 and len(task.supporting_fact_groups or {})==9 for task in tasks.values())},
        {"check":"unique_calls","passed":len({spec.call_id for spec in specs})==len(specs)},
        {"check":"hidden_metadata_absent","passed":all(all(term not in "\n".join(m.content for m in prompt.messages) for term in forbidden) for prompt in prompts.values())},
        {"check":"budgets","passed":len(specs)<=config.max_requests and sum(p.token_estimate for p in prompts.values())<=config.max_input_tokens and len(specs)*config.provider.max_output_tokens<=config.max_output_tokens_total},
    ]


def _approval(config:SolvabilityConfig,payload:Mapping[str,Any])->str: return sha256_object({"config":config.to_dict(),"preflight":payload})


def prepare(config:SolvabilityConfig,output_dir:Path|None=None)->tuple[Path,dict[str,RelationalTask],dict[str,Any]]:
    root=Path(output_dir or config.output_dir); root.mkdir(parents=True,exist_ok=True); tasks=load_tasks(config)
    specs=phase_a(tasks,config.development_tasks,config.prompt_repetitions,config.seed); prompts=rendered(tasks,specs); checks=_checks(config,tasks,specs,prompts)
    quote=UniversityPricingSource(config.provider).fetch(config.provider.type,config.provider.model)
    if quote.status!="known" or quote.pricing is None: raise RuntimeError(f"live pricing does not permit launch: {quote.status}")
    total_calls=config.nominal_calls; payload={"probe":"musr_prompt_solvability","passed":all(c["passed"] for c in checks),"checks":checks,"calls":{"phase_a":len(specs),"phase_b":len(config.development_tasks)*3*config.packet_repetitions,"phase_c":len(config.heldout_tasks)*(2+config.population_size)*config.heldout_repetitions,"total":total_calls},"phase_a_call_plan_sha256":sha256_object([s.to_dict() for s in specs]),"estimated_input_tokens_phase_a":sum(p.token_estimate for p in prompts.values()),"maximum_output_tokens_total":total_calls*config.provider.max_output_tokens,"pricing":quote.to_dict(),"conservative_cost":quote.pricing.cost(config.max_input_tokens,min(config.max_output_tokens_total,total_calls*config.provider.max_output_tokens)).to_dict()}
    pre=root/"preflight"; pre.mkdir(parents=True,exist_ok=True); write_json_atomic(pre/"preflight.json",payload); write_json_atomic(pre/"pricing_snapshot.json",quote.to_dict()); write_json_atomic(pre/"phase_a_call_plan.json",[s.to_dict() for s in specs]); (pre/"preflight_id.txt").write_text(_approval(config,payload)+"\n",encoding="utf-8")
    (pre/"report.md").write_text(f"# Prompt solvability calibration preflight\n\n- Passed: **{payload['passed']}**\n- Phase A calls: {payload['calls']['phase_a']}\n- Phase B calls: {payload['calls']['phase_b']}\n- Phase C calls: {payload['calls']['phase_c']}\n- Total planned calls: {payload['calls']['total']}\n- Maximum output-token ceiling: {payload['maximum_output_tokens_total']:,}\n- Conservative cost: {payload['conservative_cost']['amount']} {payload['conservative_cost']['unit']}\n",encoding="utf-8")
    (root/"config.yaml").write_text(yaml.safe_dump(config.to_dict(),sort_keys=False),encoding="utf-8"); task_out=root/"tasks"; task_out.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for task_id,task in tasks.items():
        source_base,source_dist=map(Path,task.source_path.split("|",1)); shutil.copy2(source_base,task_out/f"{task_id}_base.json"); shutil.copy2(source_dist,task_out/f"{task_id}_distribution_N12.json")
        raw=json.loads(source_base.read_text()); scores=list(raw["latent"]["candidate_scores"]); ordered=sorted(scores,reverse=True); metadata.append({"task_id":task_id,"split":"development" if task_id in config.development_tasks else "heldout","gold_answer":task.correct_relation,"gold_score":scores[int(raw["gold_index"])],"second_best_score":ordered[1],"score_margin":ordered[0]-ordered[1]})
    write_csv(task_out/"task_metadata.csv",metadata); write_csv(task_out/"latent_score_metadata.csv",metadata); write_json_atomic(task_out/"full_profile_packets.json",packet_definitions(tasks))
    manifest={"schema_version":1,"probe":"musr_prompt_solvability","status":"planned","config_sha256":sha256_object(config.to_dict()),"task_hashes":{str(p.relative_to(root)):sha256_file(p) for p in task_out.glob("*.json")},"mas_cc_git":_git(),"provider":config.provider.type,"model":config.provider.model,"requested_decoding":{"temperature":config.provider.temperature,"max_output_tokens":config.provider.max_output_tokens,"transport_retries":config.provider.max_retries}}
    write_json_atomic(root/"manifest.json",manifest); (root/"README.md").write_text("# MuSR prompt solvability calibration 01\n\nSee `analysis/prompt_solvability_calibration_report.md`.\n",encoding="utf-8")
    return root,tasks,payload


def _verify_approval(config:SolvabilityConfig,root:Path,approval:Path|str|None)->None:
    approved=Path(approval).read_text().strip() if approval is not None and Path(str(approval)).is_file() else str(approval or "").strip(); expected=(root/"preflight/preflight_id.txt").read_text().strip()
    if approved!=expected: raise RuntimeError("probe run requires the matching preflight approval ID")


def _archive_examples(root:Path,phase_rows:list[dict[str,Any]])->None:
    examples=[]
    for variant in ("P0","P1","P2","P3"):
        row=next(r for r in phase_rows if r["task_id"]=="task_001" and r["prompt_variant"]==variant)
        examples.append(f"## {variant}\n\n```text\n"+"\n\n".join(f"[{m['role']}]\n{m['content']}" for m in row["messages"])+"\n```\n")
    (root/"prompt_ablation/rendered_prompt_examples.md").write_text("# Complete prompt ablation examples\n\n"+"\n".join(examples),encoding="utf-8")


async def run(config:SolvabilityConfig,output_dir:Path|None=None,*,approve_preflight:Path|str|None=None)->dict[str,Any]:
    root=Path(output_dir or config.output_dir); tasks=load_tasks(config); _verify_approval(config,root,approve_preflight)
    payload=json.loads((root/"preflight/preflight.json").read_text());
    if not payload.get("passed"): raise RuntimeError("preflight failed")
    a_specs=phase_a(tasks,config.development_tasks,config.prompt_repetitions,config.seed); a_rendered=rendered(tasks,a_specs); a_exec=await execute(config,tasks,a_specs,a_rendered,root/"prompt_ablation/raw_calls.jsonl"); a_rows=rows(root/"prompt_ablation/raw_calls.jsonl"); _archive_examples(root,a_rows)
    write_csv(root/"prompt_ablation/observation_level_results.csv",a_rows); a_summary=summarize(a_rows,("prompt_variant",)); a_task=summarize(a_rows,("task_id","prompt_variant")); write_csv(root/"prompt_ablation/summary_by_prompt.csv",a_summary); write_csv(root/"prompt_ablation/summary_by_prompt_task.csv",a_task); selected_prompt=select_prompt(a_summary); write_json_atomic(root/"prompt_ablation/selection.json",{"selected_prompt":selected_prompt,"rule":"highest pooled truth, parse >= 0.95, simpler within 0.05"})
    b_specs=phase_b(tasks,config.development_tasks,selected_prompt,config.packet_repetitions,config.seed); b_rendered=rendered(tasks,b_specs); b_exec=await execute(config,tasks,b_specs,b_rendered,root/"full_profile_ablation/raw_calls.jsonl"); b_rows=rows(root/"full_profile_ablation/raw_calls.jsonl"); write_csv(root/"full_profile_ablation/observation_level_results.csv",b_rows); b_summary=summarize(b_rows,("packet_variant",)); b_task=summarize(b_rows,("task_id","packet_variant")); write_csv(root/"full_profile_ablation/summary_by_packet.csv",b_summary); write_csv(root/"full_profile_ablation/summary_by_packet_task.csv",b_task); selected_packet=select_packet(b_summary); write_json_atomic(root/"full_profile_ablation/selection.json",{"selected_packet":selected_packet,"rule":"smallest packet with pooled truth >= 0.80 and parse >= 0.95; otherwise best observed"})
    c_specs=phase_c(tasks,config.heldout_tasks,selected_prompt,selected_packet,config.heldout_repetitions,config.population_size,config.seed); c_rendered=rendered(tasks,c_specs); c_exec=await execute(config,tasks,c_specs,c_rendered,root/"heldout_validation/raw_calls.jsonl"); c_rows=rows(root/"heldout_validation/raw_calls.jsonl"); write_csv(root/"heldout_validation/observation_level_results.csv",c_rows); c_summary=summarize(c_rows,("condition",)); c_task=summarize(c_rows,("task_id","condition")); write_csv(root/"heldout_validation/zero_private_full_summary.csv",c_summary); write_csv(root/"heldout_validation/zero_private_full_by_task.csv",c_task)
    task_meta=list(csv.DictReader((root/"tasks/task_metadata.csv").open())); examples={variant:"\n\n".join(f"[{m['role']}]\n{m['content']}" for m in next(r for r in a_rows if r["task_id"]=="task_001" and r["prompt_variant"]==variant)["messages"]) for variant in ("P0","P1","P2","P3")}; packets=json.loads((root/"tasks/full_profile_packets.json").read_text()); text=report(task_meta,a_summary,a_task,selected_prompt,b_summary,selected_packet,c_summary,examples,packets); report_path=root/"analysis/prompt_solvability_calibration_report.md"; report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(text,encoding="utf-8")
    tables=root/"analysis/tables"; write_csv(tables/"prompt_ablation_table.csv",a_summary); write_csv(tables/"full_profile_packet_table.csv",b_summary); write_csv(tables/"heldout_zero_private_full_table.csv",c_summary); render_plots(a_summary,b_summary,c_summary,root/"analysis/figures")
    execution={"phase_a":a_exec,"phase_b":b_exec,"phase_c":c_exec,"scheduled":len(a_specs)+len(b_specs)+len(c_specs),"successful":sum(x["successful"] for x in (a_exec,b_exec,c_exec))}; artifacts={str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="manifest.json"}; manifest=json.loads((root/"manifest.json").read_text()); manifest.update({"status":"complete" if execution["successful"]==execution["scheduled"] else "incomplete","selected_prompt":selected_prompt,"selected_full_profile_packet":selected_packet,"execution":execution,"artifact_hashes":artifacts}); manifest["manifest_content_sha256"]=sha256_object(manifest); write_json_atomic(root/"manifest.json",manifest)
    return {"output":str(root),"report":str(report_path),"selected_prompt":selected_prompt,"selected_packet":selected_packet,"heldout":c_summary,"execution":execution}


def analyze(config:SolvabilityConfig,output_dir:Path|None=None)->dict[str,Any]:
    root=Path(output_dir or config.output_dir); tasks=load_tasks(config); selection_a=json.loads((root/"prompt_ablation/selection.json").read_text()); selection_b=json.loads((root/"full_profile_ablation/selection.json").read_text()); selected_prompt=selection_a["selected_prompt"]; selected_packet=selection_b["selected_packet"]
    a_rows=rows(root/"prompt_ablation/raw_calls.jsonl"); b_rows=rows(root/"full_profile_ablation/raw_calls.jsonl"); c_rows=rows(root/"heldout_validation/raw_calls.jsonl"); a_summary=summarize(a_rows,("prompt_variant",)); b_summary=summarize(b_rows,("packet_variant",)); c_summary=summarize(c_rows,("condition",)); task_meta=list(csv.DictReader((root/"tasks/task_metadata.csv").open())); examples={variant:"\n\n".join(f"[{m['role']}]\n{m['content']}" for m in next(r for r in a_rows if r["task_id"]=="task_001" and r["prompt_variant"]==variant)["messages"]) for variant in ("P0","P1","P2","P3")}; packets=json.loads((root/"tasks/full_profile_packets.json").read_text()); report_path=root/"analysis/prompt_solvability_calibration_report.md"; report_path.write_text(report(task_meta,a_summary,summarize(a_rows,("task_id","prompt_variant")),selected_prompt,b_summary,selected_packet,c_summary,examples,packets),encoding="utf-8"); render_plots(a_summary,b_summary,c_summary,root/"analysis/figures")
    all_rows=(*a_rows,*b_rows,*c_rows); successful=sum(row.get("parse_success") is True for row in all_rows); full=next(row for row in c_summary if row["condition"]=="full"); zero=next(row for row in c_summary if row["condition"]=="zero"); private=next(row for row in c_summary if row["condition"]=="private"); gate_passed=float(full["truth_rate"])>=.8 and float(zero["truth_rate"])<=.5 and float(private["truth_rate"])<=.5
    artifacts={str(path.relative_to(root)):sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name!="manifest.json"}; manifest=json.loads((root/"manifest.json").read_text()); manifest.update({"status":"complete" if successful==config.nominal_calls else "incomplete","acceptance_decision":"PASS" if gate_passed else "FAIL","selected_prompt":selected_prompt,"selected_full_profile_packet":selected_packet,"execution":{"scheduled":config.nominal_calls,"successful":successful,"failed":config.nominal_calls-successful},"observed_usage":{"requests":len(all_rows),"input_tokens":sum(int((row.get("usage") or {}).get("input_tokens") or 0) for row in all_rows),"output_tokens":sum(int((row.get("usage") or {}).get("output_tokens") or 0) for row in all_rows),"transport_retries":sum(int(row.get("transport_retries") or 0) for row in all_rows)},"artifact_hashes":artifacts}); manifest.pop("manifest_content_sha256",None); manifest["manifest_content_sha256"]=sha256_object(manifest); write_json_atomic(root/"manifest.json",manifest)
    return {"output":str(root),"report":str(report_path),"selected_prompt":selected_prompt,"selected_packet":selected_packet,"heldout":c_summary,"acceptance_decision":"PASS" if gate_passed else "FAIL"}


__all__=["analyze","load_tasks","prepare","run"]
