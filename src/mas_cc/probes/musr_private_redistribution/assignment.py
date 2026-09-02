"""Deterministic six-card assignments with controlled latent breadth."""

from __future__ import annotations
import itertools,json,random,statistics
from collections import Counter
from collections.abc import Mapping,Sequence
from typing import Any
from pathlib import Path
from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.musr_team_allocation_generator.io_utils import sha256_object
from mas_cc.musr_team_allocation_generator.schemas import LatentProblem
from mas_cc.musr_team_allocation_generator.validate import agent_can_certify_unique_allocation,candidate_score_terms

REGIMES=("NAT","R2","R3","R4")

def candidate_terms(task:RelationalTask)->dict[str,frozenset[str]]:
    base=json.loads(Path(task.source_path.split("|",1)[0]).read_text(encoding="utf-8"))
    problem=LatentProblem.from_dict(base["latent"])
    return {f"ALLOCATION_{i}":candidate_score_terms(problem,i) for i in range(3)}

def _holder_targets(k:int)->dict[str,int]:
    values=[3]*6+[2]*3 if k==2 else [4]*9 if k==3 else [6]*3+[5]*6
    return {f"slot_{i}":v for i,v in enumerate(values)}

def _latent_sets(task:RelationalTask,k:int,seed:Seed)->list[tuple[str,...]]:
    latent=sorted((task.supporting_fact_groups or {}).keys()); target_values=list(_holder_targets(k).values()); rng=seed.create_random(); rng.shuffle(target_values); targets=dict(zip(latent,target_values,strict=True)); terms=set(candidate_terms(task).values())
    combinations=[combo for combo in itertools.combinations(latent,k) if not(k==4 and frozenset(combo) in terms)]
    rng.shuffle(combinations); remaining=targets.copy(); chosen=[]
    def search(agent:int)->bool:
        if agent==12: return all(v==0 for v in remaining.values())
        candidates=[c for c in combinations if all(remaining[x]>0 for x in c)]
        candidates.sort(key=lambda c:(-sum(remaining[x] for x in c),sum(sum(x in old for x in c) for old in chosen),sha256_object([int(seed),agent,c])))
        for combo in candidates:
            if any(remaining[x]>12-agent for x in latent if x not in combo): continue
            for x in combo: remaining[x]-=1
            chosen.append(combo)
            if search(agent+1): return True
            chosen.pop()
            for x in combo: remaining[x]+=1
        return False
    if not search(0): raise RuntimeError(f"no valid R{k} assignment")
    return chosen

def build_assignment(task:RelationalTask,regime:str,seed:int)->dict[str,Any]:
    if regime=="NAT": holdings={agent:list(task.known_facts(agent)) for agent in task.agent_ids}
    else:
        k=int(regime[1]); stream=Seed(seed).derive(f"{task.task_id}:{regime}:latent-assignment:v1"); latent_sets=_latent_sets(task,k,stream); groups={key:list(value) for key,value in (task.supporting_fact_groups or {}).items()}; holdings={}
        branch_cursor=Counter(); bonus_cursor=Counter()
        for index,(agent,latents) in enumerate(zip(task.agent_ids,latent_sets,strict=True)):
            cards=[]
            if k==2:
                for latent in latents: cards.extend(groups[latent])
            elif k==3:
                for latent in latents:
                    start=branch_cursor[latent]%3; cards.extend((groups[latent][start],groups[latent][(start+1)%3])); branch_cursor[latent]+=2
            else:
                for latent in latents:
                    cards.append(groups[latent][branch_cursor[latent]%3]); branch_cursor[latent]+=1
                bonus=list(latents); random.Random(int(stream.derive(f"bonus:{index}"))).shuffle(bonus)
                for latent in bonus[:2]: cards.append(groups[latent][branch_cursor[latent]%3]); branch_cursor[latent]+=1; bonus_cursor[latent]+=1
            holdings[agent]=[card for card in task.fact_order if card in set(cards)]
    return assignment_payload(task,regime,holdings,seed)

def view_diagnostics(task:RelationalTask,agent:str,cards:Sequence[str])->dict[str,Any]:
    groups=task.supporting_fact_groups or {}; card_group={card:latent for latent,ids in groups.items() for card in ids}; latents={card_group[c] for c in cards}; terms=candidate_terms(task); fractions={key:len(latents&value)/len(value) for key,value in terms.items()}; full={key:value<=latents for key,value in terms.items()}
    return {"task_id":task.task_id,"agent_id":agent,"num_cards":len(cards),"num_latent_values":len(latents),"latent_fraction":len(latents)/9,"latent_ids":"|".join(sorted(latents)),"evidence_ids":"|".join(cards),"redundancy_profile":"|".join(f"{key}:{sum(card_group[c]==key for c in cards)}" for key in sorted(latents)),**{f"{key.lower()}_term_fraction":fractions[key] for key in sorted(terms)},**{f"covers_all_terms_{key.lower()}":full[key] for key in sorted(terms)},"num_fully_scoreable_allocations":sum(full.values())}

def assignment_payload(task:RelationalTask,regime:str,holdings:Mapping[str,Sequence[str]],seed:int)->dict[str,Any]:
    diagnostics=[view_diagnostics(task,agent,cards) for agent,cards in holdings.items()]; groups=task.supporting_fact_groups or {}; holder=Counter(); card_holder=Counter()
    for cards in holdings.values():
        latent={next(key for key,ids in groups.items() if card in ids) for card in cards}; holder.update(latent); card_holder.update(cards)
    payload={"schema_version":1,"algorithm_version":"latent_breadth_six_cards_v1","root_seed":seed,"derived_seed":int(Seed(seed).derive(f"{task.task_id}:{regime}:latent-assignment:v1")),"task_id":task.task_id,"regime":regime,"target_latents_per_agent":None if regime=="NAT" else int(regime[1]),"target_cards_per_agent":None if regime=="NAT" else 6,"candidate_score_terms":{key:sorted(value) for key,value in candidate_terms(task).items()},"agent_assignments":{key:list(value) for key,value in holdings.items()},"latent_holder_counts":dict(sorted(holder.items())),"card_holder_counts":dict(sorted(card_holder.items())),"diagnostics":diagnostics}
    payload["assignment_sha256"]=sha256_object(payload); return payload

def structural_summary(payload:Mapping[str,Any])->dict[str,Any]:
    rows=payload["diagnostics"]; cards=[r["num_cards"] for r in rows]; latent=[r["num_latent_values"] for r in rows]; holders=list(payload["latent_holder_counts"].values())
    return {"task_id":payload["task_id"],"regime":payload["regime"],"mean_cards_per_agent":statistics.mean(cards),"std_cards_per_agent":statistics.pstdev(cards),"min_cards":min(cards),"max_cards":max(cards),"mean_latent_values_per_agent":statistics.mean(latent),"std_latent_values_per_agent":statistics.pstdev(latent),"min_latent_values":min(latent),"max_latent_values":max(latent),"global_9_of_9":len(holders)==9,"min_holders_per_value":min(holders),"max_holders_per_value":max(holders),"agents_fully_scoring_all_candidates":sum(r["num_fully_scoreable_allocations"]==3 for r in rows),"agents_fully_scoring_any_candidate":sum(r["num_fully_scoreable_allocations"]>0 for r in rows),"fully_scoreable_candidate_incidences":sum(r["num_fully_scoreable_allocations"] for r in rows)}

__all__=["REGIMES","assignment_payload","build_assignment","candidate_terms","structural_summary","view_diagnostics"]
