"""Version 4 Santa Fe microscopic process with a same-round live board.

Round row t is the pre-action state X_t and the gate at t; row t+1 is its
outcome. The terminal row has no gate. Separate random streams make a virtual
B=0 gate unable to change the physical trajectory.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math

import numpy as np

from .game import binary_entropy, make_fact_weights, sigmoid
from .state import EpisodeResult, Message
from .v3_game import _counts, _coverage


VERSION="santa_fe_live_board_v4"


@dataclass
class LiveAgent:
    vote: int
    active_facts: set[int]=field(default_factory=set)
    historical_facts: set[int]=field(default_factory=set)


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      default=lambda x: x.item() if isinstance(x,np.generic) else TypeError(type(x)))


def overload_factor(load:int,threshold:int,alpha:float)->float:
    return 1.0/(1.0+alpha*max(0,load-threshold))


def _agent_rows(agents,weights):
    return [{"agent_id":i,"vote":int(a.vote),
        "active_fact_ids":sorted(a.active_facts),
        "historical_fact_ids":sorted(a.historical_facts),
        "r":int(sum(weights[f]==1 for f in a.active_facts)),
        "s":int(sum(weights[f]==-1 for f in a.active_facts))} for i,a in enumerate(agents)]


def _message_row(message,weights,round_index,slot):
    return {"message_id":message.message_id,"author":int(message.author),
        "vote":int(message.vote),"fact_id":message.fact_id,
        "fact_sign":int(weights[message.fact_id]) if message.fact_id is not None else 0,
        "source":"controller" if message.is_controller else "peer",
        "born_round":round_index,"born_slot":slot,"expires_before_round":round_index+1}


def _histogram(agents,weights):
    h=Counter((sum(weights[f]==1 for f in a.active_facts),
               sum(weights[f]==-1 for f in a.active_facts),int(a.vote)) for a in agents)
    return [{"r":r,"s":s,"vote":v,"count":n} for (r,s,v),n in sorted(h.items())]


def _summary(agents,weights,target,overload_threshold):
    coverage=_coverage(agents,weights,target)
    plus=np.array([sum(weights[f]==1 for f in a.active_facts) for a in agents])
    minus=np.array([sum(weights[f]==-1 for f in a.active_facts) for a in agents])
    coverage.update(phi_plus=float(np.mean(plus==int(sum(weights==1)))),
        overloaded_share=float(np.mean(plus+minus>overload_threshold)),
        mean_active_load=float(np.mean(plus+minus)),
        memory_histogram_json=_json(np.bincount(plus,minlength=int(sum(weights==1))+1).tolist()),
        epistemic_joint_histogram_json=_json(_histogram(agents,weights)))
    return coverage


def _sample(messages,capacity,rng):
    n=min(capacity,len(messages))
    return [messages[int(i)] for i in rng.choice(len(messages),size=n,replace=False)] if n else []


def run_live_board_episode(game,seed:int)->EpisodeResult:
    p=game.p
    if p.model_version!=VERSION:raise ValueError("live-board runner requires v4")
    # init, sensor, gate, forgetting, focal, read, vote, peer post, controller fact
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(9)]
    init,sensor,gate,forget,focal,read,vote,post,ctrl=streams
    weights=make_fact_weights(p.F,p.truth_fact_fraction,init,n_truth=p.F_plus)
    slots=np.repeat(np.arange(p.F),p.initial_fact_redundancy)
    if len(slots)!=p.N:raise ValueError("one initial fact per agent required")
    init.shuffle(slots)
    agents=[]
    for fact in slots:
        fact=int(fact);e=float(weights[fact]);prob=sigmoid(p.beta_evidence*e)
        agents.append(LiveAgent(1 if vote.random()<prob else -1,{fact},{fact}))
    initial_states=_agent_rows(agents,weights)
    target_pool=[int(i) for i in np.flatnonzero(weights==p.controller_target)]
    if not target_pool:raise ValueError("controller fact pool is empty")
    rounds=[];micro=[];previous_board=[]
    for t in range(p.rounds):
        start=_agent_rows(agents,weights)
        start_target=int(sum(a.vote==p.controller_target for a in agents))
        sensed_ids=[int(i) for i in sensor.choice(p.N,size=min(p.N,round(p.sensing_fraction*p.N)),replace=False)]
        q_c=len(sensed_ids)
        if q_c<1:raise ValueError("controller sample must be positive")
        y=sum(agents[i].vote==p.controller_target for i in sensed_ids)
        observed=y/q_c;e=sigmoid(p.policy_beta*(p.policy_threshold-observed))
        u=int(gate.random()<e)
        expired=[_message_row(m,weights,t-1,
                 -1 if m.is_controller else int(m.message_id.rsplit("-",1)[1]))
                 for m in previous_board]
        forgotten=[]
        for i,a in enumerate(agents):
            lost=[]
            for fact in sorted(a.active_facts):
                if forget.random()>=p.rho:lost.append(fact)
            a.active_facts.difference_update(lost)
            forgotten.append({"agent_id":i,"lost_fact_ids":lost})
        post_forgetting=_agent_rows(agents,weights)
        board=[];controller=[]
        if u and p.budget:
            for j in range(p.budget):
                fact=int(ctrl.choice(target_pool))
                message=Message(p.N,p.controller_target,fact,True,f"r{t}-controller-{j}")
                board.append(message);controller.append(message)
        controller_rows=[_message_row(m,weights,t,-1) for m in controller]
        for slot in range(p.N):
            who=int(focal.integers(p.N));a=agents[who]
            eligible=[m for m in board if m.author!=who]
            seen=_sample(eligible,p.q,read)
            before=sorted(a.active_facts);history_before=set(a.historical_facts)
            incoming={m.fact_id for m in seen if m.fact_id is not None}
            novel=sorted(incoming-history_before)
            reactivated=sorted((incoming&history_before)-a.active_facts)
            a.active_facts.update(incoming);a.historical_facts.update(incoming)
            r=sum(weights[f]==1 for f in a.active_facts)
            s=len(a.active_facts)-r
            evidence=(r-s)/(r+s) if r+s else 0.0
            social=float(np.mean([m.vote for m in seen])) if seen else 0.0
            g=overload_factor(r+s,p.overload_threshold,p.overload_alpha)
            logit=g*(p.beta_evidence*evidence+p.beta_social*social)
            prob=sigmoid(logit);old_vote=a.vote
            a.vote=1 if vote.random()<prob else -1
            aligned=sorted(f for f in a.active_facts if weights[f]==a.vote)
            posted_fact=int(post.choice(aligned)) if aligned else None
            message=Message(who,a.vote,posted_fact,False,f"r{t}-peer-{slot}")
            board.append(message)
            micro.append({"round":t,"slot":slot,"focal":who,
                "vote_before":old_vote,"vote_after":a.vote,
                "facts_before_json":_json(before),"facts_after_json":_json(sorted(a.active_facts)),
                "history_before_json":_json(sorted(history_before)),
                "history_after_json":_json(sorted(a.historical_facts)),
                "new_fact_ids_json":_json(novel),"reactivated_fact_ids_json":_json(reactivated),
                "sampled_message_ids_json":_json([m.message_id for m in seen]),
                "sampled_message_authors_json":_json([m.author for m in seen]),
                "sampled_fact_ids_json":_json([m.fact_id for m in seen]),
                "eligible_message_count":len(eligible),"live_board_size_before":len(board)-1,
                "q_effective":len(seen),"sampled_controller_messages":sum(m.is_controller for m in seen),
                "active_positive_count":r,"active_negative_count":s,"active_load":r+s,
                "overload_factor":g,"evidence_signal":evidence,"social_signal":social,
                "vote_logit":logit,"p_vote_plus":prob,
                "posted_message_json":_json(_message_row(message,weights,t,slot)),
                "posted_message_id":message.message_id,"posted_fact_id":posted_fact,
                "controller_U":u,"controller_p_act":e})
        end=_agent_rows(agents,weights)
        truth_share=game.truth_share([LiveAgent(a["vote"]) for a in start])
        row={"round":t,"time_stage":"pre_action",
            "initialization_id":seed,"fact_weights_json":_json(weights.tolist()),
            "initial_agent_states_json":_json(initial_states) if t==0 else None,
            "agent_states_json":_json(start),
            "post_forgetting_agent_states_json":_json(post_forgetting),
            "post_agent_states_json":_json(end),
            "forgotten_fact_ids_json":_json(forgotten),
            "expired_board_json":_json(expired),
            "controller_board_json":_json(controller_rows),
            "live_board_end_json":_json([_message_row(m,weights,t,
                -1 if m.is_controller else int(m.message_id.rsplit("-",1)[1])) for m in board]),
            "B_total_counts_json":_json(_counts(board,weights)),
            "truth_share":truth_share,"target_share":start_target/p.N,
            "target_count":start_target,
            "next_truth_share":game.truth_share(agents),
            "next_target_share":game.target_share(agents),
            "next_target_count":sum(a.vote==p.controller_target for a in agents),
            "vote_entropy_bits":binary_entropy(truth_share),
            "controller_U":u,"controller_effective_U":int(u and p.budget>0),
            "controller_p_act":e,"controller_observed_target_share":observed,
            "controller_sensed_messages":q_c,"controller_sensor_target_count":y,
            "sensor_agent_ids_json":_json(sensed_ids),
            "sensor_source_target_count":start_target,
            "controller_messages_current_board":len(controller),
            "controller_requested_budget":p.budget,"budget_used":len(controller),
            "unique_controller_fact_ids":len({m.fact_id for m in controller}),
            "controller_target":p.controller_target,
            "board_clock":"live_board","action_outcome_lag":1,
            **_summary([LiveAgent(a["vote"],set(a["active_fact_ids"]),set(a["historical_fact_ids"])) for a in start],
                       weights,p.controller_target,p.overload_threshold)}
        rounds.append(row)
        previous_board=board
    # Terminal state is retained for direct joins, with no fictitious action.
    truth=game.truth_share(agents)
    rounds.append({"round":p.rounds,"time_stage":"terminal","initialization_id":seed,
        "fact_weights_json":_json(weights.tolist()),"agent_states_json":_json(_agent_rows(agents,weights)),
        "truth_share":truth,"target_share":game.target_share(agents),
        "target_count":sum(a.vote==p.controller_target for a in agents),
        "vote_entropy_bits":binary_entropy(truth),"controller_U":None,
        "controller_effective_U":None,"controller_p_act":math.nan,
        "budget_used":0,"controller_target":p.controller_target,
        "board_clock":"live_board",**_summary(agents,weights,p.controller_target,p.overload_threshold)})
    return EpisodeResult(seed=seed,params={**asdict(p),"budget":p.budget},
                         fact_weights=weights.tolist(),rounds=rounds,micro=micro)
