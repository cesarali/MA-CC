"""Additional v4 entropy and propensity-response diagnostics from saved rounds.

All information estimates below are observational counts in bits. The
start-of-round rows define the pre-action state and the t+1 outcome.
"""
from __future__ import annotations

from collections import Counter
import json
import math

import numpy as np
import pandas as pd

def _entropy(values):
    counts=Counter(values)
    total=sum(counts.values())
    if not total:return math.nan
    return -sum((n/total)*math.log2(n/total) for n in counts.values())


def _conditional_entropy(outcomes,conditions):
    if len(outcomes)!=len(conditions):raise ValueError("entropy input lengths differ")
    if not outcomes:return math.nan
    pairs=Counter(zip(conditions,outcomes))
    groups=Counter(conditions)
    total=len(outcomes)
    return -sum((n/total)*math.log2(n/groups[condition])
                for (condition,_),n in pairs.items())


def extra_information(rounds:pd.DataFrame)->pd.DataFrame:
    """Entropy decomposition and action MI for one physical cell."""
    d=rounds.loc[rounds.time_stage=="pre_action"].copy()
    if d.empty:raise ValueError("no pre-action rows")
    k=d.target_count.astype(int).tolist()
    kp=d.next_target_count.astype(int).tolist()
    u=d.controller_U.astype(int).tolist()
    levels={
        "none":[()] * len(d),
        "memory_exact":[tuple(json.loads(s)) for s in d.memory_histogram_json],
        "kappa_plus_3bin":[(min(2,int(3*x)),) for x in d.kappa_plus],
        "phi_plus_3bin":[(min(2,int(3*x)),) for x in d.phi_plus],
        "susceptible_3bin":[(min(2,int(3*(1-x))),) for x in d.phi_plus],
        "kappa_phi_joint_4bin":[(min(3,int(4*x)),min(3,int(4*y)))
                                 for x,y in zip(d.kappa_plus,d.phi_plus)],
        "joint_r_s_vote_exact":[tuple(tuple((item["r"],item["s"],item["vote"],item["count"]))
                                   for item in json.loads(s)) for s in d.epistemic_joint_histogram_json],
        "overloaded_share_3bin":[(min(2,int(3*x)),) for x in d.overloaded_share],
    }
    rows=[]
    def add(metric,value,level="none",details=""):
        rows.append({"metric":metric,"conditioning":level,"estimate_bits":value,
                     "events":len(d),"independent_episodes":d.seed.nunique(),
                     "window":"all_60_transitions","details":details})
    add("H_K",_entropy(k))
    add("H_U",_entropy(u))
    add("H_U_given_sensor_full_known_propensity",float(np.mean(
        [-p*math.log2(p)-(1-p)*math.log2(1-p) for p in d.controller_p_act])),
        details="mean binary entropy of known e_t; additionally conditions on sampled votes")
    add("I_U_Knext",_entropy(u)-_conditional_entropy(u,kp))
    for level,extra in levels.items():
        conditions=list(zip(k,extra))
        action_condition=list(zip(conditions,u))
        h_u=_conditional_entropy(u,conditions)
        h0=_conditional_entropy(kp,conditions)
        h1=_conditional_entropy(kp,action_condition)
        add("H_U_given_K_E",h_u,level)
        add("H_Knext_given_K_E",h0,level)
        add("H_Knext_given_K_E_U",h1,level)
        add("I_U_Knext_given_K_E_entropy_difference",h0-h1,level,
            "same unsmoothed plug-in count table; check against shared CMI")
        support=pd.DataFrame({"condition":list(map(str,conditions)),"action":u})
        counts=support.groupby("condition").action.nunique()
        add("dual_action_state_fraction",float((counts==2).mean()),level)
        add("conditioning_state_count",float(len(counts)),level)
    return pd.DataFrame(rows)


def propensity_response(rounds:pd.DataFrame)->pd.DataFrame:
    """Horvitz-Thompson action contrast with whole-episode standard errors."""
    d=rounds.sort_values(["seed","round"]).copy()
    active=d.loc[d.time_stage=="pre_action"].copy()
    active["weight"]=(active.controller_U/active.controller_p_act-
                      (1-active.controller_U)/(1-active.controller_p_act))
    rows=[]
    for h in (1,2,3):
        active[f"future_{h}"]=d.groupby("seed").target_share.shift(-h).loc[active.index]
        eligible=active.loc[active[f"future_{h}"].notna()].copy()
        eligible["contribution"]=eligible.weight*(eligible[f"future_{h}"]-eligible.target_share)
        per_episode=eligible.groupby("seed").contribution.mean()
        n=len(per_episode);mean=float(per_episode.mean())
        se=float(per_episode.std(ddof=1)/math.sqrt(n)) if n>1 else math.nan
        rows.append({"metric":f"tau_ipw_lag_{h}","estimate":mean,"se_across_episodes":se,
                     "ci_low":mean-1.96*se,"ci_high":mean+1.96*se,
                     "independent_episodes":n,"events":len(eligible),
                     "window":f"t=0..{60-h}","units":"target_fraction"})
    eligible=active.loc[active.target_share<1].copy()
    eligible["delta"]=eligible.next_target_share-eligible.target_share
    numerator=float(np.sum(eligible.weight*eligible.delta))
    mass=float(np.sum(1-eligible.target_share))
    rows.append({"metric":"available_susceptibility_ratio","estimate":numerator/mass if mass>0 else math.nan,
                 "se_across_episodes":math.nan,"ci_low":math.nan,"ci_high":math.nan,
                 "independent_episodes":eligible.seed.nunique(),"events":len(eligible),
                 "excluded_saturated_events":len(active)-len(eligible),
                 "numerator":numerator,"denominator_available_mass":mass,
                 "window":"all_60_transitions","units":"target_fraction_per_available_mass"})
    return pd.DataFrame(rows)


def state_local_response(rounds:pd.DataFrame)->pd.DataFrame:
    """Observed action contrasts in exact K and declared pre-action evidence strata."""
    d=rounds.loc[rounds.time_stage=="pre_action"].copy()
    d["delta"]=d.next_target_share-d.target_share
    d["available_contribution"]=np.where(d.target_share<1,
        (d.controller_U/d.controller_p_act-(1-d.controller_U)/(1-d.controller_p_act)) *
        d["delta"]/(1-d.target_share).replace(0,np.nan),np.nan)
    evidence={
        "none":["all"]*len(d),
        "memory_exact":d.memory_histogram_json.astype(str).tolist(),
        "kappa_plus_3bin":[str(min(2,int(3*x))) for x in d.kappa_plus],
        "phi_plus_3bin":[str(min(2,int(3*x))) for x in d.phi_plus],
        "susceptible_3bin":[str(min(2,int(3*(1-x)))) for x in d.phi_plus],
        "kappa_phi_joint_4bin":[f"{min(3,int(4*x))}:{min(3,int(4*y))}"
                                 for x,y in zip(d.kappa_plus,d.phi_plus)],
        "joint_r_s_vote_exact":d.epistemic_joint_histogram_json.astype(str).tolist(),
        "overloaded_share_3bin":[str(min(2,int(3*x))) for x in d.overloaded_share],
    }
    output=[]
    for level,states in evidence.items():
        frame=d[["target_count","controller_U","delta","available_contribution"]].copy()
        frame["evidence_state"]=states
        for (k,state),group in frame.groupby(["target_count","evidence_state"],sort=False):
            silence=group.loc[group.controller_U==0,"delta"]
            act=group.loc[group.controller_U==1,"delta"]
            identified=bool(len(silence) and len(act))
            output.append({"conditioning":level,"target_count":int(k),
                "target_share":int(k)/int(d.N.iloc[0]),"evidence_state":state,
                "events":len(group),"independent_episodes":d.seed.nunique(),
                "silence_events":len(silence),"action_events":len(act),
                "empirical_action_rate":float(group.controller_U.mean()),
                "chi_target_fraction":float(act.mean()-silence.mean()) if identified else math.nan,
                "chi_identified":identified,
                "available_ipw_local_mean":float(group.available_contribution.mean())
                    if group.available_contribution.notna().any() else math.nan,
                "available_eligible_events":int(group.available_contribution.notna().sum()),
                "window":"all_60_transitions"})
    return pd.DataFrame(output)
