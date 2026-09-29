"""Forced next-day reduced-theory predictions; continuous outputs remain continuous."""
import json
import numpy as np
from .core import (KernelBank,validate_state,validate_board,controlled_board,
                   advance_day,new_diagnostics,action_propensity)
from .io import read_config,load_snapshots,prepare_output,write_csv,manifest

def branch_snapshot(p,s,snapshot,pairs,seed):
    if not isinstance(pairs,int) or isinstance(pairs,bool) or pairs<2:raise ValueError('pairs must be >=2')
    state=validate_state([snapshot['x'],snapshot['kappa_plus'],snapshot['kappa_minus']])
    peers=validate_board(snapshot['peer_counts'],p.N);bank=KernelBank(p,s)
    rows=[];diag=[];differences=[]
    for pair,child in enumerate(np.random.SeedSequence(seed).spawn(pairs)):
        targets=[]
        for action in (0,1):
            rng=np.random.default_rng(child)  # common Gaussian streams; independent seed pairs
            diagnostics=new_diagnostics();front=controlled_board(peers,action,p)
            end,_,_=advance_day(state,front,p,s,bank,rng,diagnostics)
            target=float(end[0] if p.controller_target==1 else 1-end[0]);targets.append(target)
            rows.append({'snapshot_id':snapshot['snapshot_id'],'pair':pair,'forced_action':action,
                         'pre_action_round':snapshot['round'],'outcome_round':snapshot['round']+1,
                         'x_next':float(end[0]),'target_next':target,'kappa_plus_next':float(end[1]),
                         'kappa_minus_next':float(end[2])})
            diag.append({'snapshot_id':snapshot['snapshot_id'],'pair':pair,'forced_action':action,**diagnostics})
        differences.append(targets[1]-targets[0])
    delta=np.array(differences);se=float(delta.std(ddof=1)/np.sqrt(pairs));m=float(delta.mean())
    result={'snapshot_id':snapshot['snapshot_id'],'pre_action_round':snapshot['round'],
            'pairs':pairs,'a_sensor_averaged':action_propensity(peers,p),
            'chi_target':m,'se_paired':se,'ci95_low':m-1.96*se,'ci95_high':m+1.96*se}
    return rows,result,diag

def export_branches(config,snapshot_file,outdir,pairs=256):
    p,s,provenance=read_config(config);snaps=load_snapshots(snapshot_file,'pre_action')
    allrows=[];summary=[];diagnostics=[]
    for i,snap in enumerate(snaps):
        seed=np.random.SeedSequence([s.seed,991,i]).generate_state(4).tolist()
        rows,stat,diag=branch_snapshot(p,s,snap,pairs,seed)
        allrows.extend(rows);summary.append(stat);diagnostics.extend(diag)
    out=prepare_output(outdir)
    write_csv(out/'branch_samples.csv',allrows);write_csv(out/'branch_summary.csv',summary)
    write_csv(out/'diagnostics.csv',diagnostics)
    (out/'manifest.json').write_text(json.dumps(manifest(p,s,provenance,{
       'branch_pairs':pairs,'information_quantities':'not estimated; choose matched output bins and estimator externally',
       'identity_randomness':'sign-level closure integrates iid fact identities; no retained individual controller IDs'}),indent=2))
    from pathlib import Path
    (out/'config.json').write_text(Path(config).read_text())
    (out/'snapshots.json').write_text(Path(snapshot_file).read_text())
    return out
