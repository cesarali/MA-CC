"""Strict configuration/state contract and trajectory exports."""
from __future__ import annotations
import csv,json,platform,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .core import (Parameters,Solver,strict_dataclass,validate_state,validate_board,
                   KernelBank,advance_day,feedback,new_diagnostics,VERSION,CATEGORIES)

METRICS=['x','target_share','kappa_plus','kappa_minus','active_plus','active_minus',
         'peer_target_share','action_next','effective_action_next','posts_next']

def read_config(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema_version')!=1:raise ValueError('Require schema_version=1')
    if set(data)-{'schema_version','parameters','solver','provenance'}:raise ValueError('Unknown top-level config key')
    p=strict_dataclass(Parameters,data['parameters']);s=strict_dataclass(Solver,data.get('solver',{}))
    return p,s,data.get('provenance',{})

def load_snapshots(path,kind='initial'):
    d=json.loads(Path(path).read_text())
    if d.get('schema_version')!=1 or d.get('kind')!=kind:raise ValueError(f'Require schema_version=1, kind={kind}')
    records=d.get('snapshots',[])
    if not records:raise ValueError('No snapshots')
    seen=set()
    for r in records:
        allowed={'snapshot_id','round','x','kappa_plus','kappa_minus',
                 'front_page_counts' if kind=='initial' else 'peer_counts'}
        if set(r)!=allowed:raise ValueError(f'Snapshot keys must be exactly {sorted(allowed)}')
        if not isinstance(r['snapshot_id'],str) or not r['snapshot_id'] or r['snapshot_id'] in seen:
            raise ValueError('snapshot_id must be a unique nonempty string')
        if isinstance(r['round'],bool) or not isinstance(r['round'],int) or r['round']<0:
            raise ValueError('snapshot round must be nonnegative integer')
        seen.add(r['snapshot_id']);validate_state([r['x'],r['kappa_plus'],r['kappa_minus']])
        validate_board(r['front_page_counts' if kind=='initial' else 'peer_counts'])
    if kind=='initial' and len({r['round'] for r in records})!=1:
        raise ValueError('Initial blocks must share a round origin for ensemble summaries')
    return records

def write_csv(path,rows):
    if not rows:raise ValueError('Cannot write empty table')
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows):
    """Equal initial-block weighting. Replica means are nested within blocks.
    SE across blocks when >=2; across replicas when exactly one block.
    """
    result=[]
    for round_ in sorted({r['round'] for r in rows}):
        rr=[r for r in rows if r['round']==round_]
        for metric in METRICS:
            available=[r for r in rr if r.get(metric) not in ('',None)]
            if not available:continue
            by={}
            for r in available:by.setdefault(r['initial_id'],[]).append(float(r[metric]))
            blocks=np.array([np.mean(v) for v in by.values()])
            units=blocks if len(blocks)>1 else np.array(next(iter(by.values())))
            mean=float(blocks.mean());se=float(units.std(ddof=1)/np.sqrt(len(units))) if len(units)>1 else None
            result.append({'round':round_,'metric':metric,'mean':mean,'se':se if se is not None else '',
                           'ci95_low':mean-1.96*se if se is not None else '',
                           'ci95_high':mean+1.96*se if se is not None else '',
                           'n_blocks':len(blocks),'n_paths':len(available),
                           'uncertainty_unit':'initial_block' if len(blocks)>1 else 'conditional_replica'})
    return result

def manifest(p,s,provenance,extra=None):
    out={'schema_version':1,'runner_version':VERSION,'theory_level':'reduced_binomial_langevin',
         'parameters':asdict(p),'solver':asdict(s),'provenance':provenance,
         'category_order':CATEGORIES,'python':sys.version,'numpy':np.__version__,
         'clock':'round d row = after night/day d; action_next=U_d affects round d+1',
         'initial_state':'explicit empirical projection and supplied front page; no initialization generator',
         'night':'Gaussian aggregate moment closure, exact deterministic endpoints rho=0/1',
         'board':'independent multinomial draw from time-integrated emission probabilities',
         'identities':'iid uniform sign identities conditional on sampled categories; actual board IDs not retained',
         'coverage':'independent binomial fact counts, independent of old vote',
         'sampling_scope':'without replacement is exact for category counts, not identity-resolved acquisition',
         'outputs':'raw continuous x; never rounded into a fabricated finite-count SDE output law',
         'weighting':'equal initial blocks, equal replicas within block',
         'limitations':['not full heterogeneous class-density theory',
                        'population/board noise correlation omitted',
                        'boundary projection can bias results',
                        'daytime_noise=false retains stochastic nights, board and feedback']}
    if extra:out.update(extra)
    return out

def state_row(p,initial_id,replica,round_,state,front,peers=None,action=None,y=None,prob=None,night=None):
    x,kp,km=state;row={'model':'theory','initial_id':initial_id,'replica':replica,'round':round_,
       'x':float(x),'target_share':float(x if p.controller_target==1 else 1-x),
       'kappa_plus':float(kp),'kappa_minus':float(km),'active_plus':float(p.F_plus*kp),
       'active_minus':float(p.F_minus*km),'peer_target_share':'','action_next':'',
       'effective_action_next':'','posts_next':'','sensor_count':'','P_U1_given_Y':'',
       'night_kappa_plus':'','night_kappa_minus':''}
    for j,c in enumerate(CATEGORIES):
        row['front_next_'+c]=int(front[j]);row['peer_'+c]='' if peers is None else int(peers[j])
    if peers is not None:
        row['peer_target_share']=float((peers[:2].sum() if p.controller_target==1 else peers[2:].sum())/p.N)
    if action is not None:
        row.update(action_next=action,effective_action_next=int(action==1 and p.b>0),posts_next=p.b*action,
                   sensor_count=y,P_U1_given_Y=prob)
    if night is not None:row.update(night_kappa_plus=float(night[1]),night_kappa_minus=float(night[2]))
    return row

def run_ensemble(p,s,initials):
    """Return rows and diagnostics; each record gets independent downstream replicas."""
    bank=KernelBank(p,s);rows=[];diagnostics=[]
    seeds=np.random.SeedSequence(s.seed).spawn(len(initials)*s.replicas_per_initial)
    for block,init in enumerate(initials):
        for rep in range(s.replicas_per_initial):
            seed=seeds[block*s.replicas_per_initial+rep];rng=np.random.default_rng(seed)
            state=validate_state([init['x'],init['kappa_plus'],init['kappa_minus']])
            front=validate_board(init['front_page_counts']);diag=new_diagnostics()
            rows.append(state_row(p,init['snapshot_id'],rep,init['round'],state,front))
            for day in range(1,s.rounds+1):
                state,peers,night=advance_day(state,front,p,s,bank,rng,diag)
                front,y,u,prob=feedback(peers,p,rng)
                rows.append(state_row(p,init['snapshot_id'],rep,init['round']+day,state,front,peers,u,y,prob,night))
            diagnostics.append({'initial_id':init['snapshot_id'],'replica':rep,
                                'seed_spawn_key':'.'.join(map(str,seed.spawn_key)),
                                'day_proposals':s.rounds*s.substeps,'night_proposals':s.rounds,**diag})
    return rows,diagnostics

def prepare_output(path):
    p=Path(path)
    if p.exists() and any(p.iterdir()):raise ValueError(f'Output directory is not empty: {p}; choose a new run directory')
    p.mkdir(parents=True,exist_ok=True);return p

def export_run(config,initial_file,outdir):
    p,s,provenance=read_config(config);initials=load_snapshots(initial_file)
    rows,diag=run_ensemble(p,s,initials);out=prepare_output(outdir)
    write_csv(out/'trajectories.csv',rows);write_csv(out/'summary.csv',summarize(rows));write_csv(out/'diagnostics.csv',diag)
    (out/'manifest.json').write_text(json.dumps(manifest(p,s,provenance),indent=2))
    (out/'initials.json').write_text(json.dumps({'schema_version':1,'kind':'initial','snapshots':initials},indent=2))
    (out/'config.json').write_text(Path(config).read_text())
    return out
