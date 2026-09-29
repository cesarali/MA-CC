"""Overlay explicitly adapted simulation exports; never guess repository schemas."""
import csv,json
from pathlib import Path
import numpy as np
from .core import Parameters,strict_dataclass
from .io import summarize,prepare_output,write_csv

def read_rows(path,p):
    with Path(path).open() as f:raw=list(csv.DictReader(f))
    if not raw:raise ValueError('Empty trajectory file')
    required={'initial_id','replica','round','x','kappa_plus','kappa_minus'}
    if not required<=set(raw[0]):raise ValueError(f'Missing canonical columns: {required-set(raw[0])}')
    rows=[];keys=set()
    for r in raw:
        r=dict(r);r['replica']=int(r['replica']);r['round']=int(r['round'])
        if min(r['replica'],r['round'])<0:raise ValueError('Negative replica/round')
        key=(r['initial_id'],r['replica'],r['round'])
        if key in keys:raise ValueError(f'Duplicate trajectory key {key}')
        keys.add(key)
        for m in ['x','kappa_plus','kappa_minus']:
            r[m]=float(r[m])
            if not np.isfinite(r[m]) or not 0<=r[m]<=1:raise ValueError(f'Invalid {m}')
        expected=r['x'] if p.controller_target==1 else 1-r['x']
        if r.get('target_share') not in ('',None) and not np.isclose(float(r['target_share']),expected):
            raise ValueError('target_share has inconsistent target orientation')
        r['target_share']=expected;r['active_plus']=p.F_plus*r['kappa_plus'];r['active_minus']=p.F_minus*r['kappa_minus']
        for m in ['peer_target_share','action_next','effective_action_next','posts_next']:
            if r.get(m) not in ('',None):
                r[m]=float(r[m])
                upper=p.b if m=='posts_next' else 1
                if not np.isfinite(r[m]) or not 0<=r[m]<=upper:raise ValueError(f'Invalid optional {m}')
        rows.append(r)
    return rows

def compare_exports(theory_dir,simulation_csv,simulation_manifest,outdir):
    td=Path(theory_dir);tm=json.loads((td/'manifest.json').read_text());sm=json.loads(Path(simulation_manifest).read_text())
    if sm.get('round_stage')!='end_of_day_before_controller_effect':
        raise ValueError('Simulation manifest must declare round_stage=end_of_day_before_controller_effect')
    if sm.get('sampling') not in ('with_replacement','without_replacement'):raise ValueError('Declare simulation sampling')
    if sm.get('data_kind') not in ('simulation','interface_test'):raise ValueError('Declare data_kind=simulation or interface_test')
    p=strict_dataclass(Parameters,sm['parameters'])
    from dataclasses import asdict
    if asdict(p)!=tm['parameters']:raise ValueError('Physical parameters differ; resolve b/q_c/F and response/policy values explicitly')
    theory_initial=json.loads((td/'initials.json').read_text())['snapshots']
    if sm.get('initial_snapshots')!=theory_initial:
        raise ValueError('Initial snapshot records/order differ; export actual shared projections/front-page counts')
    tt=read_rows(td/'trajectories.csv',p);ss=read_rows(simulation_csv,p)
    def groups(rows):
        out={}
        for r in rows:out.setdefault((r['initial_id'],r['round']),[]).append(r)
        return out
    tg,sg=groups(tt),groups(ss)
    if set(tg)!=set(sg):raise ValueError('Different block/round coverage; no silent time truncation or inner join is allowed')
    # Require a complete replica panel within each block; do not silently condition on survivors.
    for groups_ in [tg,sg]:
        ids={}
        for (block,day),rows in groups_.items():
            present={r['replica'] for r in rows}
            if block in ids and ids[block]!=present:raise ValueError('Incomplete replica panel; resolve attrition explicitly')
            ids[block]=present
    for init in theory_initial:
        for rows in [tg[(init['snapshot_id'],init['round'])],sg[(init['snapshot_id'],init['round'])]]:
            for r in rows:
                if any(not np.isclose(r[k],init[k],atol=1e-12,rtol=0) for k in ['x','kappa_plus','kappa_minus']):
                    raise ValueError('Trajectory initial state does not match supplied snapshot')
    ts=summarize(tt);ssummary=summarize(ss)
    ts={(r['round'],r['metric']):r for r in ts};su={(r['round'],r['metric']):r for r in ssummary}
    common=sorted(set(ts)&set(su));res=[]
    for day,metric in common:
        diffs=[]
        for block,d in sorted(tg):
            if d!=day:continue
            a=[float(r[metric]) for r in tg[(block,d)] if r.get(metric) not in ('',None)]
            b=[float(r[metric]) for r in sg[(block,d)] if r.get(metric) not in ('',None)]
            if a and b:diffs.append(np.mean(a)-np.mean(b))
        se=float(np.std(diffs,ddof=1)/np.sqrt(len(diffs))) if len(diffs)>1 else None
        delta=float(np.mean(diffs));res.append({'round':day,'metric':metric,'theory_mean':ts[(day,metric)]['mean'],
              'simulation_mean':su[(day,metric)]['mean'],'theory_minus_simulation':delta,
              'se_paired_blocks':se if se is not None else '',
              'ci95_low':delta-1.96*se if se is not None else '',
              'ci95_high':delta+1.96*se if se is not None else '', 'n_blocks':len(diffs)})
    out=prepare_output(outdir);write_csv(out/'comparison.csv',res)
    write_csv(out/'theory_summary.csv',list(ts.values()));write_csv(out/'simulation_summary.csv',list(su.values()))
    errors=[]
    for metric in sorted({r['metric'] for r in res}):
        a=np.array([r['theory_minus_simulation'] for r in res if r['metric']==metric])
        errors.append({'metric':metric,'bias_over_exported_rounds':float(a.mean()),'rmse_over_exported_rounds':float(np.sqrt((a*a).mean()))})
    write_csv(out/'errors.csv',errors)
    report={'data_kind':sm['data_kind'],'parameters_checked':True,'initials_checked':True,
            'simulation_sampling':sm['sampling'],'theory_sampling':tm['solver']['sampling'],
            'sampling_mismatch_is_declared_ablation':sm['sampling']!=tm['solver']['sampling'],
            'theory_covariance':tm['solver']['covariance'],'round_stage':sm['round_stage'],
            'CI':'paired initial-block mean differences; undefined with only one block',
            'interpretation':'a simulator comparison is meaningful only when data_kind=simulation',
            'not_provided':['distribution-distance inference','information estimation','stationarity or phase-transition inference']}
    (out/'comparison_manifest.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[m for m in ['target_share','kappa_plus','kappa_minus','peer_target_share','action_next','posts_next']
              if any(k[1]==m for k in common)]
    fig,axs=plt.subplots(len(selected),2,figsize=(10,2.6*len(selected)),squeeze=False,constrained_layout=True)
    for i,metric in enumerate(selected):
        for summary,label,color in [(ts,'Reduced theory','#0072b2'),(su,'Simulation' if sm['data_kind']=='simulation' else 'INTERFACE TEST','#d55e00')]:
            rr=[summary[k] for k in common if k[1]==metric];days=np.array([r['round'] for r in rr]);mean=np.array([r['mean'] for r in rr])
            axs[i,0].plot(days,mean,label=label,color=color)
            if all(r['se']!='' for r in rr):
                se=np.array([r['se'] for r in rr]);axs[i,0].fill_between(days,mean-1.96*se,mean+1.96*se,color=color,alpha=.13)
        rr=[r for r in res if r['metric']==metric];x=[r['round'] for r in rr];y=[r['theory_minus_simulation'] for r in rr]
        axs[i,1].plot(x,y,color='#555555');axs[i,1].axhline(0,color='black',lw=.6)
        if all(r['se_paired_blocks']!='' for r in rr):
            axs[i,1].fill_between(x,[r['ci95_low'] for r in rr],[r['ci95_high'] for r in rr],alpha=.15,color='#555555')
        axs[i,0].set_ylabel(metric);axs[i,1].set_ylabel('Theory − simulation');axs[i,0].legend(fontsize=8)
        for ax in axs[i]:ax.set_xlabel('Round');ax.grid(alpha=.15)
    fig.suptitle('Theory versus simulator' if sm['data_kind']=='simulation' else 'INTERFACE TEST ONLY — not game validation')
    fig.savefig(out/'overlays.pdf');fig.savefig(out/'overlays.png',dpi=150);plt.close(fig)
    return out
