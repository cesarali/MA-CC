import sys, multiprocessing
sys.path.insert(0,'/Users/rsanchez/Projects/MA-CC/analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/theory/models')
from mean_field import Model, Spec
def one(args):
    setup, arm, R, seed = args
    o=Model(Spec(setup,arm,0.5,6,board='stochastic',replicas=R,seed=seed)).run()[120]
    return setup, arm, R, seed, o['x2'], o['messages']/R, o['ctrl_b']
if __name__=='__main__':
    eps={1:300,2:200,4:120,8:60}
    jobs=[(s,a,R,e) for s in ('symmetric','nosolution') for R in eps for a in ('silent','false') for e in range(eps[R])]
    with multiprocessing.Pool(12) as p: r=p.map(one,jobs,chunksize=4)
    import pandas as pd
    d=pd.DataFrame(r,columns=['setup','arm','R','seed','x2','msgs_per_R','ctrl_b'])
    d.to_csv('/Users/rsanchez/Projects/MA-CC/analysis/task003_llm_free/results/key_phenomena/theory_runs/diag_scale_check.csv',index=False)
    g=d.pivot_table(index=['setup','R','seed'],columns='arm',values='x2'); g['G']=g.false-g.silent
    print(g.groupby(['setup','R']).G.agg(['mean','sem']).round(3))
    print(d.groupby(['setup','R','arm'])[['x2','msgs_per_R','ctrl_b']].mean().round(3))
