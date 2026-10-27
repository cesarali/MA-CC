import sys, multiprocessing
import numpy as np
sys.path.insert(0,'/Users/rsanchez/Projects/MA-CC/analysis/theory_claude/models')
from mean_field import Model, Spec
def one(args):
    setup, arm, seed = args
    o=Model(Spec(setup,arm,0.5,6,board='stochastic',seed=seed)).run()[120]
    return setup, arm, seed, o['x2'], o['messages'], o['ctrl_b']
if __name__=='__main__':
    jobs=[(s,a,e) for s in ('symmetric','nosolution') for a in ('silent','false') for e in range(300)]
    with multiprocessing.Pool(12) as p: r=p.map(one,jobs,chunksize=10)
    import pandas as pd
    d=pd.DataFrame(r,columns=['setup','arm','seed','x2','msgs','ctrl_b'])
    g=d.pivot_table(index=['setup','seed'],columns='arm',values='x2'); g['G']=g.false-g.silent
    print(d.groupby(['setup','arm'])[['x2','msgs','ctrl_b']].mean().round(3))
    print(g.groupby('setup').G.agg(['mean','sem']).round(3))
