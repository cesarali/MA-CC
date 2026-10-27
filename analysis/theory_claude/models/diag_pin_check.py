import sys, multiprocessing
sys.path.insert(0,'/Users/rsanchez/Projects/MA-CC/analysis/theory_claude/models')
from mean_field import Model, Spec
def one(args):
    setup, rho, arm, pin, seed = args
    o=Model(Spec(setup,arm,rho,6,board='stochastic',pin_own=pin,seed=seed)).run()[120]
    return setup, rho, arm, pin, seed, o['x2'], o['ctrl_b']
if __name__=='__main__':
    jobs=[(s,r,a,pin,e) for s in ('symmetric','nosolution') for r in (0.25,0.5) for a in ('silent','false') for pin in (False,True) for e in range(300)]
    with multiprocessing.Pool(12) as p: res=p.map(one,jobs,chunksize=8)
    import pandas as pd
    d=pd.DataFrame(res,columns=['setup','rho','arm','pin','seed','x2','ctrl_b'])
    g=d.pivot_table(index=['setup','rho','pin','seed'],columns='arm',values='x2'); g['G']=g.false-g.silent
    print(g.groupby(level=[0,1,2])[['silent','false','G']].mean().round(3))
