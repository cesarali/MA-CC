import sys, multiprocessing
sys.path.insert(0,'/Users/rsanchez/Projects/MA-CC/analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/theory/models')
from mean_field import Model, Spec
CELLS=[('symmetric',0.25),('symmetric',0.75),('nosolution',0.25),('nosolution',1.0)]
def one(args):
    setup, rho, q, arm, R, seed = args
    o=Model(Spec(setup,arm,rho,q,board='stochastic',replicas=R,seed=seed)).run()[120]
    return setup, rho, q, arm, R, seed, o['x2']
if __name__=='__main__':
    eps={1:300,4:150}
    jobs=[(s,r,q,a,R,e) for s,r in CELLS for q in (3,6) for a in ('silent','false') for R in eps for e in range(eps[R])]
    with multiprocessing.Pool(12) as p: res=p.map(one,jobs,chunksize=4)
    import pandas as pd, numpy as np
    d=pd.DataFrame(res,columns=['setup','rho','q','arm','R','seed','x2'])
    d.to_csv('/Users/rsanchez/Projects/MA-CC/analysis/task003_llm_free/results/key_phenomena/theory_runs/diag_reversal_N.csv',index=False)
    g=d.pivot_table(index=['setup','rho','R','seed'],columns=['q','arm'],values='x2')
    G3=g[(3,'false')]-g[(3,'silent')]; G6=g[(6,'false')]-g[(6,'silent')]; D=(G6-G3)
    out=pd.DataFrame({'G3':G3,'G6':G6,'D':D}).groupby(level=[0,1,2]).agg(['mean','sem'])
    print(out.round(3))
