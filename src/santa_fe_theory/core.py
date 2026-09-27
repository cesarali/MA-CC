"""Reduced hybrid Langevin theory; this module does not evolve individual agents.

Category order: (+,+), (+,0), (-,-), (-,0).
The epistemic law is independent Binomial(F+,k+) x Binomial(F-,k-),
independent of the old vote. Board identities are exchangeable within sign.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from functools import lru_cache
from math import comb, factorial, isfinite, prod
import numpy as np

VERSION = '1.0.0'
CATEGORIES = ['plus_fact', 'plus_empty', 'minus_fact', 'minus_empty']

@dataclass(frozen=True)
class Parameters:
    N: int
    F_plus: int
    F_minus: int
    q: int
    q_c: int
    b: int
    rho: float
    beta_evidence: float
    beta_social: float
    policy_beta: float
    policy_threshold: float
    controller_target: int
    model_version: str = 'santa_fe_epistemic_feedback_v3'

    def __post_init__(self):
        for key in ['N','F_plus','F_minus','q','q_c','b','controller_target']:
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int):
                raise ValueError(f'{key} must be an already-resolved integer, not a fraction or float')
        if self.N < 1 or self.F_minus < 1 or self.F_plus <= self.F_minus:
            raise ValueError('Require N>=1 and F_plus>F_minus>=1 for this binary theory')
        if self.q<0 or self.b<0 or not 1<=self.q_c<=self.N:
            raise ValueError('Require q>=0, b>=0, and 1<=q_c<=N')
        if self.controller_target not in (-1,1): raise ValueError('controller_target must be -1 or +1')
        for key in ['rho','beta_evidence','beta_social','policy_beta','policy_threshold']:
            if not isfinite(getattr(self,key)): raise ValueError(f'{key} must be finite')
        if not 0<=self.rho<=1 or not 0<=self.policy_threshold<=1:
            raise ValueError('rho and policy_threshold must lie in [0,1]')
        if min(self.beta_evidence,self.beta_social,self.policy_beta)<0:
            raise ValueError('Sensitivities and policy_beta must be nonnegative')
        if self.model_version!='santa_fe_epistemic_feedback_v3':
            raise ValueError('Legacy v2 is a different process and is not supported')

@dataclass(frozen=True)
class Solver:
    rounds: int = 60
    replicas_per_initial: int = 128
    seed: int = 20260926
    substeps: int = 48
    sampling: str = 'without_replacement'
    covariance: str = 'fixed_clock'
    daytime_noise: bool = True
    boundary: str = 'project'
    max_kernel_work: int = 50_000_000

    def __post_init__(self):
        for key in ['rounds','replicas_per_initial','seed','substeps','max_kernel_work']:
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int): raise ValueError(f'{key} must be integer')
        if self.rounds<0 or self.seed<0 or min(self.replicas_per_initial,self.substeps,self.max_kernel_work)<1:
            raise ValueError('Invalid solver size/seed')
        if self.sampling not in ('with_replacement','without_replacement'):
            raise ValueError('Unknown sampling; choose with_replacement or without_replacement')
        if self.covariance not in ('paper_diagonal','full_poisson','fixed_clock'):
            raise ValueError('Unknown covariance')
        if self.boundary not in ('project','raise'): raise ValueError('boundary must be project or raise')
        if not isinstance(self.daytime_noise,bool): raise ValueError('daytime_noise must be boolean')

def strict_dataclass(cls, data):
    unknown=set(data)-{f.name for f in fields(cls)}
    if unknown: raise ValueError(f'Unknown {cls.__name__} keys: {sorted(unknown)}')
    return cls(**data)

def sigmoid(z):
    z=np.asarray(z,dtype=float)
    ans=np.exp(-np.logaddexp(0.,-z))
    return float(ans) if ans.ndim==0 else ans

def validate_state(state):
    a=np.asarray(state,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError('State [x,kappa_plus,kappa_minus] must be finite and within [0,1]')
    return a.copy()

def validate_board(counts, total=None):
    a=np.asarray(counts)
    if a.shape!=(4,) or not np.issubdtype(a.dtype,np.number):
        raise ValueError('Board counts must be four integer counts in documented category order')
    if not np.isfinite(a).all() or np.any(a<0) or np.any(a!=np.floor(a)):
        raise ValueError('Board counts must be finite nonnegative integers')
    a=a.astype(np.int64)
    if total is not None and int(a.sum())!=total: raise ValueError(f'Expected board size {total}')
    return a.copy()

def compositions(q):
    return np.array([(a,b,c,q-a-b-c) for a in range(q+1)
                     for b in range(q-a+1) for c in range(q-a-b+1)],dtype=int)

@lru_cache(maxsize=512)
def acquisition(F, held, draws):
    """Exchangeable identities: each fact-bearing message draws a uniform sign-ID.
    Exact duplicate handling for THIS closure, not for a fixed identity-resolved board.
    """
    p=np.zeros(F+1);p[held]=1.
    for _ in range(draws):
        nxt=p*np.arange(F+1)/F
        nxt[1:]+=p[:-1]*(1-np.arange(F)/F)
        p=nxt
    p.setflags(write=False)
    return p

class KernelBank:
    def __init__(self,p: Parameters,solver: Solver):
        self.p,self.solver=p,solver
        self.cache={}
        self.r,self.s=np.meshgrid(np.arange(p.F_plus+1),np.arange(p.F_minus+1),indexing='ij')
        self.br=np.array([comb(p.F_plus,r) for r in range(p.F_plus+1)],float)
        self.bs=np.array([comb(p.F_minus,s) for s in range(p.F_minus+1)],float)

    def kernel(self,qeff):
        if qeff in self.cache:return self.cache[qeff]
        p=self.p; size=(p.F_plus+1)*(p.F_minus+1)
        work=comb(qeff+3,3)*size*size
        if work>self.solver.max_kernel_work:
            raise ValueError(f'Kernel work estimate {work:,} exceeds max_kernel_work; '
                             'reduce F/q or explicitly raise limit after estimating resources')
        ns=compositions(qeff)
        # Emission four, mean gains two, gain squares/cross three, new-vote*gains two.
        tab=np.zeros((len(ns),p.F_plus+1,p.F_minus+1,11))
        rr,ss=self.r,self.s
        E=np.divide(rr-ss,rr+ss,out=np.zeros_like(rr,dtype=float),where=(rr+ss)>0)
        for j,n in enumerate(ns):
            H=(n[0]+n[1]-n[2]-n[3])/qeff if qeff else 0.
            vote=sigmoid(p.beta_evidence*E+p.beta_social*H)
            for r in range(p.F_plus+1):
                ar=acquisition(p.F_plus,r,int(n[0]))
                for s in range(p.F_minus+1):
                    weights=ar[:,None]*acquisition(p.F_minus,s,int(n[2]))[None,:]
                    dp=(rr-r)/p.F_plus; dm=(ss-s)/p.F_minus
                    channels=[vote*(rr>0),vote*(rr==0),(1-vote)*(ss>0),(1-vote)*(ss==0),
                              dp,dm,dp*dp,dm*dm,dp*dm,vote*dp,vote*dm]
                    tab[j,r,s]=[np.sum(weights*v) for v in channels]
        if not np.allclose(tab[...,:4].sum(-1),1,atol=1e-12):
            raise ArithmeticError('Emission kernel is not normalized')
        self.cache[qeff]=(ns,tab)
        return ns,tab

    def effective(self,board):
        board=validate_board(board);M=int(board.sum())
        qeff=(min(self.p.q,M) if self.solver.sampling=='without_replacement' else self.p.q) if M else 0
        ns,tab=self.kernel(qeff)
        if not qeff:weights=np.ones(1)
        elif self.solver.sampling=='with_replacement':
            probs=board/M
            weights=np.array([factorial(qeff)/prod(factorial(int(k)) for k in n)*
                              np.prod(probs**n) for n in ns])
        else:
            denominator=comb(M,qeff)
            weights=np.array([np.prod([comb(int(v),int(k)) if k<=v else 0
                                       for v,k in zip(board,n)],dtype=object)/denominator for n in ns],float)
        if not np.isclose(weights.sum(),1,atol=1e-12):raise ArithmeticError('Sample law not normalized')
        return np.tensordot(weights,tab,axes=(0,0))

    def evaluate(self,state,effective):
        x,kp,km=validate_state(state);p=self.p
        r=np.arange(p.F_plus+1);s=np.arange(p.F_minus+1)
        pr=self.br*kp**r*(1-kp)**(p.F_plus-r)
        ps=self.bs*km**s*(1-km)**(p.F_minus-s)
        moments=np.einsum('r,s,rsk->k',pr,ps,effective,optimize=False)
        phi=moments[0]+moments[1]
        A=np.array([phi-x,moments[4],moments[5]])
        D=np.array([[(1-x)*phi+x*(1-phi),moments[9]-x*A[1],moments[10]-x*A[2]],
                    [moments[9]-x*A[1],moments[6],moments[8]],
                    [moments[10]-x*A[2],moments[8],moments[7]]])
        if self.solver.covariance=='fixed_clock':D-=np.outer(A,A)
        elif self.solver.covariance=='paper_diagonal':
            D=np.diag([D[0,0],A[1]/p.F_plus,A[2]/p.F_minus])
        return A,(D+D.T)/2,moments[:4]

def covariance_root(D):
    vals,vecs=np.linalg.eigh(D)
    if vals.min() < -1e-10: raise ArithmeticError(f'Non-PSD diffusion: eigenvalues={vals}')
    # Numerical negative eigenvalues only; no artificial positive variance floor.
    return vecs*np.sqrt(np.maximum(vals,0))[None,:]

def bound(state,solver,diagnostics,prefix):
    bad=(state<0)|(state>1)
    for i,key in enumerate(['x','kappa_plus','kappa_minus']):
        diagnostics[f'{prefix}_projections_{key}']+=int(bad[i])
    if np.any(bad) and solver.boundary=='raise':raise ArithmeticError(f'{prefix} boundary violation: {state}')
    return np.clip(state,0,1)

def new_diagnostics():
    return {f'{stage}_projections_{key}':0 for stage in ['night','day']
            for key in ['x','kappa_plus','kappa_minus']}

def advance_day(state,board,p,solver,bank,rng,diagnostics=None):
    """Night + one daytime unit. Returns end population, peer counts, post-night state.
    No controller action is applied inside this function.
    """
    if diagnostics is None:diagnostics=new_diagnostics()
    state=validate_state(state);effective=bank.effective(board)
    old=state[1:].copy();sizes=np.array([p.F_plus,p.F_minus])
    state[1:]=p.rho*old+np.sqrt(p.rho*(1-p.rho)*old/(p.N*sizes))*rng.normal(size=2)
    state=bound(state,solver,diagnostics,'night');night=state.copy()
    dt=1/solver.substeps;emission=np.zeros(4)
    for _ in range(solver.substeps):
        A,D,R=bank.evaluate(state,effective)
        emission+=dt*R
        increment=A*dt
        if solver.daytime_noise:
            increment+=np.sqrt(dt/p.N)*(covariance_root(D)@rng.normal(size=3))
        state=bound(state+increment,solver,diagnostics,'day')
    if np.min(emission)<-1e-12:raise ArithmeticError('Negative emission probability')
    emission=np.maximum(emission,0);emission/=emission.sum()
    peers=rng.multinomial(p.N,emission)
    return state,peers,night

def action_propensity(peer_counts,p):
    C=validate_board(peer_counts,p.N);nc=int(C[:2].sum() if p.controller_target==1 else C[2:].sum())
    out=0.
    for y in range(max(0,p.q_c-(p.N-nc)),min(p.q_c,nc)+1):
        w=comb(nc,y)*comb(p.N-nc,p.q_c-y)/comb(p.N,p.q_c)
        out+=w*sigmoid(p.policy_beta*(p.policy_threshold-y/p.q_c))
    return float(out)

def controlled_board(peer_counts,action,p):
    if action not in (0,1):raise ValueError('Forced action must be 0 or 1')
    C=validate_board(peer_counts,p.N)
    C[0 if p.controller_target==1 else 2]+=p.b*action
    return C

def feedback(peers,p,rng):
    nc=int(peers[:2].sum() if p.controller_target==1 else peers[2:].sum())
    y=int(rng.hypergeometric(nc,p.N-nc,p.q_c))
    probability=sigmoid(p.policy_beta*(p.policy_threshold-y/p.q_c))
    u=int(rng.random()<probability)
    return controlled_board(peers,u,p),y,u,probability

def project_snapshot(agent_states,fact_signs,board_messages,N=None):
    """Adapter helper for CANONICAL records, not a parser of repository internals.
    agent_states: [{'vote': +/-1, 'active_fact_ids': [...]}]
    board_messages: [{'vote': +/-1,'fact_id': id or None}]
    fact_signs: mapping of globally unique IDs to +/-1; ID types must match.
    """
    if not fact_signs or any(v not in (-1,1) for v in fact_signs.values()):
        raise ValueError('fact_signs must map fact IDs to -1/+1')
    fp=sum(v==1 for v in fact_signs.values());fm=len(fact_signs)-fp
    if min(fp,fm)==0:raise ValueError('Both fact signs required')
    if not agent_states or (N is not None and len(agent_states)!=N):raise ValueError('Agent count mismatch')
    n=len(agent_states);truth=0;rp=0;rm=0
    for agent in agent_states:
        if agent['vote'] not in (-1,1):raise ValueError('Invalid vote')
        ids=agent['active_fact_ids']
        if len(set(ids))!=len(ids):raise ValueError('Duplicate active ID in one agent')
        if any(f not in fact_signs for f in ids):raise ValueError('Unknown active fact ID')
        truth+=agent['vote']==1
        rp+=sum(fact_signs[f]==1 for f in ids);rm+=sum(fact_signs[f]==-1 for f in ids)
    C=np.zeros(4,dtype=int)
    for message in board_messages:
        v=message['vote'];f=message.get('fact_id')
        if v not in (-1,1):raise ValueError('Invalid message vote')
        if f is not None and (f not in fact_signs or fact_signs[f]!=v):
            raise ValueError('Unknown/cross-sign board fact; incompatible v3 message')
        C[(0 if v==1 else 2)+(f is None)]+=1
    return {'x':truth/n,'kappa_plus':rp/(n*fp),'kappa_minus':rm/(n*fm),
            'front_page_counts':C.tolist(),'N':n,'F_plus':fp,'F_minus':fm}
