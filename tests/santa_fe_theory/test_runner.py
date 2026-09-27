import csv,json,tempfile,unittest
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from santa_fe_theory.core import *
from santa_fe_theory.io import run_ensemble,summarize,export_run
from santa_fe_theory.branch import branch_snapshot,export_branches
from santa_fe_theory.compare import compare_exports

P=Parameters(N=4,F_plus=3,F_minus=1,q=2,q_c=2,b=2,rho=.7,
             beta_evidence=1.,beta_social=1.3,policy_beta=8.,policy_threshold=.55,controller_target=-1)
S=Solver(rounds=2,replicas_per_initial=2,substeps=8,seed=3)
INIT={'snapshot_id':'a','round':0,'x':.5,'kappa_plus':.25,'kappa_minus':.25,'front_page_counts':[1,1,1,1]}
SNAP={'snapshot_id':'closed','round':5,'x':.5,'kappa_plus':.25,'kappa_minus':.25,'peer_counts':[1,1,1,1]}

class KernelTests(unittest.TestCase):
    def test_reject_legacy_fractional_budget_unknown_key(self):
        with self.assertRaises(ValueError):replace(P,model_version='santa_fe_legacy_v2')
        with self.assertRaises(ValueError):replace(P,b=.5)
        with self.assertRaises(ValueError):strict_dataclass(Solver,{'made_up':2})
    def test_acquisition_duplicates(self):
        np.testing.assert_allclose(acquisition(2,0,2),[0,.5,.5])
    def test_full_message_sample_vs_replacement(self):
        p=replace(P,q=2,beta_evidence=0,beta_social=0)
        no=KernelBank(p,S);yes=KernelBank(p,replace(S,sampling='with_replacement'))
        _,_,r=no.evaluate([.2,0,0],no.effective([1,0,1,0]))
        _,_,w=yes.evaluate([.2,0,0],yes.effective([1,0,1,0]))
        np.testing.assert_allclose(r,[.5,0,.5,0],atol=1e-12)
        np.testing.assert_allclose(w,[.375,.125,.375,.125],atol=1e-12)
    def test_zero_sensitivities_vote_drift(self):
        p=replace(P,beta_evidence=0,beta_social=0)
        bank=KernelBank(p,S);A,D,R=bank.evaluate([.3,.2,.8],bank.effective([1,1,1,1]))
        self.assertAlmostEqual(A[0],.2);self.assertAlmostEqual(R[0]+R[1],.5)
    def test_sampling_cap_and_empty_board(self):
        bank=KernelBank(replace(P,q=5),S)
        A,D,R=bank.evaluate([.4,0,0],bank.effective([0,0,0,0]))
        np.testing.assert_allclose(A,[.1,0,0],atol=1e-12)
        bank.effective([1,0,0,0])
    def test_exact_one_message_moments(self):
        p=replace(P,q=1);bank=KernelBank(p,replace(S,covariance='full_poisson'))
        x=.3;A,D,R=bank.evaluate([x,0,0],bank.effective([2,0,2,0]))
        m=np.zeros(3);raw=np.zeros((3,3));em=np.zeros(4)
        for sign in [1,-1]:
            pp=sigmoid(sign*(p.beta_evidence+p.beta_social))
            for old,po in [(0,1-x),(1,x)]:
                for new,pn in [(0,1-pp),(1,pp)]:
                    v=np.array([new-old,1/p.F_plus if sign==1 else 0,1/p.F_minus if sign==-1 else 0])
                    wt=.5*po*pn;m+=wt*v;raw+=wt*np.outer(v,v)
            em[0 if sign==1 else 1]+=.5*pp
            em[2 if sign==-1 else 3]+=.5*(1-pp)
        np.testing.assert_allclose(A,m,atol=1e-12);np.testing.assert_allclose(D,raw,atol=1e-12)
        np.testing.assert_allclose(R,em,atol=1e-12)
    def test_fixed_clock_centering_and_psd(self):
        po=KernelBank(P,replace(S,covariance='full_poisson'));fixed=KernelBank(P,S)
        e=po.effective([1,1,1,1])
        for x,kp,km in [(0,0,0),(1,1,1),(.2,.6,.3)]:
            A,D,_=po.evaluate([x,kp,km],e);_,C,_=fixed.evaluate([x,kp,km],e)
            np.testing.assert_allclose(C,D-np.outer(A,A),atol=1e-12)
            root=covariance_root(C);np.testing.assert_allclose(root@root.T,C,atol=1e-12)
    def test_full_sensor_propensity_and_target(self):
        p=replace(P,q_c=4,controller_target=1)
        self.assertAlmostEqual(action_propensity([1,0,2,1],p),sigmoid(8*(.55-.25)))
        np.testing.assert_array_equal(controlled_board([1,0,2,1],1,p),[3,0,2,1])
    def test_projection_ids_and_aligned_rules(self):
        signs={0:1,1:1,2:-1};agents=[{'vote':1,'active_fact_ids':[0]}, {'vote':-1,'active_fact_ids':[2]}]
        messages=[{'vote':1,'fact_id':0},{'vote':-1,'fact_id':None}]
        out=project_snapshot(agents,signs,messages,N=2)
        self.assertEqual(out['front_page_counts'],[1,0,0,1]);self.assertEqual(out['kappa_plus'],.25)
        with self.assertRaises(ValueError):project_snapshot(agents,signs,[{'vote':1,'fact_id':2}])
    def test_no_artificial_covariance_floor(self):
        np.testing.assert_array_equal(covariance_root(np.zeros((3,3))),np.zeros((3,3)))
        with self.assertRaises(ArithmeticError):covariance_root(-np.eye(3))
    def test_general_fact_dimensions(self):
        p=replace(P,F_plus=5,F_minus=2,q=1)
        bank=KernelBank(p,S);A,D,R=bank.evaluate([.5,.2,.3],bank.effective([1,1,1,1]))
        self.assertEqual(D.shape,(3,3));self.assertAlmostEqual(R.sum(),1)

class RunTests(unittest.TestCase):
    def test_repeatability_and_bounds(self):
        a,ad=run_ensemble(P,S,[INIT]);b,bd=run_ensemble(P,S,[INIT])
        self.assertEqual(a,b);self.assertEqual(ad,bd);self.assertEqual(len(a),6)
        self.assertEqual(a[0]['action_next'],'')
        for row in a:
            self.assertTrue(0<=row['x']<=1)
            if row['round']:
                self.assertEqual(sum(row['peer_'+c] for c in CATEGORIES),P.N)
                self.assertEqual(sum(row['front_next_'+c] for c in CATEGORIES),P.N+row['posts_next'])
    def test_b0_exact_null_branch_and_day_alignment(self):
        rows,stat,diag=branch_snapshot(replace(P,b=0),S,SNAP,4,99)
        self.assertEqual(stat['chi_target'],0);self.assertEqual(stat['se_paired'],0)
        self.assertTrue(all(r['outcome_round']==6 for r in rows))
    def test_night_endpoints_no_acquisition(self):
        # q=0 removes daytime acquisition; noise on coverage then identically zero.
        for rho,expected in [(0,0),(1,.25)]:
            p=replace(P,q=0,rho=rho);rows,_=run_ensemble(p,S,[INIT])
            for row in rows:
                if row['round']:self.assertAlmostEqual(row['kappa_plus'],expected)
    def test_no_daytime_noise_keeps_hybrid_sampling(self):
        rows,_=run_ensemble(P,replace(S,daytime_noise=False),[INIT])
        self.assertIsInstance(rows[1]['action_next'],int)
    def test_boundary_raise(self):
        with self.assertRaises(ArithmeticError):bound(np.array([1.1,0,0]),replace(S,boundary='raise'),new_diagnostics(),'day')
    def test_equal_block_summary(self):
        rows=[dict(initial_id='a',round=0,x=0),dict(initial_id='a',round=0,x=0),dict(initial_id='b',round=0,x=1)]
        out=summarize(rows)[0];self.assertEqual(out['mean'],.5);self.assertEqual(out['n_blocks'],2)
    def test_export_and_compare_interface_only(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);cfg={'schema_version':1,'parameters':asdict(P),'solver':asdict(S)}
            (d/'config.json').write_text(json.dumps(cfg));(d/'initials.json').write_text(json.dumps({'schema_version':1,'kind':'initial','snapshots':[INIT]}))
            out=export_run(d/'config.json',d/'initials.json',d/'run')
            sm={'parameters':asdict(P),'sampling':S.sampling,'round_stage':'end_of_day_before_controller_effect',
                'data_kind':'interface_test','initial_snapshots':[INIT]}
            (d/'sim_manifest.json').write_text(json.dumps(sm))
            compared=compare_exports(out,out/'trajectories.csv',d/'sim_manifest.json',d/'comparison')
            self.assertTrue((compared/'overlays.pdf').exists())
            with (compared/'comparison.csv').open() as f: rr=list(csv.DictReader(f))
            self.assertTrue(all(float(r['theory_minus_simulation'])==0 for r in rr))
            sm['parameters']['b']=3;(d/'bad.json').write_text(json.dumps(sm))
            with self.assertRaises(ValueError):compare_exports(out,out/'trajectories.csv',d/'bad.json',d/'bad')

if __name__=='__main__':unittest.main()
