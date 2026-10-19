import unittest
import numpy as np
from v7.multimodel_residual import (L,H,PHASES,refs_for,choose_cases,feature_blocks,fit_probe,apply_probe)


class MultiModelTests(unittest.TestCase):
    def test_complete_windows_do_not_cross_temporal_roles(self):
        s=dict(n=20000,runs=[(0,20000)])
        for phase,(lo,hi,_,_) in PHASES.items():
            starts=refs_for(s,phase)
            self.assertTrue(np.all(starts>=int(lo*s['n'])))
            self.assertTrue(np.all(starts+L+H<=int(hi*s['n'])))
            self.assertTrue(np.all(starts%PHASES[phase][2]==0))
        self.assertLessEqual(refs_for(s,'C').max()+L+H,refs_for(s,'D').min())

    def test_current_target_is_not_a_feature_and_old_history_matters(self):
        rng=np.random.default_rng(3);n=24
        bank=dict(x=rng.normal(size=(n,L)),y=rng.normal(size=(n,H)),sid=np.zeros(n,dtype=int),
            start=np.arange(n)*96,std=np.ones(n))
        q=dict(x=rng.normal(size=(5,L)),y=rng.normal(size=(5,H)),sid=np.zeros(5,dtype=int),
            start=np.arange(5)*96+6000,std=np.ones(5))
        r,j=choose_cases(q,bank,13);base=np.zeros((5,H));old=np.zeros((n,H))
        c,e,s=feature_blocks(q,base,bank,old,r,j,'real',13)
        q['y'][:]=np.nan
        cc,ee,ss=feature_blocks(q,base,bank,old,r,j,'real',13)
        np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(e,ee)
        changed=dict(bank,x=bank['x']*2+2)
        _,different,_=feature_blocks(q,base,changed,old,r,j,'real',13)
        self.assertGreater(np.max(abs(different-e)),.01)
        bank['start'][:]=7000
        with self.assertRaises(ValueError):choose_cases(q,bank,13)

    def test_random_cases_are_distinct_disjoint_and_outcome_free(self):
        rng=np.random.default_rng(7)
        bank=dict(x=rng.normal(size=(24,L)),y=rng.normal(size=(24,H)),sid=np.zeros(24,dtype=int),start=np.arange(24)*96,std=np.ones(24))
        q=dict(x=rng.normal(size=(3,L)),sid=np.zeros(3,dtype=int),start=np.arange(3)*96+6000,std=np.ones(3))
        r,a=choose_cases(q,bank,41);bank['y'][:]=np.nan;rr,b=choose_cases(q,bank,41)
        np.testing.assert_array_equal(r,rr);np.testing.assert_array_equal(a,b)
        for x,y in zip(r,a):
            self.assertEqual(len(np.unique(x)),10);self.assertEqual(len(np.unique(y)),10)
            self.assertFalse(np.intersect1d(x,y).size)

    def test_effective_capacity_is_matched_without_reading_targets(self):
        rng=np.random.default_rng(4);q=rng.normal(size=(180,608));y=rng.normal(size=(180,H))
        a=fit_probe(q,rng.normal(size=(180,32)),y,13)
        b=fit_probe(q,rng.normal(size=(180,32))*5,y,13)
        self.assertAlmostEqual(float(a['effective_df']),64.,places=5)
        self.assertAlmostEqual(float(b['effective_df']),64.,places=5)
        self.assertEqual(a['coef'].shape,b['coef'].shape)
        pred=apply_probe(a,q,np.zeros((180,32)),np.zeros((180,H)),np.ones(180))
        self.assertTrue(np.isfinite(pred).all());self.assertTrue((abs(pred)<=2+1e-8).all())

    def test_query_state_is_shared_and_level_not_erased_by_normalization(self):
        rng=np.random.default_rng(9);bank=dict(x=rng.normal(size=(24,L)),y=rng.normal(size=(24,H)),
            sid=np.zeros(24,dtype=int),start=np.arange(24)*96,std=np.ones(24))
        q=dict(x=rng.normal(size=(3,L)),sid=np.zeros(3,dtype=int),start=np.arange(3)*96+6000,
            std=np.ones(3),training_mean=np.zeros(3),daily_period=np.full(3,24.),series_length=np.full(3,10000))
        r,n=choose_cases(q,bank,13);base=np.zeros((3,H));prior=np.zeros((24,H))
        common,_,_=feature_blocks(q,base,bank,prior,r,n,'query_only',13)
        other,_,_=feature_blocks(q,base,bank,prior,r,n,'real',13)
        np.testing.assert_array_equal(common,other)
        shifted=dict(q,x=q['x']+5)
        changed,_,_=feature_blocks(shifted,base+5,bank,prior,r,n,'query_only',13)
        np.testing.assert_allclose(changed[:,:608],common[:,:608],atol=1e-12)
        np.testing.assert_allclose(changed[:,608]-common[:,608],5.)


if __name__=='__main__':unittest.main()
