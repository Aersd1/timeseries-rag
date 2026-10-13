import inspect
import unittest
import numpy as np
from v7.benefit_rerank import pair_features,fit_ridge,covariance,select_candidates,guarded_policy,apply_policy


class BenefitTests(unittest.TestCase):
    def test_calibration_cannot_override_failed_final_guard(self):
        target=np.ones((18,96));target[12:]=0;correction=np.ones((18,96))*2
        p=guarded_policy(target,correction,np.zeros(18,dtype=int),np.arange(18)*192)
        self.assertEqual(p['0']['proposed_beta'],.5);self.assertEqual(p['0']['beta'],0.)
        np.testing.assert_array_equal(apply_policy(p,correction,np.zeros(18,dtype=int)),np.zeros((18,96)))
    def test_consistent_benefit_adopted_but_small_gain_rejected(self):
        sid=np.zeros(18,dtype=int);starts=np.arange(18)*192;target=np.ones((18,96))
        self.assertEqual(guarded_policy(target,target*2,sid,starts)['0']['beta'],.5)
        self.assertEqual(guarded_policy(target,target*.00001,sid,starts)['0']['beta'],0.)
    def test_utility_rejects_destructive_vectors_despite_positive_covariance(self):
        e=np.ones((100,96))*10;cov=np.ones(100)*.1
        r=select_candidates(cov,e,np.zeros(100),np.arange(100)*192,0)
        self.assertFalse(r['permitted']);np.testing.assert_array_equal(r['correction'],np.zeros(96))
        e[:10]=1;cov[:10]=1
        r=select_candidates(cov,e,np.zeros(100),np.arange(100)*192,0)
        self.assertTrue(r['permitted']);self.assertTrue(np.all(r['selected']<10))
    def test_feature_api_has_no_query_future_and_enforces_complete_episode(self):
        self.assertNotIn('target',inspect.signature(pair_features).parameters)
        args=dict(history=np.arange(96),base=np.arange(96),query_start=1000,candidate_starts=np.arange(100)*2,
            candidate_histories=np.zeros((100,96)),candidate_residual_histories=np.zeros((100,96)),values=np.zeros((100,96)),
            future_scores=np.ones(100),history_scores=np.ones(100),memory_mean=0,memory_std=1,period=24,window=25)
        self.assertEqual(pair_features(**args).shape,(100,59))
        args['candidate_starts'][0]=900
        with self.assertRaises(ValueError):pair_features(**args)
    def test_ridge_serialization_predictions_match(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(500,6));y=x[:,0]-2*x[:,1]
        state=fit_ridge(x,y,.01);p=covariance(state,x)
        self.assertLess(float(np.mean((p-y)**2)),.05)
        np.testing.assert_array_equal(covariance({k:np.array(v,copy=True) for k,v in state.items()},x),p)


if __name__=='__main__':unittest.main()
