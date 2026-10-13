import inspect
import unittest
import numpy as np
from v7.residual_adapter import features,train
from v7.benefit_rerank import covariance
from v7.safe_error_policy import bound_delta,choose_policy,apply_policy


class ResidualAdapterTests(unittest.TestCase):
    def test_error_training_recovers_predictable_component_with_purged_folds(self):
        rng=np.random.default_rng(11);n=32
        x=rng.normal(size=(n,96));b=rng.normal(size=(n,96));q=rng.normal(size=(n,9,96));rag=rng.normal(size=(n,96))
        f=features(x,b,q,rag,np.arange(n)*192,np.ones(n),np.zeros(n),24,True)
        target=.5*f[:,:,0]+.3*f[:,:,-3]
        state,selection=train(f,target,np.arange(n)*192)
        self.assertLess(float(np.mean((covariance(state,f)-target)**2)),.03)
        for fold in selection['folds']:self.assertLessEqual(fold['train_last_episode_end'],fold['held_first_start'])
        self.assertEqual(f.shape,(n,96,14))
        self.assertEqual(features(x,b,q,rag,np.arange(n)*192,np.ones(n),np.zeros(n),24,False).shape,(n,96,11))
        self.assertNotIn('target',inspect.signature(features).parameters)

    def test_bound_and_final_guard_rejects_a_harmful_expert(self):
        quantiles=np.repeat(np.arange(9)[None,:,None],12,axis=0);quantiles=np.repeat(quantiles,96,axis=2)
        np.testing.assert_array_equal(bound_delta(np.full((12,96),100),quantiles),np.full((12,96),2))
        target=np.ones((12,96));target[8:]=0;delta=np.full((12,96),2.)
        policy=choose_policy(target,{'case':delta},np.zeros(12,dtype=int),np.arange(12)*192)
        self.assertEqual(policy['0']['family'],'direct');self.assertEqual(policy['0']['proposed_family'],'case')
        np.testing.assert_array_equal(apply_policy(policy,{'case':delta},np.zeros(12,dtype=int)),np.zeros((12,96)))


if __name__=='__main__':unittest.main()
