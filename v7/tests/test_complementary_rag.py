import inspect
import unittest
import numpy as np
from v7.complementary_rag import forward_base,candidate_innovations
from v7.benefit_rerank import fit_ridge
from v7.benchmark_state_recall import corrections


class ComplementaryTests(unittest.TestCase):
    def test_outer_future_labels_cannot_change_teacher_predictions_or_inner_selection(self):
        rng=np.random.default_rng(7);n=64
        features=rng.normal(size=(n,96,3));target=.5*features[:,:,0]+rng.normal(scale=.1,size=(n,96))
        starts=np.arange(n)*256;base=np.zeros((n,96));q=np.repeat(np.linspace(-4,4,9)[None,:,None],n,axis=0);q=np.repeat(q,96,axis=2)
        a=forward_base(features,target,starts,base,q,np.ones(n),.5)
        target[48:]=1e8
        b=forward_base(features,target,starts,base,q,np.ones(n),.5)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[2],b[2])
        for fold in a[2]:self.assertLessEqual(fold['last_fitted_episode_end'],fold['held_first_start'])

    def test_conditioned_utility_distinguishes_identical_history_opposite_continuations(self):
        n=16;context=np.where(np.arange(n)%2,1.,-1.);v=np.tile(np.r_[-np.ones(50),np.ones(50)],(n,1))
        features=np.stack([context[:,None]*v,v],axis=-1)
        values=np.repeat(v[:,:,None],96,axis=-1);labels=context[:,None]*v
        state=fit_ridge(features.reshape(-1,2),labels.reshape(-1),.1)
        table=dict(features=features,values=values,prior=np.ones((n,100)),starts=np.tile(np.arange(100)*192,(n,1)))
        r=corrections(state,table)
        self.assertTrue(r['permitted'].all())
        np.testing.assert_array_equal(np.sign(r['correction'].mean(1)),context)

    def test_innovation_requires_no_query_future_and_preserves_level_transfer(self):
        history=np.arange(96,dtype=float)+100;past=np.repeat(np.arange(96,dtype=float)[None],5,axis=0)
        future=np.repeat(np.arange(96,192,dtype=float)[None],5,axis=0);base=np.arange(96,192,dtype=float)+100
        np.testing.assert_allclose(candidate_innovations(history,base,past,future,1),0,atol=1e-5)
        self.assertNotIn('target',inspect.signature(candidate_innovations).parameters)


if __name__=='__main__':unittest.main()
