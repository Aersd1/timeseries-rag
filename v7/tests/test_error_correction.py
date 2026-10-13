import unittest
import numpy as np
import torch
from v7.error_correction import ErrorCorrection,retrieve,features,innovations
from v7.trend_rag import ResidualPool
from v7.tests.test_trend_rag import FakeStore


class ErrorCorrectionTests(unittest.TestCase):
    def test_zero_initial_output_and_zero_memory_after_learning(self):
        torch.manual_seed(42);model=ErrorCorrection();q=torch.randn(3,200)
        keys=torch.randn(3,5,96);errors=torch.randn(3,5,96);weights=torch.softmax(torch.randn(3,5),-1)
        torch.testing.assert_close(model(q,keys,errors,weights),torch.zeros(3,96),rtol=0,atol=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        for _ in range(3):
            optimizer.zero_grad();loss=(model(q,keys,errors,weights)-1).square().mean();loss.backward();optimizer.step()
        self.assertGreater(float(model(q,keys,errors,weights).abs().sum().detach()),0.)
        torch.testing.assert_close(model(q,keys,errors*0,weights),torch.zeros(3,96),rtol=0,atol=0)
    def test_candidate_pairs_permutation_and_error_content_matters(self):
        torch.manual_seed(43);model=ErrorCorrection();torch.nn.init.normal_(model.temporal[-1].weight,std=.05)
        q=torch.randn(2,200);k=torch.randn(2,5,96);e=torch.randn(2,5,96);w=torch.ones(2,5)/5
        expected=model(q,k,e,w);order=torch.tensor([4,2,3,0,1])
        torch.testing.assert_close(model(q,k[:,order],e[:,order],w[:,order]),expected)
        self.assertFalse(torch.allclose(model(q,k,e+1,w),expected))
    def test_raw_forecast_projection_and_no_query_overlap(self):
        store=FakeStore();pool=ResidualPool(store,0);x=store.window(0,700,96)
        predicted=np.repeat(x[-1:],96)[None]+np.arange(9)[:,None]*.1
        q,m=retrieve(x,predicted,pool,700,25)
        self.assertTrue(np.all(m['candidate_starts']+192<=700))
        vector=features(x,predicted,q,m,np.ones((5,96)),20.)
        self.assertEqual(vector.shape,(200,));self.assertTrue(np.isfinite(vector).all())
        # Input features require history/predictions/known old errors; no true query future argument exists.
    def test_online_bank_never_uses_query_or_future_values(self):
        first=FakeStore();second=FakeStore();second.x[700:]+=500
        p1=ResidualPool(first,0,end=900,stride=4);p2=ResidualPool(second,0,end=900,stride=4)
        x=first.window(0,700,96);pred=np.repeat(x[-1:],96)[None]+np.arange(9)[:,None]*.1
        q1,m1=retrieve(x,pred,p1,700,25);q2,m2=retrieve(x,pred,p2,700,25)
        np.testing.assert_array_equal(m1['selected_starts'],m2['selected_starts'])
        e1,g1=innovations(x,pred,p1,m1,20.,25,700);e2,g2=innovations(x,pred,p2,m2,20.,25,700)
        np.testing.assert_array_equal(e1,e2);np.testing.assert_array_equal(g1,g2)


if __name__=='__main__':unittest.main()
