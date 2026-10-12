import inspect
import unittest
import numpy as np
import torch
from v7.future_cluster import (FutureClusterEncoder,eligible,future_features,fit_clusters,
    assignments,encoder_loss,rank_from_history,row_loss)


class FutureClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_full_episode_exclusion_and_half_open_boundary(self):
        mask=eligible([0],[100],[0,0,0,0,0,1],[100,99,70,60,59,0],40)[0]
        np.testing.assert_array_equal(mask,[False,False,False,True,True,False])
        # A history ending before the query is STILL forbidden if its future overlaps.
        self.assertFalse(eligible([0],[100],[0],[70],40)[0,0])

    def test_future_features_preserve_order_and_history_only_scaling(self):
        x=torch.arange(40).float()[None]; y=torch.arange(40,56).float()[None]; floor=torch.ones(1)*.1
        a=future_features(x,y,floor,[4,8,16])
        b=future_features(x,y.flip(-1),floor,[4,8,16])
        self.assertFalse(torch.equal(a,b))
        torch.testing.assert_close(a,future_features(x*3+10,y*3+10,floor*3,[4,8,16]),rtol=1e-5,atol=1e-5)
        self.assertEqual(a.shape,(1,44))
        # Teacher distance must describe the same mean/scale mapping as analog prediction.
        mean=x.mean(); scale=x.std(correction=0)
        torch.testing.assert_close(a[:,:4]*np.sqrt(3*4),(y[:,:4]-mean)/scale)

    def test_training_and_inference_have_no_pretrained_dependency(self):
        c=dict(length=40,horizons=[4,8,16],width=32,dim=8,clusters=4,dropout=0.)
        model=FutureClusterEncoder(c); x=torch.randn(6,40); y=torch.randn(6,16); floor=torch.ones(6)*.1
        feature=future_features(x,y,floor,c['horizons']); centers=feature[:4].clone()
        teacher=assignments(feature,centers,.5)
        batch=dict(x=x,y=y,floor=floor,candidates=torch.arange(6)[None].repeat(6,1),
            eligible=~torch.eye(6,dtype=torch.bool),future_distance=torch.cdist(feature,feature).square())
        with torch.no_grad(): bank=model(x,floor)['z']
        loss,parts=encoder_loss(model,batch,bank,teacher,centers,.5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for prefix in ['backbone','embedding','cluster','forecast']:
            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
                if n.startswith(prefix) and p.grad is not None),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(list(inspect.signature(model.forward).parameters),['x','floor'])
        model.eval()
        with torch.no_grad(): a=model(x,floor)
        second=FutureClusterEncoder(c); second.load_state_dict(model.state_dict()); second.eval()
        with torch.no_grad(): torch.testing.assert_close(second(x,floor)['probability'],a['probability'],rtol=0,atol=0)

    def test_excluded_pairs_have_zero_loss_gradient(self):
        logits=torch.randn(2,4,requires_grad=True); valid=torch.tensor([[True,True,False,False],[False]*4])
        loss=row_loss(logits,torch.ones(2,4),valid); loss.backward()
        self.assertEqual(float(logits.grad[~valid].abs().sum()),0.)
        self.assertGreater(float(logits.grad[valid].abs().sum()),0.)
        self.assertEqual(float(row_loss(logits,torch.ones(2,4),valid&False)),0.)

    def test_future_regime_can_separate_identical_histories(self):
        out=dict(z=np.array([1.,0.]),probability=np.array([.99,.01]),normalized_history=np.zeros(40),center=0.,scale=1.)
        past=np.zeros((2,40)); future=np.array([np.ones(16)*-2,np.ones(16)*2])
        prob=np.array([[.01,.99],[.99,.01]])
        result=rank_from_history(out,past,future,np.ones((2,2)),prob,'future_cluster',1)
        self.assertEqual(int(result['indices'][0]),1)
        np.testing.assert_allclose(result['prediction'],2.)
        # No unknown query future exists in this retrieval interface.
        self.assertNotIn('y',inspect.signature(rank_from_history).parameters)

    def test_clusters_fit_and_probabilities_are_finite(self):
        rng=np.random.default_rng(9); feature=np.r_[rng.normal(-3,.1,(12,8)),rng.normal(3,.1,(12,8))].astype(np.float32)
        centers,temp=fit_clusters(feature,2,9)
        p=assignments(torch.as_tensor(feature),torch.as_tensor(centers),temp)
        torch.testing.assert_close(p.sum(-1),torch.ones(24))
        self.assertTrue(torch.isfinite(p).all())
        self.assertGreater(float((p.max(-1).values>.9).float().mean()),.9)

    def test_history_shortlist_respects_history_before_future_ranking(self):
        out=dict(z=np.array([1.,0.]),probability=np.array([.99,.01]),normalized_history=np.zeros(40),
            center=0.,scale=1.,normalized_forecast=np.zeros(16),horizons=[4,8,16])
        past=np.r_[np.zeros((50,40)),np.ones((10,40))*10]
        future=np.r_[np.ones((50,16)),np.zeros((10,16))]
        probabilities=np.r_[np.tile([.5,.5],(50,1)),np.tile([.99,.01],(10,1))]
        for mode in ['history_future_cluster','history_future_regression']:
            r=rank_from_history(out,past,future,np.ones((60,2)),probabilities,mode)
            self.assertTrue((r['indices']<50).all())
            self.assertTrue((r['recall']<50).all())


if __name__=='__main__': unittest.main()
