import unittest
import numpy as np
from v7.future_state_encoder import history_features,future_state,prototype_state,train,predict
from v7.query_future_state_rag import FutureStatePredictor


class FakeStore:
    def __init__(self):
        self.x=[np.sin(np.arange(1200)/8).astype(np.float32),np.cos(np.arange(1200)/10).astype(np.float32)]
        self.series=[dict(sid=i,memory_end=600,validation_end=700,memory_mean=0.,memory_std=1.) for i in range(2)]
    def values(self,sid):return self.x[sid]


class FakeAdapter:
    def forecast(self,x,horizon):
        import torch
        return torch.tensor(np.broadcast_to(np.asarray(x).mean(1)[:,None,None]+np.arange(-4,5)[None,:,None],(len(x),9,horizon)).copy(),dtype=torch.float32)


class FakePool:
    floor=.01
    def retrieve(self,*args):
        return dict(candidate_starts=np.arange(100)*4,selected_starts=np.arange(5)*80,
            history_scores=np.ones(100),future_scores=np.ones(100),weights=np.ones(5)/5)


class FutureStateTests(unittest.TestCase):
    def test_shared_encoder_learns_future_states_with_purged_temporal_folds(self):
        rng=np.random.default_rng(13);n=64;x=rng.normal(size=(n,96,3));starts=np.arange(n)*256
        y=x.mean(1)[:,None,:]+np.ones((n,96,3))*.5
        f=history_features(x,starts,np.zeros(3),np.ones(3),24);labels=future_state(x,y,np.ones(3))
        state,selection,_,_=train(f,labels,starts)
        self.assertLess(float(np.mean((predict(state,f)-labels)**2)),.03)
        for fold in selection['folds']:self.assertLessEqual(fold['last_fitted_episode_end'],fold['held_first_start'])
        bad=x.copy();bad[0,0,1]=np.nan
        prototypes=prototype_state(bad,y,np.zeros(3),np.ones(3))
        self.assertFalse(prototypes['mask'][0].reshape(6,3)[:,1].any())
        self.assertTrue(prototypes['mask'][1:].all())

    def test_actual_prediction_is_invariant_to_query_and_sibling_future_poison(self):
        p=object.__new__(FutureStatePredictor);p.store=FakeStore();p.means=np.zeros(2);p.stds=np.ones(2)
        p.synchronous_values=p.store.x;p.adapter=FakeAdapter();p.pools={0:FakePool()}
        p.encoder=dict(mean=np.zeros(26),scale=np.ones(26),coef=np.random.default_rng(22).normal(scale=.05,size=(27,12)),precision=np.ones(12))
        p.base_states={0:dict(mean=np.zeros(28),scale=np.ones(28),coef=np.zeros(29))}
        p.base_run=dict(period=24);p.meta=dict(window=25);p.plan={'0':[1]};p.selection=dict(retrieval_weight=1.)
        p.base_policies={'0':dict(beta=0.,proposed_beta=0.)}
        p.policies={kind:{'0':dict(family='future_state_rag' if kind=='routed' else kind,beta=.5)} for kind in ['encoder_only','future_state_rag','routed']}
        history=p.store.x[0][700:796].copy();a=p.predict(history,0,700)
        for values in p.store.x:values[796:]=1e8
        b=p.predict(history,0,700);self.assertEqual(a,b)
        self.assertTrue(all(start+192<=700 for start in a['selected_starts']))

    def test_grouped_prediction_handles_global_id_in_a_local_channel_zero_slot(self):
        p=object.__new__(FutureStatePredictor);p.store=FakeStore();p.means=np.zeros(2);p.stds=np.ones(2)
        p.synchronous_values=p.store.x;p.adapter=FakeAdapter();p.pools={1:FakePool()};p.grouped=True;p.groups={'1':[1,0]}
        p.encoders={1:dict(mean=np.zeros(30),scale=np.ones(30),coef=np.random.default_rng(24).normal(scale=.05,size=(31,12)),precision=np.ones(12))}
        p.base_states={1:dict(mean=np.zeros(28),scale=np.ones(28),coef=np.zeros(29))}
        p.base_run=dict(period=24);p.meta=dict(window=25);p.plan={'1':[0]};p.selection=dict(groups={'1':dict(retrieval_weight=1.)})
        p.base_policies={'1':dict(beta=0.,proposed_beta=0.)}
        p.policies={kind:{'1':dict(family='future_state_rag' if kind=='routed' else kind,beta=.5)} for kind in ['encoder_only','future_state_rag','routed']}
        history=p.store.x[1][700:796].copy();a=p.predict(history,1,700)
        for values in p.store.x:values[796:]=-1e8
        self.assertEqual(a,p.predict(history,1,700))


if __name__=='__main__':unittest.main()
