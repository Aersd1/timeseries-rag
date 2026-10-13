import unittest
import numpy as np
from v7.contextual_adapter import context_plan
from v7.past_error_adapter import feature,rerank,fit_completed
from v7.safe_error_policy import choose_policy


class FakeStore:
    def __init__(self):
        rng=np.random.default_rng(23);self.x=[rng.normal(size=1200).astype(np.float32) for _ in range(5)]
        self.series=[dict(sid=i,source='same',memory_end=600,memory_mean=0.,memory_std=1.) for i in range(5)]
    def values(self,sid):return self.x[sid]
    def window(self,sid,start,length):return self.x[sid][start:start+length].copy()


class PastErrorTests(unittest.TestCase):
    def test_pending_or_overlapping_feedback_cannot_change_online_fit(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(8,96,5));y=rng.normal(size=(8,96));starts=np.arange(8)*192
        a=fit_completed(x,y,starts,5*192,1.)
        x[5:]=1e8;y[5:]=-1e8
        b=fit_completed(x,y,starts,5*192,1.)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_query_truth_and_unobserved_channels_cannot_change_inputs(self):
        store=FakeStore();plan=context_plan(store);p=700
        d=dict(raw_x=store.window(0,p,96)[None],base=np.zeros((1,96)),raw_quantiles=np.zeros((1,9,96)),
            start=np.array([p]),std=np.ones(1),weights=np.full((1,5),.2),errors=np.zeros((1,5,96)),
            selected_starts=np.array([[0,80,160,240,320]]),raw_y=np.zeros((1,96)))
        error=np.ones((1,32));a=feature(store,d,[0],store.series[0],24,plan,error)
        self.assertEqual(a.shape,(1,96,51))
        d['raw_y'][:]=999
        for x in store.x:x[p+96:]=1e9
        np.testing.assert_array_equal(a,feature(store,d,[0],store.series[0],24,plan,error))
        self.assertEqual(plan,context_plan(store))
        d['selected_starts'][0,0]=p-191
        with self.assertRaises(ValueError):feature(store,d,[0],store.series[0],24,plan,error)

    def test_error_state_can_select_matching_known_error_without_future_target(self):
        starts=np.arange(100)*2;memory=np.repeat(np.linspace(-2,2,100)[:,None],96,axis=1)
        data=dict(candidate_starts=starts[None],start=np.array([500]),std=np.ones(1),sid=np.zeros(1,dtype=int),
            history_scores=np.ones((1,100)),future_scores=np.ones((1,100)))
        a=rerank(data,[0],np.full((1,32),2.),starts,memory,1.)
        self.assertGreater(float(a['errors'].mean()),1.)
        data['start'][0]=200
        with self.assertRaises(ValueError):rerank(data,[0],np.zeros((1,32)),starts,memory,1.)

    def test_mae_guard_rejects_mse_only_improvement(self):
        target=np.zeros((12,96));target[:,0]=24
        delta=np.ones_like(target)
        args=(target,{'case':delta},np.zeros(12,dtype=int),np.arange(12)*192)
        self.assertGreater(choose_policy(*args)['0']['beta'],0)
        self.assertEqual(choose_policy(*args,require_mae=True)['0']['beta'],0)


if __name__=='__main__':unittest.main()
