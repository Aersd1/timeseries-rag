import inspect
import unittest
import numpy as np
from v7.local_error_rag import LocalErrorRAG,independent_nearest
from v7.benchmark_local_error_rag import walk


class LocalErrorTests(unittest.TestCase):
    def fixture(self):
        class Pool:pass
        rng=np.random.default_rng(8);p=Pool();p.starts=np.arange(150);p.sid=0;p.memory_end=400;p.floor=.01
        p.past=rng.normal(size=(150,96)).astype(np.float32);p.future=rng.normal(size=(150,96)).astype(np.float32)
        raw=rng.normal(size=(150,96)).astype(np.float32)
        return LocalErrorRAG(p,raw,np.ones((150,96),dtype=np.float32),1,24,
            [dict(mode='compatibility',weight=.2,anchors=8)])

    def test_observation_requires_complete_episode_and_clock_never_rewinds(self):
        m=self.fixture();x=np.zeros(96,dtype=np.float32)
        with self.assertRaises(ValueError):m.observe(500,x,x,x,np.ones((1,96)),691)
        m.observe(500,x,x,x,np.ones((1,96)),692)
        with self.assertRaises(ValueError):m.predict(x,np.zeros((9,96)),691)
        self.assertNotIn('future',inspect.signature(m.predict).parameters)

    def test_feedback_cases_are_pairwise_nonoverlapping(self):
        starts=np.array([0,32,64,192,224,384]);scores=np.arange(6)
        ids=independent_nearest(scores,starts,5)
        np.testing.assert_array_equal(starts[ids],[0,192,384])

    def test_future_poisoning_cannot_change_earlier_inputs_or_selection(self):
        rng=np.random.default_rng(3);data=dict(start=np.array([500,550,800]),raw_x=rng.normal(size=(3,96)).astype(np.float32),
            raw_quantiles=rng.normal(size=(3,9,96)).astype(np.float32),raw_y=rng.normal(size=(3,96)).astype(np.float32),
            base=np.zeros((3,96),dtype=np.float32),std=np.ones(3,dtype=np.float32))
        other={k:v.copy() for k,v in data.items()};other['raw_y'][:]=1e6
        first,trace,_=walk(self.fixture(),data,[0,1,2]);second,changed,_=walk(self.fixture(),other,[0,1,2])
        np.testing.assert_array_equal(first[:2],second[:2])
        for i in [0,1]:
            np.testing.assert_array_equal(trace[i]['potentials'],changed[i]['potentials'])
            np.testing.assert_array_equal(trace[i]['selected_starts'],changed[i]['selected_starts'])
            self.assertEqual(trace[i]['observed_queries'],0)
        self.assertEqual(trace[2]['observed_queries'],2)

    def test_consistent_past_corrections_adopted_and_harmful_direction_rejected(self):
        models=[self.fixture(),self.fixture()];x=np.sin(np.arange(96)/4).astype(np.float32);end=500+192*12
        for sign,m in zip([1,-1],models):
            for i in range(12):m.observe(500+i*192,x,np.ones(96),np.zeros(96),np.full((1,96),2*sign),end)
        self.assertEqual(models[0].predict(x,np.zeros((9,96)),end)['beta'][0],.5)
        self.assertEqual(models[1].predict(x,np.zeros((9,96)),end)['beta'][0],0.)


if __name__=='__main__':unittest.main()
