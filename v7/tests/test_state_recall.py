import unittest
import numpy as np
from v7.state_recall import history_state,state_distance,StateRecall
from v7.benchmark_state_recall import corrections
from v7.benchmark_benefit_rerank import corrections as original_corrections
from v7.error_correction import retrieve


class StateRecallTests(unittest.TestCase):
    def test_state_distinguishes_level_scale_and_trend(self):
        x=np.sin(np.arange(96)*2*np.pi/24)
        q=history_state(x,1)
        states=history_state(np.stack([x,x+4,x*4,x+np.arange(96)/24]),1)
        d=state_distance(q,states,100,np.repeat(100,4),24)
        self.assertEqual(d[0],0);self.assertTrue(np.all(d[1:]>0))

    def test_phase_is_circular_and_weekly_omits_day_phase(self):
        d=state_distance(np.zeros(3),np.zeros((3,3)),24,np.array([0,23,12]),24)
        self.assertAlmostEqual(d[0],0);self.assertLess(d[1],d[2])
        np.testing.assert_array_equal(state_distance(np.zeros(3),np.zeros((3,3)),24,np.array([0,23,12]),None),np.zeros(3))

    def test_zero_weight_reproduces_old_recall_and_excludes_overlap(self):
        class Pool:
            pass
        p=Pool();rng=np.random.default_rng(4);p.starts=np.arange(150);p.sid=0;p.memory_end=400;p.floor=.01
        p.past=rng.normal(size=(150,96)).astype(np.float32);p.future=rng.normal(size=(150,96)).astype(np.float32)
        from v7.trend_rag import ResidualPool
        p.retrieve=lambda *args:ResidualPool.retrieve(p,*args)
        histories=rng.normal(size=(150,96)).astype(np.float32)
        x=rng.normal(size=96).astype(np.float32);quantiles=rng.normal(size=(9,96)).astype(np.float32)
        new=StateRecall(p,histories,1,24).retrieve(x,quantiles,300,0)
        _,old=retrieve(x,quantiles,p,300,25)
        for k in ['candidate_starts','history_scores','future_scores','selected_starts','weights']:
            np.testing.assert_allclose(new[k],old[k],rtol=0,atol=1e-7)
        self.assertTrue(np.all(new['candidate_starts']+192<=300))
        # Continuations may affect reranking, but never history-only recall.
        before=new['candidate_starts'].copy();p.future*=100
        np.testing.assert_array_equal(StateRecall(p,histories,1,24).retrieve(x,quantiles,300,0)['candidate_starts'],before)
        with self.assertRaises(ValueError):StateRecall(p,histories,1,24).retrieve(x,quantiles,280,.2)

    def test_vectorized_ranking_matches_legacy_including_ties(self):
        rng=np.random.default_rng(7);n=30
        state=dict(mean=np.zeros(3),scale=np.ones(3),coef=np.array([0,1.,2.,-.5]))
        table=dict(features=rng.normal(size=(n,100,3)),values=rng.normal(size=(n,100,96)),
            prior=rng.random((n,100)),starts=np.tile(np.arange(100)*16,(n,1)),sid=np.arange(n),target=np.zeros((n,96)))
        table['features'][0]=0;table['values'][0]=1;table['prior'][0]=0
        expected=original_corrections(state,table);actual=corrections(state,table)
        for k in ['selected','permitted']:np.testing.assert_array_equal(actual[k],expected[k])
        for k in ['weights','correction']:np.testing.assert_allclose(actual[k],expected[k],rtol=0,atol=2e-7)


if __name__=='__main__':unittest.main()
