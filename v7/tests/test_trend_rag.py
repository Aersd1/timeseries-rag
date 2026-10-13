import unittest
import numpy as np
from v7.trend_rag import decompose, normalize, ResidualPool


class FakeStore:
    def __init__(self):
        t=np.arange(900,dtype=float)
        self.x=(.1*t+np.sin(t/3)).astype(np.float32)
        self.series=[dict(sid=0,runs=[[0,900]],memory_end=700,memory_std=20.,n=900)]
    def window(self,sid,start,length): return self.x[start:start+length].copy()


class TrendRagTests(unittest.TestCase):
    def test_future_cannot_change_history_or_features(self):
        x=np.random.default_rng(1).normal(size=(4,192))
        a,b=decompose(x); x[:,96:]+=1000
        aa,bb=decompose(x)
        np.testing.assert_array_equal(a[:,:96],aa[:,:96])
        np.testing.assert_array_equal(b[:,:96],bb[:,:96])
        np.testing.assert_array_equal(decompose(x[:,:96])[0],a[:,:96])
    def test_prefix_causality_at_every_position(self):
        x=np.arange(192,dtype=float)**2
        trend,_=decompose(x)
        for end in [1,9,25,96,150]:
            np.testing.assert_array_equal(decompose(x[:end])[0],trend[:end])
    def test_exact_reconstruction_constant_and_batch(self):
        x=np.stack([np.full(192,5.),np.arange(192,dtype=float)])
        t,r=decompose(x)
        np.testing.assert_allclose(t+r,x,atol=1e-5)
        np.testing.assert_array_equal(r[0],np.zeros(192))
    def test_retrieval_excludes_self_and_full_overlap(self):
        store=FakeStore();pool=ResidualPool(store,0)
        _,r=decompose(store.window(0,400,96));q,_,_=normalize(r,pool.floor)
        out=pool.retrieve(q,400,np.zeros((3,96)),[.1,.5,.9])
        self.assertEqual(len(out['candidate_starts']),100)
        self.assertTrue(np.all(out['candidate_starts']+192<=400))
        self.assertTrue(np.all(out['selected_starts']+192<=400))
        self.assertNotIn(400,out['candidate_starts'])
        self.assertEqual(out['examples'].shape,(5,192))
        self.assertAlmostEqual(float(out['weights'].sum()),1.,places=5)
    def test_no_relaxation_when_nonoverlapping_pool_too_small(self):
        pool=ResidualPool(FakeStore(),0)
        with self.assertRaises(ValueError): pool.retrieve(np.zeros(96),250,np.zeros((3,96)),[.1,.5,.9])


if __name__=='__main__': unittest.main()
