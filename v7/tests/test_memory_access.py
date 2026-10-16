import unittest
import numpy as np
from v7.memory_access import CONTEXT,align_pack,select_clock,history_controls


class MemoryAccessTests(unittest.TestCase):
    def test_clock_controls_complete_cases_and_distinct_episodes(self):
        starts=np.arange(5000)
        for mode in ('recent','periodic'):
            selected=select_clock(starts,5000,4000,24,mode)
            self.assertEqual(len(np.unique(selected)),10)
            self.assertTrue(np.all(selected+192<=4000))
            self.assertTrue(np.all(np.diff(np.sort(selected))>=192))
        periodic=select_clock(starts,5000,4000,24,'periodic')
        np.testing.assert_array_equal((5000-periodic)%24,0)

    def test_packing_contains_both_actual_histories_and_futures(self):
        rng=np.random.default_rng(3);x=rng.normal(size=96);past=rng.normal(size=(10,96));future=rng.normal(size=(10,96))
        starts=np.arange(10)*192
        result=align_pack(x,past,future,starts)
        self.assertEqual(result.shape,(CONTEXT,))
        np.testing.assert_array_equal(result[-96:],x.astype(np.float32))
        changed=align_pack(x,past[:,::-1],future,starts)
        self.assertGreater(np.max(abs(result-changed)),.1)
        # Future values cannot change a history-only fitted alignment of old pasts.
        changed_future=align_pack(x,past,future+100,starts).reshape(-1)
        np.testing.assert_array_equal(result[:-96].reshape(10,192)[:,:96],
                                      changed_future[:-96].reshape(10,192)[:,:96])

    def test_single_channel_selection_cannot_read_current_future(self):
        rng=np.random.default_rng(4);values=rng.normal(size=6500);starts=np.arange(4000)
        x=values[5000:5096].copy()
        first=history_controls(values,starts,x,5000,4000)
        values[5096:]=1e15
        np.testing.assert_array_equal(first,history_controls(values,starts,x,5000,4000))
        self.assertEqual(len(np.unique(first)),10)
        self.assertTrue(np.all(first+192<=4000))

    def test_pair_shuffle_keeps_values_histories_and_query(self):
        from v7.memory_shuffle import shuffle_futures
        x=np.arange(2*CONTEXT,dtype=np.float32).reshape(2,CONTEXT)
        y=shuffle_futures(x)
        np.testing.assert_array_equal(np.sort(x,axis=1),np.sort(y,axis=1))
        np.testing.assert_array_equal(x[:,-96:],y[:,-96:])
        a=x[:,:1920].reshape(2,10,192);b=y[:,:1920].reshape(2,10,192)
        np.testing.assert_array_equal(a[:,:,:96],b[:,:,:96])
        self.assertTrue(np.all(np.any(a[:,:,96:]!=b[:,:,96:],axis=2)))


if __name__=='__main__':unittest.main()
