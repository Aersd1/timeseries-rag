import unittest
import numpy as np
from v7.dense_forward import layout,prefix_state,PrefixRetriever,complete_origins


class TinyStore:
    def __init__(self):
        t=np.arange(6500);rng=np.random.default_rng(73)
        self.x=np.stack([np.sin(t/12)+.1*rng.normal(size=len(t)),np.cos(t/12)+.1*rng.normal(size=len(t))],1).astype(np.float32)
        self.series=[dict(sid=i,n=len(t),runs=[[0,len(t)]],memory_end=2800,train_end=5200,validation_end=5900,memory_std=1.,memory_mean=0.) for i in range(2)]
    def values(self,sid):return self.x[:,sid]
    def window(self,sid,start,length):
        out=self.x[start:start+length,sid].copy()
        if len(out)!=length or not np.isfinite(out).all():raise ValueError('Invalid/gapped window')
        return out


class DenseForwardTests(unittest.TestCase):
    def test_all_forward_labels_and_expert_labels_respect_cutoffs(self):
        s=TinyStore()
        with self.assertRaises(ValueError):layout(s)
        # The common protocol intentionally refuses a very short TRAIN prefix.
        # Use the same clock scaled to a realistic ETT-style length.
        for v in s.series:v.update(train_end=10452,validation_end=13936)
        p=layout(s)
        for a,b in zip(p['cuts'][:-1],p['cuts'][1:]):
            starts=[x for x in p['oof_starts'] if a<=x and x+192<=b]
            self.assertTrue(starts);self.assertGreaterEqual(min(starts),a);self.assertLessEqual(max(starts)+192,b)
        for cut,starts in p['expert_starts'].items():self.assertLessEqual(max(starts)+192,int(cut))

    def test_prefix_normalization_and_groups_ignore_later_values(self):
        s=TinyStore();a=prefix_state(s,3200);s.x[3200:]=1e9;b=prefix_state(s,3200)
        self.assertEqual(a,b)

    def test_refitted_encoder_ignores_future_after_fit_cutoff(self):
        s=TinyStore();stats=prefix_state(s,3200);a=PrefixRetriever(s,stats,0,24.,16)
        s.x[3200:]=1e9;b=PrefixRetriever(s,stats,0,24.,16)
        for k in a.encoder:np.testing.assert_array_equal(a.encoder[k],b.encoder[k])
        self.assertLessEqual(a.raw.end,3200)

    def test_retrieval_uses_ten_distinct_complete_old_episodes(self):
        s=TinyStore();stats=prefix_state(s,3200);r=PrefixRetriever(s,stats,0,24.,16)
        x=s.window(0,3300,96);q=np.repeat(x[-1:][None],9*96).reshape(9,96)
        selected,random=r.select(x,3300,q)
        self.assertEqual(len(np.unique(selected)),10);self.assertEqual(len(np.unique(random)),10)
        self.assertTrue(np.all(selected+192<=3300));self.assertTrue(np.all(random+192<=3300))
        self.assertFalse(np.isin(random,selected).any())
        for a in [selected[5:],random[5:]]:self.assertTrue(np.all(np.diff(np.sort(a))>=192))

    def test_query_future_does_not_change_case_selection(self):
        s=TinyStore();stats=prefix_state(s,3200);r=PrefixRetriever(s,stats,0,24.,16)
        x=s.window(0,3300,96);q=np.zeros((9,96),np.float32);a=r.select(x,3300,q)
        s.x[3396:]=np.nan;b=r.select(x,3300,q)
        for u,v in zip(a,b):np.testing.assert_array_equal(u,v)

    def test_missing_future_windows_are_excluded_consistently(self):
        s=dict(runs=[[0,500],[700,1500]])
        np.testing.assert_array_equal(complete_origins(s,[0,400,600,700,1308,1309]),[0,700,1308])

    def test_missing_observed_sibling_is_masked_without_dropping_valid_query(self):
        s=TinyStore();stats=prefix_state(s,3200);r=PrefixRetriever(s,stats,0,24.,16)
        s.x[3310:3320,1]=np.nan
        chosen,_=r.select(s.window(0,3300,96),3300,np.zeros((9,96),np.float32))
        self.assertEqual(len(chosen),10)


if __name__=='__main__':unittest.main()
