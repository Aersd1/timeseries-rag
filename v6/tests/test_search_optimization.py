import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from v6.index import Candidates, Library, diverse, boxes
from v6.native_search import LIB, NativeCandidates
from v6.inference import Retriever
from v6.data import scale_floor
from v6.common import write_json
from test_core import small_config


class SearchOptimizationTests(unittest.TestCase):
    @unittest.skipIf(LIB is None, 'Optional C heap not built')
    def test_native_heap_matches_exhaustive_including_ties(self):
        rng=np.random.default_rng(82)
        ds=rng.integers(0,20,2200).astype(float); starts=np.arange(len(ds)); sids=rng.integers(0,4,len(ds))
        for capacity in (1,31,512,3000):
            native=NativeCandidates(capacity); numpy=Candidates(capacity)
            for sid in range(4):
                ids=np.flatnonzero(sids==sid); rng.shuffle(ids)
                for a in range(0,len(ids),39):
                    ii=ids[a:a+39]; native.add(ds[ii],sid,starts[ii]); numpy.add(ds[ii],sid,starts[ii])
                    self.assertEqual(native.hits(),numpy.hits())
                    self.assertEqual(native.threshold,numpy.threshold)
            order=np.lexsort((starts,sids,ds))[:capacity]
            self.assertEqual(native.hits(),[dict(distance=float(ds[i]),sid=int(sids[i]),start=int(starts[i])) for i in order])
        with self.assertRaises(ValueError): NativeCandidates(0)

    def test_vectorized_diversity_preserves_greedy_order(self):
        rng=np.random.default_rng(86)
        hits=[dict(sid=int(rng.integers(0,5)),start=int(rng.integers(0,10000)),distance=float(i)) for i in range(1200)]
        for k in (1,10,200):
            expected=[]
            for h in hits:
                if all(h['sid']!=old['sid'] or abs(h['start']-old['start'])>=244 for old in expected):
                    expected.append(h)
                    if len(expected)==k: break
            self.assertEqual(diverse(hits,k,244),expected)
        self.assertEqual(diverse([],10,244),[])

    def test_numpy_fallback_and_native_budget_parity(self):
        rng=np.random.default_rng(87); c=small_config(); c['index']['channels']=['learned']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); shards=[]
            for sid in range(2):
                path=root/str(sid); cp=path/'learned'; cp.mkdir(parents=True)
                vectors=rng.normal(size=(257,6)).astype('f4'); starts=np.arange(257,dtype=np.int64)*10
                np.save(cp/'vectors.npy',vectors); np.save(path/'starts.npy',starts)
                levels=boxes(cp,vectors,16,4)
                shards.append(dict(path=str(sid),sid=sid,group='test',n=len(starts),levels={'learned':levels}))
            write_json(root/'manifest.json',dict(version=6,complete=True,config=c,shards=shards))
            lib=Library(root,cache_size=2)
            try:
                for budget in (0,1,7):
                    for sid in (None,0):
                        kw=dict(k=4,capacity=35,query_start=100,query_sid=0,sid=sid,leaf_budget=budget)
                        q=rng.normal(size=6)
                        expected,stats=lib.search(q,candidate_backend='numpy',**kw)
                        with patch('v6.native_search.LIB',None):
                            fallback,fs=lib.search(q,**kw)
                        self.assertEqual(expected,fallback); self.assertEqual(fs['candidate_backend'],'numpy')
                        if LIB is not None:
                            actual,ns=lib.search(q,candidate_backend='native',**kw)
                            self.assertEqual(expected,actual)
                            for field in ('leaves','nodes','scored','certified','candidates_retained'):
                                self.assertEqual(stats[field],ns[field])
                self.assertTrue(lib.geometry)
            finally: lib.close()
            self.assertFalse(lib.geometry)

    def test_raw_history_gate_matches_scalar_reference(self):
        c=small_config(); c['evaluation'].update(k=7,history_max_nmse=.5,scope='all',time_budget_ms=0)
        series_values=[np.sin(np.arange(1000)*.13+i*.3).astype('f4') for i in range(2)]
        class Store:
            series=[dict(memory_std=1.,group='test') for _ in range(2)]
            def values(self,sid): return series_values[sid]
            def window(self,sid,start,length): return series_values[sid][start:start+length]
        store=Store(); x=series_values[0][800:840]
        hits=[dict(sid=i%2,start=(i*11)%800,distance=float(i+1)) for i in range(240)]
        q=x.astype(float); qz=(q-q.mean())/max(q.std(),scale_floor(store.series[0],c))
        expected=[]
        for hit in hits:
            if any(hit['sid']==old['sid'] and abs(hit['start']-old['start'])<40 for old in expected): continue
            past=store.window(hit['sid'],hit['start'],40).astype(float)
            z=(past-past.mean())/max(past.std(),scale_floor(store.series[hit['sid']],c))
            error=float(np.mean((z-qz)**2))
            if error<=.5:
                expected.append(dict(hit,history_nmse=error))
                if len(expected)==7: break
        class FakeLibrary:
            def search(self,*args,**kwargs): return kwargs['candidate_filter'](hits),{}
        engine=Retriever.__new__(Retriever); engine.c=c; engine.store=store; engine.library=FakeLibrary()
        actual,stats,_,_=engine.search_encoded(x,0,800,np.zeros(6),'joint',0)
        self.assertEqual(actual,expected)
        self.assertGreater(stats['history_checked'],0)


if __name__=='__main__': unittest.main()
