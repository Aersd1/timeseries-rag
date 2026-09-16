import tempfile,unittest
from pathlib import Path
import numpy as np
from model import LearnedShapeModel,znorm
from index import IndexBuilder
from cascade import Cascade
from diverse import DiverseIndex,Exclusions

def oracle_distances(engine,q):
    """Independent vectorized full distances, no symbol/index pruning."""
    z=znorm(q);out=[]
    for sid in range(len(engine.keys)):
        windows=np.lib.stride_tricks.sliding_window_view(np.asarray(engine.idx.values(sid)),len(q))
        ds=np.sum((znorm(windows)-z)**2,axis=1);ds[ds<1e-20]=0.;out.append(ds)
    return out

def verify_greedy(engine,q,result,separation=244,k=10,limit=np.inf):
    distances=oracle_distances(engine,q);available=[d.copy() for d in distances]
    for d in available:d[d>limit**2]=np.inf
    for hit in result['hits']:
        best=min(float(a.min()) for a in available)
        if abs(best-hit['squared_distance'])>2e-5:raise AssertionError(f'Not nearest remaining: {best}, {hit}')
        sid=hit['sid'];p=hit['local_start']
        if not np.isfinite(available[sid][p]):raise AssertionError('Returned an excluded position')
        if abs(distances[sid][p]-hit['squared_distance'])>2e-5:raise AssertionError('Wrong distance')
        key=engine.keys[sid];global_start=engine.offsets[sid]+p
        for other in range(len(available)):
            if engine.keys[other]==key:
                starts=np.arange(len(available[other]))+engine.offsets[other]
                available[other][abs(starts-global_start)<separation]=np.inf
    if len(result['hits'])<k and any(np.isfinite(a).any() for a in available):raise AssertionError('Stopped before exhaustion')

class DiverseTests(unittest.TestCase):
    def test_bounded_cap_never_claims_completion(self):
        rng=np.random.default_rng(15);model=LearnedShapeModel().fit(rng.normal(size=(30,244)))
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'base';builder=IndexBuilder(base,model)
            builder.add(np.zeros(5000),dict(source='constant.csv',column='x',device='0',start=0),np.arange(5000))
            builder.finish();Cascade.build(base,Path(temp)/'cascade');engine=DiverseIndex(Path(temp)/'cascade')
            try:
                result=engine.search_bounded(np.zeros(244),initial_candidates=10,max_candidates=10)
                self.assertFalse(result['certified']);self.assertFalse(result['complete']);self.assertFalse(result['exhausted'])
                self.assertTrue(result['candidate_limit_reached']);self.assertLess(len(result['hits']),10)
                self.assertEqual(result['max_retained_candidates'],10)
                result=engine.search_bounded(np.zeros(244),initial_candidates=4384,max_candidates=4384)
                self.assertTrue(result['certified']);verify_greedy(engine,np.zeros(244),result)
            finally:engine.close()

    def test_merged_intervals_and_exact_boundary(self):
        e=Exclusions();e.add('x',1000,244)
        self.assertTrue(e.contains('x',1243));self.assertFalse(e.contains('x',1244))
        self.assertTrue(e.contains('x',757));self.assertFalse(e.contains('x',756))
        self.assertFalse(e.contains('other',1000));e.add('x',1244,244)
        self.assertEqual(e.ranges['x'],[(757,1488)])

    def test_exact_greedy_across_chunks_and_columns(self):
        rng=np.random.default_rng(911);x=rng.normal(size=6000);q=x[2950:3194].copy();x[4500:4744]=q
        model=LearnedShapeModel().fit(x[rng.integers(0,5500,100)[:,None]+np.arange(244)])
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'base';builder=IndexBuilder(base,model)
            builder.add(x[:3200],dict(source='same.csv',column='a',device='0',start=10000000000),np.arange(3200))
            builder.add(x[2957:],dict(source='same.csv',column='a',device='0',start=10000002957),np.arange(2957,6000))
            builder.add(x[:1200],dict(source='same.csv',column='b',device='0',start=10000000000),np.arange(1200))
            builder.finish();Cascade.build(base,Path(temp)/'cascade',long_size=2048,mid_size=512)
            engine=DiverseIndex(Path(temp)/'cascade')
            try:
                for query in (q,rng.normal(size=244),np.zeros(244)):
                    for sep in (244,1000):
                        result=engine.search(query,k=8,min_separation=sep)
                        self.assertTrue(result['certified']);verify_greedy(engine,query,result,separation=sep,k=8)
                        prefix=engine.search_prefix(query,k=8,min_separation=sep,initial_candidates=1)
                        self.assertTrue(prefix['certified']);verify_greedy(engine,query,prefix,separation=sep,k=8)
                        bounded=engine.search_bounded(query,k=8,min_separation=sep,initial_candidates=16,max_candidates=8192)
                        self.assertTrue(bounded['certified']);verify_greedy(engine,query,bounded,separation=sep,k=8)
                        self.assertLessEqual(bounded['max_retained_candidates'],8192)
                        self.assertLessEqual(bounded['peak_block_distance_count'],2048)
                result=engine.search(q,k=30,max_distance=0)
                verify_greedy(engine,q,result,k=30,limit=0)
                self.assertTrue(result['exhausted']);self.assertEqual(len(result['hits']),2)
                prefix=engine.search_prefix(q,k=30,max_distance=0,initial_candidates=1)
                verify_greedy(engine,q,prefix,k=30,limit=0)
                other=engine.search_prefix(x[:244],k=20,max_distance=0,initial_candidates=1)
                self.assertEqual({h['column'] for h in other['hits']},{'a','b'})
                empty=engine.search_bounded(rng.normal(size=244),max_distance=0)
                self.assertTrue(empty['certified']);self.assertTrue(empty['exhausted']);self.assertEqual(empty['hits'],[])
                with self.assertRaises(ValueError):engine.search(q,min_separation=243)
                with self.assertRaises(ValueError):engine.search(q,k=0)
                with self.assertRaises(ValueError):engine.search(q[:-1])
            finally:engine.close()

if __name__=='__main__':unittest.main()
