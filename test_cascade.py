import tempfile,unittest
from pathlib import Path
import numpy as np
from model import LearnedShapeModel
from index import IndexBuilder
from cascade import Cascade
import kernels

class CascadeTests(unittest.TestCase):
    def test_cached_distances_match_oracle(self):
        rng=np.random.default_rng(44);q=rng.normal(size=244);q=(q-q.mean())/q.std()
        for x in (rng.normal(size=1000),1e12+rng.normal(size=1000)*.01,np.zeros(1000)):
            starts=np.arange(len(x)-243,dtype=np.int64);stats=kernels.local_stats(x,244)
            for cutoff in (np.inf,10.,300.):
                direct=kernels.exact(x,starts,q,cutoff=cutoff)[0]
                cached=kernels.exact_cached(x,starts,q,stats,cutoff=cutoff)
                np.testing.assert_array_equal(cached<=cutoff,direct<=cutoff)
                keep=direct<=cutoff
                np.testing.assert_allclose(cached[keep],direct[keep],atol=1e-10,rtol=1e-12)

    def test_boundaries_exact_distances_and_input(self):
        rng=np.random.default_rng(412);x=rng.normal(size=19000)
        model=LearnedShapeModel().fit(x[rng.integers(0,17000,100)[:,None]+np.arange(244)])
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'base';builder=IndexBuilder(base,model)
            builder.add(x,dict(source='synthetic',column='x',start=10000000000),np.arange(len(x))*2)
            builder.finish();Cascade.build(base,Path(temp)/'cascade');c=Cascade(Path(temp)/'cascade')
            for start in (1020,16380,18756):
                q=x[start:start+244];r=c.search(q,long_candidates=100,mid_candidates=100,k=5,verification='none')
                self.assertFalse(r['certified'])
                truth=c.idx.exhaustive(q,k=5)
                np.testing.assert_allclose([h['squared_distance'] for h in r['hits']],
                    [h['squared_distance'] for h in truth['hits']],atol=2e-5)
                self.assertEqual(r['hits'][0]['start'],10000000000+start)
                self.assertEqual(r['hits'][0]['source_row_start'],2*start)
                # Deliberately restrict the approximate stage; verification must
                # recover global neighbors even outside the retained regions.
                for method in ('scan','tree'):
                    verified=c.search(q,k=5,long_candidates=1,mid_candidates=1,verification=method)
                    self.assertTrue(verified['certified'])
                    np.testing.assert_allclose([h['squared_distance'] for h in verified['hits']],
                        [h['squared_distance'] for h in truth['hits']],atol=2e-5)
            q=rng.normal(size=244);z,ql,qu,_=model.query(q)
            lookup=np.maximum(np.maximum(model.left-1e-4-ql[:,None],ql[:,None]-model.right-1e-4),0.)**2
            codes=c.codes[:10000];lower=kernels.bounds(codes,codes,model,ql,qu)
            for cutoff in (0.,float(np.median(lower)),np.inf):
                kept=kernels.filter_codes(codes,lookup,model.pairs*2,cutoff)
                np.testing.assert_array_equal(kept,np.flatnonzero(lower<=cutoff+1e-7))
            result=c.search(q,k=5,long_candidates=1,mid_candidates=1,verification='scan')
            truth=c.idx.exhaustive(q,k=5)
            np.testing.assert_allclose([h['squared_distance'] for h in result['hits']],
                [h['squared_distance'] for h in truth['hits']],atol=2e-5)
            with self.assertRaises(ValueError):c.search(np.zeros(243))
            with self.assertRaises(ValueError):c.search(np.full(244,np.nan))
            with self.assertRaises(ValueError):c.search(x[:244],mid_candidates=0)
            c.idx.close()

if __name__=='__main__':unittest.main()
