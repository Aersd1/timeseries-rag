import tempfile,unittest
from pathlib import Path
import numpy as np
import kernels
from model import LearnedShapeModel,znorm
from index import IndexBuilder,SearchIndex

def reference_dtw(a,b,r):
    n=len(a);prev=np.full(n+1,np.inf);prev[0]=0
    for i in range(1,n+1):
        curr=np.full(n+1,np.inf)
        for j in range(max(1,i-r),min(n,i+r)+1):
            curr[j]=(a[i-1]-b[j-1])**2+min(prev[j-1],prev[j],curr[j-1])
        prev=curr
    return prev[n]

class CertifiedIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng=np.random.default_rng(77)
        train=cls.rng.normal(size=(1000,244)).cumsum(axis=1)
        cls.model=LearnedShapeModel().fit(train)

    def test_sliding_features_match_direct_transform(self):
        rng=np.random.default_rng(33)
        for x in (rng.normal(size=5000),np.cumsum(rng.normal(size=5000)),
                  np.r_[np.zeros(1200),rng.normal(size=1500),np.ones(2300)],
                  1e12+rng.normal(size=5000)*.01):
            codes,flat,feature=kernels.encode(x,self.model,debug=True)
            pos=np.r_[0,1,255,256,257,1000,1199,2000,4000]
            direct=self.model.transform(x[pos[:,None]+np.arange(244)])
            np.testing.assert_allclose(feature[pos],direct,atol=1e-4,rtol=1e-5)
            for i,p in enumerate(pos):
                lo=self.model.left[np.arange(self.model.dim),codes[p]]-1e-4
                hi=self.model.right[np.arange(self.model.dim),codes[p]]+1e-4
                self.assertTrue(np.all((direct[i]>=lo)&(direct[i]<=hi)))

    def test_projection_bounds_ed_and_dtw(self):
        rng=np.random.default_rng(88); x=rng.normal(size=900)
        codes,_,_=kernels.encode(x,self.model)
        for q in (rng.normal(size=244),np.zeros(244)):
            for metric in ('ed','dtw'):
                z,ql,qu,env=self.model.query(q,metric,8)
                lb=kernels.bounds(codes,codes,self.model,ql,qu,metric)
                positions=np.arange(len(codes),dtype=np.int64)
                ds,_=kernels.exact(x,positions,z,metric,8,np.inf,env)
                self.assertTrue(np.all(lb<=ds+1e-6))
                for pos in (0,311):
                    if metric=='dtw':
                        self.assertAlmostEqual(ds[pos],reference_dtw(znorm(x[pos:pos+244]),z,8),places=8)

    def test_tree_exact_matches_exhaustive_and_positions(self):
        rng=np.random.default_rng(19)
        series=[rng.normal(size=5300),rng.normal(size=5300).cumsum(),np.r_[np.zeros(3000),rng.normal(size=2300)]]
        with tempfile.TemporaryDirectory() as temp:
            build=IndexBuilder(temp,self.model,shard_records=1000,leaf_size=64)
            for sid,x in enumerate(series):build.add(x,dict(source=f's{sid}',start=200),np.arange(len(x))*2+7)
            manifest=build.finish();self.assertEqual(manifest['windows'],sum(len(x)-243 for x in series))
            idx=SearchIndex(temp)
            for q in (series[0][4093:4337],series[1][1337:1581]+rng.normal(0,.1,244),np.zeros(244)):
                for metric in ('ed','dtw'):
                    a=idx.search(q,k=5,metric=metric);b=idx.exhaustive(q,k=5,metric=metric)
                    self.assertTrue(a['certified'])
                    np.testing.assert_allclose([h['squared_distance'] for h in a['hits']],
                        [h['squared_distance'] for h in b['hits']],atol=2e-5)
            h=idx.search(series[0][4093:4337],k=1)['hits'][0]
            self.assertEqual(h['start'],4293);self.assertEqual(h['source_row_start'],8193)
            fast=idx.search(rng.normal(size=244),time_budget_ms=.001)
            self.assertFalse(fast['certified']);self.assertTrue(fast['budget_exhausted'])
            # Explicitly release mappings before TemporaryDirectory cleanup on Windows.
            idx.close()

if __name__=='__main__':unittest.main()
