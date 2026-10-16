import unittest
import numpy as np
from v7.local_pair_adaptation import local_update,transfer_check,train_global,forecasts


class LocalPairAdaptationTests(unittest.TestCase):
    def test_pair_relationship_changes_local_solution(self):
        rng=np.random.default_rng(8);phi=rng.normal(size=(2,10,4));truth=rng.normal(size=(4,12))
        target=phi@truth;prior=np.zeros((4,12))
        result=local_update(phi,target,prior,1e-9)
        np.testing.assert_allclose(result,np.broadcast_to(truth,result.shape),atol=1e-8)
        corrupt=local_update(phi,np.roll(target,1,axis=1),prior,1e-9)
        self.assertGreater(float(np.mean((corrupt-truth)**2)),.1)

    def test_no_case_means_no_update(self):
        phi=np.ones((2,10,4));target=np.ones((2,10,12));prior=np.ones((4,12))
        np.testing.assert_array_equal(local_update(phi,target,prior,10.,np.zeros((2,10))),0.)

    def test_overlapping_case_copies_cannot_validate_each_other(self):
        rng=np.random.default_rng(9);phi=rng.normal(size=(2,10,4));target=rng.normal(size=(2,10,12))
        starts=np.broadcast_to(np.arange(10),(2,10))
        np.testing.assert_array_equal(transfer_check(phi,target,starts,np.zeros((4,12)),10.),False)

    def data(self,n):
        rng=np.random.default_rng(2)
        d=dict(raw_x=rng.normal(size=(n,96)),raw_y=rng.normal(size=(n,96)),
               raw_quantiles=np.broadcast_to(np.arange(9)[None,:,None]-4,(n,9,96)).astype(float).copy(),
               sid=np.zeros(n,dtype=int),start=np.arange(n)*192+3000)
        starts=np.broadcast_to(np.arange(10)*192,(n,10)).copy()
        for prefix in ('','random_'):
            d[prefix+'selected_starts']=starts.copy()
            for key in ('case_past','case_future','case_baseline'):d[prefix+key]=rng.normal(size=(n,10,96))
        return d

    def test_prediction_uses_real_old_history_not_current_future(self):
        stats=dict(memory_std=[1.],memory_mean=[0.]);train=self.data(40)
        model=train_global(train,stats,24);data=self.data(2)
        first,_=forecasts(model,data,stats,24,penalties=(10.,))
        poison=dict(data,raw_y=np.full((2,96),1e20))
        second,_=forecasts(model,poison,stats,24,penalties=(10.,))
        for key in first:np.testing.assert_array_equal(first[key],second[key])
        changed=dict(data,case_past=data['case_past'][:,:,::-1].copy())
        third,_=forecasts(model,changed,stats,24,penalties=(10.,))
        self.assertGreater(np.max(abs(first['real_10.0']-third['real_10.0'])),1e-6)
        np.testing.assert_array_equal(first['random_10.0'],third['random_10.0'])

    def test_query_overlap_is_rejected(self):
        stats=dict(memory_std=[1.],memory_mean=[0.]);model=train_global(self.data(40),stats,24)
        d=self.data(2);d['selected_starts'][0,0]=d['start'][0]-100
        with self.assertRaises(ValueError):forecasts(model,d,stats,24,penalties=(10.,))

    def test_zero_prior_supplement_does_not_use_train_future_labels(self):
        from v7.benchmark_local_pair_zero_prior import fit_zero_prior
        stats=dict(memory_std=[1.],memory_mean=[0.]);train=self.data(40)
        first=fit_zero_prior(train,stats,24)
        changed=fit_zero_prior(dict(train,raw_y=np.full((40,96),1e15)),stats,24)
        for key in first:np.testing.assert_array_equal(first[key],changed[key])
        np.testing.assert_array_equal(first['prior'],0.)


if __name__=='__main__':unittest.main()
