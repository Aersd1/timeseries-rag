import unittest
import numpy as np
import torch
from v7.paired_history_increment import (ARMS,PairedHistoryIncrement,pack,with_base,predict,summary,mask_modalities,fit_increment)
from v7.benchmark_paired_history_increment import baseline_policy,utility_labels,carrier


def fixture(n=8):
    rng=np.random.default_rng(17);x=rng.normal(size=(n,96)).astype(np.float32)
    q=np.repeat(x[:,-1,None,None],9*96,axis=2).reshape(n,9,96)
    q=q+np.linspace(-2,2,9,dtype=np.float32)[None,:,None]
    d=dict(raw_x=x,raw_y=rng.normal(size=(n,96)).astype(np.float32),raw_quantiles=q,
        probe_quantiles=np.zeros((n,9,32),np.float32),sid=np.zeros(n,dtype=int),start=5000+np.arange(n)*192,std=np.ones(n,np.float32),
        case_past=rng.normal(size=(n,10,96)).astype(np.float32),case_future=rng.normal(size=(n,10,96)).astype(np.float32),
        case_baseline=rng.normal(size=(n,10,96)).astype(np.float32),selected_starts=np.tile(np.arange(10)*192,(n,1)))
    for k in ['case_past','case_future','case_baseline','selected_starts']:d['random_'+k]=d[k].copy()
    d['shuffle_order']=np.tile(np.r_[np.roll(np.arange(5),1),5+np.roll(np.arange(5),1)],(n,1))
    return d,dict(mean=[0.],std=[1.],memory_std=[1.],case_count=10)


class PairedIncrementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_sequence_is_an_input_and_continuation_is_not_only_a_residual(self):
        d,s=fixture();p,sc=pack(d,s,24)
        self.assertEqual(p['past'].shape,(8,10,96));self.assertEqual(p['continuation'].shape,(8,10,12,24))
        self.assertGreater(float(p['past'].std()),0.)
        self.assertGreater(float(p['continuation'][:,:,:,:8].std()),0.)
        self.assertFalse(np.array_equal(p['continuation'][:,:,:,:8],p['continuation'][:,:,:,16:]))

    def test_current_future_is_not_a_forecast_or_gate_input(self):
        for arm in ARMS+['empty']:
            d,s=fixture();a,sc=pack(d,s,24,arm);a=with_base(a,d['raw_quantiles'][:,4],d,sc)
            before=summary(a,np.zeros((8,96),np.float32));d['raw_y'][:]=np.nan
            b,_=pack(d,s,24,arm);b=with_base(b,d['raw_quantiles'][:,4],d,sc)
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            np.testing.assert_array_equal(before,summary(b,np.zeros((8,96),np.float32)))

    def test_history_only_does_not_read_true_old_future(self):
        d,s=fixture();p,_=pack(d,s,24,'history_only');d['case_future'][:]=np.nan
        other,_=pack(d,s,24,'history_only')
        for k in p:np.testing.assert_array_equal(p[k],other[k])
        self.assertFalse(p['future_present'].any());self.assertTrue(p['past_present'].all())

    def test_empty_ignores_all_case_values_and_returns_nonzero_base_exactly(self):
        d,s=fixture()
        for k in ['case_past','case_future','case_baseline']:d[k][:]=np.nan
        d['selected_starts'][:]=10**8
        p,sc=pack(d,s,24,'empty');base=d['raw_quantiles'][:,4]+3.25;p=with_base(p,base,d,sc,1)
        model=PairedHistoryIncrement()
        with torch.no_grad():model.output[-1].weight.normal_(0,.2)
        np.testing.assert_array_equal(predict(model,p,base,sc,'cpu'),base)

    def test_self_and_overlap_rejected(self):
        d,s=fixture();d['selected_starts'][0,0]=d['start'][0]-191
        with self.assertRaisesRegex(ValueError,'overlap'):pack(d,s,24)

    def test_history_has_a_value_path_even_when_key_weights_are_fixed(self):
        d,s=fixture();p,sc=pack(d,s,24);base=d['raw_quantiles'][:,4];p=with_base(p,base,d,sc)
        torch.manual_seed(18);m=PairedHistoryIncrement()
        with torch.no_grad():
            m.output[-1].weight.normal_(0,.1);m.key.weight.zero_();m.key.bias.zero_()
        a=predict(m,p,base,sc,'cpu');changed=dict(p,past=p['past']+1.)
        b=predict(m,changed,base,sc,'cpu')
        self.assertGreater(float(abs(a-b).max()),1e-5)

    def test_derangement_preserves_history_and_both_futures_and_errors_as_pairs(self):
        d,s=fixture();a,_=pack(d,s,24);b,_=pack(d,s,24,'deranged')
        np.testing.assert_array_equal(a['past'],b['past']);np.testing.assert_array_equal(a['metadata'],b['metadata'])
        np.testing.assert_array_equal(b['continuation'],a['continuation'][np.arange(8)[:,None],d['shuffle_order']])

    def test_explicit_history_ablation_has_identical_future_representation(self):
        d,s=fixture();a,_=pack(d,s,24);b,_=pack(d,s,24,'without_history_sequence')
        np.testing.assert_array_equal(a['continuation'],b['continuation'])
        self.assertFalse(b['past'].any());self.assertFalse(b['metadata'].any())

    def test_gain_label_is_increment_over_supplied_base(self):
        d,s=fixture();d['raw_y'][:]=2.;b=np.ones((8,96),np.float32);c=np.full_like(b,1.5)
        g,h=utility_labels(d,b,c,s);np.testing.assert_array_equal(g,.75);np.testing.assert_array_equal(h,0.)
        np.testing.assert_array_equal(carrier(d,b)['raw_quantiles'][:,4],b)

    def test_confirmation_cannot_change_chosen_base_mode(self):
        d,s=fixture(16);d['raw_y'][:]=1.;d['raw_quantiles'][:,4]=0.;a=np.ones((16,96),np.float32)
        blocks=[np.arange(i,i+4) for i in [0,4,8,12]];p=baseline_policy(d,a,s,blocks)
        self.assertEqual(p['mode'],1);self.assertTrue(p['enabled'])
        d['raw_y'][8:]=-1.;p=baseline_policy(d,a,s,blocks)
        self.assertEqual(p['mode'],1);self.assertFalse(p['enabled'])

    def test_small_real_training_has_finite_losses_and_bounded_correction(self):
        d,s=fixture();b=d['raw_quantiles'][:,4];m,loss=fit_increment(d,[b,b+.2],s,24,'paired','cpu')
        self.assertEqual(len(loss),16);self.assertTrue(np.isfinite(loss).all())
        p,sc=pack(d,s,24);p=with_base(p,b,d,sc);out=predict(m,p,b,sc,'cpu')
        self.assertTrue(np.all(abs(out-b)<=p['radius']*sc[:,None]+1e-6))


if __name__=='__main__':unittest.main()
