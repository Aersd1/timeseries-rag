import unittest
import numpy as np
import torch
from v7.tests import test_case_fusion as fixtures
from v7.unified_residual import MODES,features,UnifiedResidualFusion,matched_starts,aggregate_guard
from v7.aligned_case_fusion import tensor_inputs,forecast


class UnifiedResidualTests(unittest.TestCase):
    def data(self):
        d=fixtures.CaseFusionTests().data();d['case_baseline']=np.zeros_like(d['case_past'])
        d['probe_quantiles']=np.zeros((4,9,32),dtype=np.float32)
        for key in ['case_past','case_future','case_baseline','selected_starts']:d['random_'+key]=d[key].copy()
        d['shuffle_order']=np.tile(np.r_[np.arange(5)[::-1],np.arange(5,10)[::-1]],(4,1))
        return d,dict(mean=[0.],std=[1.],memory_std=[1.],case_count=10)

    def test_targets_are_never_inputs_and_controls_do_not_read_old_futures(self):
        d,s=self.data()
        for mode in MODES:
            before,_,_=features(d,s,24,mode);d['raw_y'][:]=1e12
            after,_,_=features(d,s,24,mode)
            for k in before:np.testing.assert_array_equal(before[k],after[k])
        for mode in ['query_only','history_only']:
            d,s=self.data();before,_,_=features(d,s,24,mode);d['case_future'][:]=np.nan
            if mode=='query_only':d['case_past'][:]=np.nan;d['case_baseline'][:]=np.nan;d['selected_starts'][:]=10**9
            after,_,_=features(d,s,24,mode)
            for k in before:np.testing.assert_array_equal(before[k],after[k])

    def test_uniform_reference_gives_zero_residual_and_same_capacity(self):
        d,s=self.data();d['case_future']=d['case_baseline'].copy()
        f,_,_=features(d,s,24,'real');np.testing.assert_array_equal(f['future'],0.)
        counts=[]
        for mode in MODES:
            f,_,scale=features(d,s,24,mode);m=UnifiedResidualFusion(1);counts.append(sum(p.numel() for p in m.parameters()))
            out,attention=m(**tensor_inputs(f,'cpu'),return_attention=True)
            self.assertEqual(tuple(attention.shape),(4,12,11));np.testing.assert_array_equal(out.detach(),0.)
            np.testing.assert_array_equal(forecast(m,tensor_inputs(f,'cpu'),d,scale),d['raw_quantiles'][:,4])
        self.assertEqual(len(set(counts)),1)

    def test_shuffle_only_changes_residual_pairing_and_is_within_route(self):
        d,s=self.data();a,_,_=features(d,s,24,'real');b,_,_=features(d,s,24,'shuffled')
        for key in a:
            if key!='future':np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(b['future'],a['future'][np.arange(4)[:,None],d['shuffle_order']])

    def test_state_matched_random_excludes_real_self_and_overlap(self):
        rng=np.random.default_rng(8);starts=np.arange(0,6000,16,dtype=np.int64);past=rng.normal(size=(len(starts),96))
        selected=np.r_[starts[:5],starts[[50,70,90,110,130]]]
        a=matched_starts(starts,past,selected,5000,24,0.,1.,29)
        np.testing.assert_array_equal(a,matched_starts(starts,past,selected,5000,24,0.,1.,29))
        self.assertEqual(len(np.unique(a)),10);self.assertFalse(np.isin(a,selected).any())
        self.assertTrue(np.all(a+192<=5000));self.assertTrue(np.all(np.diff(np.sort(a[5:]))>=192))

    def test_guard_rejects_harmful_changes_without_consulting_test(self):
        data=dict(sid=np.zeros(12,dtype=int),start=np.arange(12)*192,raw_y=np.zeros((12,96)))
        base=np.zeros((12,96));self.assertTrue(aggregate_guard(data,base,base,[1.])['accepted'])
        self.assertFalse(aggregate_guard(data,base+1,base,[1.])['accepted'])

    def test_model_uses_residuals_but_remains_bounded(self):
        d,s=self.data();f,_,_=features(d,s,24,'real');torch.manual_seed(4);m=UnifiedResidualFusion(1)
        with torch.no_grad():m.residual_gate.bias.fill_(.5)
        a=m(**tensor_inputs(f,'cpu')).detach().numpy();d['case_future']*=2
        g,_,_=features(d,s,24,'real');b=m(**tensor_inputs(g,'cpu')).detach().numpy()
        self.assertGreater(float(np.max(abs(a-b))),1e-6);self.assertTrue(np.all(abs(b)<=g['radius']+1e-6))

    def test_residual_value_subtracts_the_same_forecaster(self):
        d,s=self.data();a,_,_=features(d,s,24,'real')
        d['case_future']+=3;d['case_baseline']+=3;b,_,_=features(d,s,24,'real')
        np.testing.assert_allclose(a['future'],b['future'],atol=1e-6)

    def test_rejected_increment_keeps_query_network_instead_of_moirai(self):
        from v7.benchmark_unified_residual import confirmed,apply_confirmed
        d=dict(sid=np.zeros(12,dtype=int),start=np.arange(12)*192,raw_y=-np.ones((12,96)),raw_quantiles=np.zeros((12,9,96)))
        base=np.ones((12,96))*.2;proposal=np.ones_like(base)
        out,policy=confirmed(d,proposal,dict(std=[1.]),base)
        np.testing.assert_array_equal(out,base);np.testing.assert_array_equal(apply_confirmed(d,proposal,base,policy),base)

    def test_strong_shuffle_preserves_sets_and_breaks_every_pair(self):
        from v7.benchmark_unified_derangement import derange
        d,_=self.data();a=derange(d);d['raw_y'][:]=1e15;b=derange(d)
        np.testing.assert_array_equal(a['shuffle_order'],b['shuffle_order'])
        self.assertTrue(np.all(a['shuffle_order']!=np.arange(10)[None]))
        for j in [0,5]:np.testing.assert_array_equal(np.sort(a['shuffle_order'][:,j:j+5],1),np.tile(np.arange(j,j+5),(4,1)))

    def test_multivariate_policy_keeps_exact_variable_results(self):
        from v7.benchmark_case_fusion import calibrate
        sid=np.tile(np.arange(3),12);data=dict(sid=sid,start=np.repeat(np.arange(12)*192,3),raw_y=np.ones((36,96),dtype=np.float32)*.5)
        base=np.zeros((36,96),dtype=np.float32);pred=np.ones_like(base)
        combined=calibrate(data,pred,base)
        for k in range(3):
            rows=np.flatnonzero(sid==k);local={key:value[rows] for key,value in data.items()}
            self.assertEqual(combined[str(k)],calibrate(local,pred[rows],base[rows])[str(k)])
            self.assertEqual(combined[str(k)]['beta'],.5)

    def test_resume_rejects_a_different_case_source(self):
        import tempfile,json
        from pathlib import Path
        from v7.prepare_unified_residual import initialize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'saved';out.mkdir();(out/'preparation.json').write_text(json.dumps(dict(source=str(root/'first'))),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Resume source'):initialize(root/'different',out,'cpu')


if __name__=='__main__':unittest.main()
