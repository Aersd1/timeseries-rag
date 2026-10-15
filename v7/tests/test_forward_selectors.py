import unittest
import numpy as np
import torch
from v7.tests import test_unified_residual as fixtures
from v7.forward_selectors import inputs,SummaryUtility,SequenceUtility,labels,choose_blocks,calibration,apply_policy,portfolio,fit_models,prediction


class ForwardSelectorTests(unittest.TestCase):
    def data(self):
        d,s=fixtures.UnifiedResidualTests().data();d={k:np.tile(v,(10,)+(1,)*(v.ndim-1)) for k,v in d.items()}
        d['start']=5000+np.arange(40)*192;d['std']=np.ones(40);return d,s

    def test_raw_sequence_and_summary_have_expected_outputs(self):
        d,s=self.data();p=d['raw_quantiles'][:,4];x=inputs(d,p,s,24.,'real',d,p,raw=True)
        self.assertEqual(x['query'].shape,(40,12,88));self.assertEqual(x['past'].shape,(40,10,96))
        for model in [SummaryUtility(x['summary'].shape[-1]),SequenceUtility(x['summary'].shape[-1])]:
            model.eval()
            with torch.no_grad():out=model(**{k:torch.as_tensor(v) for k,v in x.items()})
            self.assertEqual(out.shape,(40,12,2));self.assertTrue(torch.all(out[:,:,1]>=0))

    def test_current_test_targets_do_not_enter_either_input(self):
        d,s=self.data();h={k:v.copy() for k,v in d.items()};p=d['raw_quantiles'][:,4];a=inputs(d,p,s,24.,'real',h,p,raw=True)
        d['raw_y'][:]=np.nan;b=inputs(d,p,s,24.,'real',h,p,raw=True)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_query_and_history_controls_do_not_read_case_futures(self):
        d,s=self.data();h={k:v.copy() for k,v in d.items()};p=d['raw_quantiles'][:,4]
        for arm in ['query_only','history_only']:
            a=inputs(d,p,s,24.,arm,h,p,raw=True);changed=dict(d,case_future=np.full_like(d['case_future'],np.nan))
            b=inputs(changed,p,s,24.,arm,h,p,raw=True)
            for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_label_changes_beyond_fit_cohort_do_not_change_fitted_models(self):
        d,s=self.data();p=d['raw_quantiles'][:,4];x=inputs(d,p,s,24.,'query_only',d,p)
        g,h=labels(d,p+1,s);ids=np.arange(4)
        a,na,_=fit_models('summary_mlp',x,g,h,d,s,ids)
        g[4:]=1e12;h[4:]=1e12;b,nb,_=fit_models('summary_mlp',x,g,h,d,s,ids)
        self.assertEqual(na,nb)
        for aa,bb in zip(a,b):
            for k in aa.state_dict():np.testing.assert_array_equal(aa.state_dict()[k],bb.state_dict()[k])

    def test_confirmation_does_not_select_a_different_hyperparameter(self):
        d,s=self.data();d['raw_quantiles'][:,4]=0.;d['raw_y'][:]=1.;p=np.ones((40,96),np.float32)
        pred=np.zeros((3,40,12,2),np.float32);pred[:,:,:,0]=1.
        blocks=choose_blocks(d,20000);a=calibration(d,p,pred,'summary_mlp',s,blocks);self.assertTrue(a['enabled'])
        d['raw_y'][np.r_[blocks[2],blocks[3]]]=-1.;b=calibration(d,p,pred,'summary_mlp',s,blocks);self.assertFalse(b['enabled'])
        for k in ['risk','uncertainty','threshold','strength','calibration_objective']:self.assertEqual(a[k],b[k])
        out,mask=apply_policy(d,p,pred,b);np.testing.assert_array_equal(out,d['raw_quantiles'][:,4]);self.assertFalse(mask.any())

    def test_portfolio_does_not_switch_to_confirmation_winner(self):
        a=dict(calibration_objective=1.,enabled=False,family='hist_gbdt');b=dict(calibration_objective=2.,enabled=True,family='raw_sequence')
        selected=portfolio(dict(hist_gbdt=a,raw_sequence=b));self.assertEqual(selected['selected_family'],'hist_gbdt');self.assertFalse(selected['enabled'])

    def test_validation_blocks_purge_whole192_point_labels(self):
        d,_=self.data();d['start']=5000+96*np.arange(40);blocks=choose_blocks(d,12000)
        for a,b in zip(blocks[:-1],blocks[1:]):self.assertLessEqual(d['start'][a].max()+192,d['start'][b].min())

    def test_tree_models_fit_the_same_incremental_utility_labels(self):
        d,s=self.data();p=d['raw_quantiles'][:,4];x=inputs(d,p,s,24.,'real',d,p)
        g,h=labels(d,p+1,s);models,normal,_=fit_models('hist_gbdt',x,g,h,d,s,np.arange(20))
        out=prediction('hist_gbdt',models,normal,x)
        self.assertEqual(out.shape,(3,40,12,2));self.assertTrue(np.isfinite(out).all());self.assertTrue(np.all(out[:,:,:,1]>=0))

    def test_serving_store_rejects_any_read_past_observed_history(self):
        from v7.query_forward_selectors import ObservedStore
        from v7.tests.test_dense_forward import TinyStore
        store=ObservedStore(TinyStore(),3396)
        self.assertEqual(len(store.values(0)),3396);self.assertEqual(len(store.window(0,3300,96)),96)
        with self.assertRaises(ValueError):store.window(0,3300,97)
        with self.assertRaises(ValueError):store.window(0,-1,96)

    def test_device_audit_accepts_roundoff_but_rejects_meaningful_score_drift(self):
        from v7.verify_forward_selectors import verify_utilities
        expected=np.zeros((3,40,12,2),np.float32)
        verify_utilities(expected+1e-5,expected,'raw_sequence','cpu')
        with self.assertRaises(ValueError):verify_utilities(expected+.01,expected,'raw_sequence','cpu')
        with self.assertRaises(AssertionError):verify_utilities(expected+.001,expected,'raw_sequence','cuda')

    def test_value_interventions_preserve_query_keys_and_case_metadata(self):
        from v7.unified_residual import features
        d,s=self.data();a,_,_=features(d,s,24.,'real')
        b,_,_=features(dict(d,case_future=d['case_baseline']),s,24.,'real')
        c,_,_=features(d,s,24.,'shuffled')
        np.testing.assert_array_equal(b['future'],np.zeros_like(b['future']))
        for key in a:
            if key!='future':
                np.testing.assert_array_equal(a[key],b[key]);np.testing.assert_array_equal(a[key],c[key])
        for start in [0,5]:np.testing.assert_array_equal(np.sort(a['future'][:,start:start+5],axis=1),np.sort(c['future'][:,start:start+5],axis=1))


if __name__=='__main__':unittest.main()
