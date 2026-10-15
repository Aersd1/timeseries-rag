import unittest
import numpy as np
import torch
from v7.selective_rag import (ARMS,SETTINGS,UtilityNet,chronological_split,options,
    selector_features,utility_targets,route,error_metrics,diagnostic,calibrate_router)
from v7.benchmark_unified_derangement import derange


class SelectiveRagTests(unittest.TestCase):
    def data(self,n=12):
        rng=np.random.default_rng(42);x=rng.normal(size=(n,96)).astype(np.float32)
        d=dict(raw_x=x,raw_y=rng.normal(size=(n,96)).astype(np.float32),raw_quantiles=np.zeros((n,9,96),dtype=np.float32),
            probe_quantiles=np.zeros((n,9,32),dtype=np.float32),sid=np.zeros(n,dtype=int),start=5000+np.arange(n)*192,std=np.ones(n),
            case_past=rng.normal(size=(n,10,96)).astype(np.float32),case_future=rng.normal(size=(n,10,96)).astype(np.float32),
            case_baseline=np.zeros((n,10,96),dtype=np.float32),selected_starts=np.tile(np.arange(10)*192,(n,1)))
        d['raw_quantiles'][:,8]=2.;d['raw_quantiles'][:,0]=-2.
        for k in ['case_past','case_future','case_baseline','selected_starts']:d['random_'+k]=d[k].copy()
        d=derange(d);s=dict(std=[1.],mean=[0.],memory_std=[1.],memory_mean=[0.],case_count=10)
        return d,s

    def test_rejection_returns_exact_foundation_not_query_network(self):
        base=np.full((2,96),3.25,dtype=np.float32);c=np.full((2,12,1,8),100.,dtype=np.float32);scores=np.ones((2,12,1))
        for kwargs in [dict(enabled=False),dict(threshold=2.),dict(strength=0.)]:
            args=dict(threshold=0.,strength=1.,enabled=True);args.update(kwargs)
            pred,mask,_=route(base,c,scores,**args);np.testing.assert_array_equal(pred,base);self.assertFalse(mask.any())

    def test_patches_have_independent_decisions_and_case_choice(self):
        base=np.zeros((2,96),dtype=np.float32);c=np.ones((2,12,10,8),dtype=np.float32);c[:,:,7]=2.
        score=np.full((2,12,10),-1.);score[:,3,7]=3.
        p,m,j=route(base,c,score,0.,.25)
        np.testing.assert_array_equal(p.reshape(2,12,8)[:,3],.5);self.assertEqual(m.sum(),2);self.assertTrue(np.all(j[:,3]==7))

    def test_chronological_split_purges_complete_episodes(self):
        d,_=self.data();fit,cal,conf=chronological_split(d,5000,10000)
        self.assertLessEqual(d['start'][fit].max()+192,d['start'][cal].min());self.assertLessEqual(d['start'][cal].max()+192,d['start'][conf].min())
        d['start']=5000+np.arange(12)*96;fit,cal,conf=chronological_split(d,5000,10000)
        self.assertEqual(len(fit),3);self.assertEqual(len(cal),3)

    def test_options_never_read_current_future(self):
        d,s=self.data();c=options(d,s,24,'case_utility');d['raw_y'][:]=np.nan
        np.testing.assert_array_equal(c,options(d,s,24,'case_utility'))

    def test_current_and_not_yet_observed_outcomes_are_not_features(self):
        d,s=self.data();c=options(d,s,24,'case_utility');a=selector_features(d,c,s,24,'case_utility',d,c)
        # Current row 5 and all later labels cannot affect row 5 inputs.
        d['raw_y'][5:]=1e12;b=selector_features(d,c,s,24,'case_utility',d,c)
        np.testing.assert_array_equal(a[5],b[5]);self.assertFalse(np.array_equal(a[-1],b[-1]))

    def test_test_targets_are_never_features_with_frozen_historical_bank(self):
        history,s=self.data();d,_=self.data();d['start']+=5000
        hc=options(history,s,24,'case_utility');c=options(d,s,24,'case_utility')
        a=selector_features(d,c,s,24,'case_utility',history,hc);d['raw_y'][:]=np.nan
        np.testing.assert_array_equal(a,selector_features(d,c,s,24,'case_utility',history,hc))

    def test_controls_ignore_case_true_future(self):
        d,s=self.data();history,_=self.data();frozen=np.zeros_like(d['raw_x'])
        for arm in ['query_only','history_only']:
            c=options(d,s,24,arm,frozen);hc=options(history,s,24,arm,frozen)
            a=selector_features(d,c,s,24,arm,history,hc);changed=dict(d,case_future=np.full_like(d['case_future'],np.nan))
            np.testing.assert_array_equal(a,selector_features(changed,c,s,24,arm,history,hc))

    def test_every_arm_uses_identical_selector_parameter_count(self):
        d,s=self.data();dims=[]
        for arm in ARMS:
            c=options(d,s,24,arm,np.zeros_like(d['raw_x']));x=selector_features(d,c,s,24,arm,d,c);dims.append(x.shape[-1])
        self.assertEqual(len(set(dims)),1)
        self.assertGreater(sum(p.numel() for p in UtilityNet(dims[0]).parameters()),0)

    def test_gain_label_is_actual_loss_difference_not_similarity(self):
        d,s=self.data();d['raw_y'][:]=2.;d['raw_quantiles'][:,4]=0.;c=np.ones((12,12,1,8),dtype=np.float32)
        gain,harm=utility_targets(d,c,s);np.testing.assert_array_equal(gain,3.);np.testing.assert_array_equal(harm,0.)

    def test_confirmation_can_reject_and_other_labels_do_not_choose_policy(self):
        d,s=self.data();d['raw_y'][:]=1.;c=np.ones((12,12,1,8),dtype=np.float32);scores=np.ones((12,12,1))
        cal=np.arange(4,8);conf=np.arange(8,12);p=calibrate_router(d,c,scores,s,cal,conf);self.assertTrue(p['enabled'])
        d['raw_y'][:4]=100.;self.assertEqual(p,calibrate_router(d,c,scores,s,cal,conf))
        d['raw_y'][conf]=-1.;self.assertFalse(calibrate_router(d,c,scores,s,cal,conf)['enabled'])

    def test_oracle_is_only_diagnostic_and_bounds_offered_strength_family(self):
        d,s=self.data();c=np.full((12,12,1,8),10.,dtype=np.float32);scores=np.ones((12,12,1))
        pred,mask,choice=route(d['raw_quantiles'][:,4],c,scores,0.,.1)
        out=diagnostic(d,c,scores,pred,mask,choice,s,strength=.1)
        self.assertLessEqual(out['patch_oracle_mse'],out['actual_mse']+1e-6)
        self.assertAlmostEqual(out['base_mse']-out['actual_mse'],out['adopted_gain_contribution']-out['adopted_harm_contribution'],places=6)


if __name__=='__main__':unittest.main()
