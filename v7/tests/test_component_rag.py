import unittest
import numpy as np
from v7.component_rag import block_mean,detail_paths,choose_segments,apply_segments
from v7.future_state_encoder import train_resolution,prototype_state,decode
from v7.query_component_rag import ComponentPredictor
from v7.tests.test_future_state_encoder import FakeStore,FakeAdapter,FakePool


class ComponentTests(unittest.TestCase):
    def test_deployed_component_forecasts_ignore_query_and_sibling_future_poison(self):
        p=object.__new__(ComponentPredictor);p.store=FakeStore();p.means=np.zeros(2);p.stds=np.ones(2)
        p.synchronous_values=p.store.x;p.adapter=FakeAdapter();p.pools={0:FakePool()}
        p.encoder=dict(mean=np.zeros(38),scale=np.ones(38),coef=np.random.default_rng(22).normal(scale=.05,size=(39,24)),
            precision=np.ones(24),history_width=8,block_width=8)
        p.base_states={0:dict(mean=np.zeros(28),scale=np.ones(28),coef=np.zeros(29))}
        p.base_run=dict(period=24);p.meta=dict(window=25);p.plan={'0':[1]};p.selection=dict(retrieval_weight=1.)
        p.base_policies={'0':dict(beta=0.,proposed_beta=0.)}
        p.policies={kind:{'0':dict(family='future_state_rag' if kind=='routed' else kind,beta=.5)} for kind in ['encoder_only','future_state_rag','routed']}
        p.component_policies=dict(detail_global={'0':dict(family='detail_1.0',beta=.5)})
        for name in ['segmented_rag','segmented_no_rag']:
            family='future_state_rag' if name=='segmented_rag' else 'encoder_only'
            p.component_policies[name]={'0':dict(accepted=True,anchor=dict(family='encoder_only',beta=.25),segments=[dict(family=family,beta=.5)]*6)}
        history=p.store.x[0][700:796].copy();a=p.predict(history,0,700)
        for values in p.store.x:values[796:]=1e9
        self.assertEqual(a,p.predict(history,0,700))
        self.assertIn('segmented_rag',a['forecasts']);self.assertTrue(a['effective_cases']>=1)

    def test_training_selects_finer_states_when_coarse_means_hide_predictable_changes(self):
        rng=np.random.default_rng(51);n=64
        amplitudes=rng.normal(size=(n,6,1));blocks=np.stack([amplitudes,-amplitudes],axis=2).reshape(n,12,1)
        x=np.repeat(blocks,8,axis=1);starts=np.arange(n)*256
        state,selection,rows,oof=train_resolution(x,x,starts,np.zeros(1),np.ones(1),24)
        self.assertEqual(int(state['block_width']),8)
        self.assertLess(selection['resolution_losses']['8'],selection['resolution_losses']['16'])
        self.assertEqual(decode(oof[0],1,8).shape,(96,1))
        for fold in selection['folds']:self.assertLessEqual(fold['last_fitted_episode_end'],fold['held_first_start'])
        bad=x[:1].copy();bad[0,10,0]=np.nan
        proto=prototype_state(x[:1],bad,np.zeros(1),np.ones(1),8)
        self.assertFalse(proto['mask'][0,1]);self.assertTrue(proto['mask'][0,0])

    def test_detail_preserves_all_six_means_and_discounts_same_event(self):
        rng=np.random.default_rng(44);encoded=rng.normal(size=96);cases=np.tile(np.sin(np.arange(96)/3),(5,1));w=np.ones(5)/5
        independent,confidence,n=detail_paths(encoded,cases,w,np.arange(5)*192)
        duplicate,dc,dn=detail_paths(encoded,cases,w,np.zeros(5))
        self.assertAlmostEqual(n,5);self.assertAlmostEqual(dn,1)
        self.assertTrue(np.all(confidence>.99));self.assertTrue(np.all(dc==0))
        for path in independent.values():np.testing.assert_allclose(block_mean(path),block_mean(encoded),atol=1e-14)
        for path in duplicate.values():np.testing.assert_allclose(path,encoded)

    def test_disagreement_reduces_detail_confidence(self):
        encoded=np.zeros(96);path=np.sin(np.arange(96)/2);starts=np.arange(5)*192
        _,consistent,_=detail_paths(encoded,np.tile(path,(5,1)),np.ones(5),starts)
        _,conflicted,_=detail_paths(encoded,np.array([path,-path,path,-path,np.zeros(96)]),np.ones(5),starts)
        self.assertTrue(np.all(conflicted<consistent));np.testing.assert_allclose(conflicted,0,atol=1e-12)

    def test_final_validation_block_can_only_reject_not_reselect(self):
        target=np.ones((12,96));experts={'encoder_only':np.ones((12,96)),'future_state_rag':np.ones((12,96))*2}
        sid=np.zeros(12,dtype=int);starts=np.arange(12)*192;legacy={'0':dict(family='direct',beta=0.)};disabled=np.zeros(12,dtype=bool)
        a=choose_segments(target,experts,sid,starts,legacy,disabled)['0'];self.assertTrue(a['accepted'])
        poisoned=target.copy();poisoned[8:]=-10
        b=choose_segments(poisoned,experts,sid,starts,legacy,disabled)['0']
        self.assertFalse(b['accepted']);self.assertEqual(a['segments'],b['segments']);self.assertEqual(a['anchor'],b['anchor'])
        np.testing.assert_array_equal(apply_segments({'0':b},experts,sid),0)

    def test_disabled_base_never_reactivated_by_horizon_search(self):
        target=np.ones((12,96));experts={'encoder_only':target*2};legacy={'0':dict(family='direct',beta=0.)}
        p=choose_segments(target,experts,np.zeros(12),np.arange(12)*192,legacy,np.ones(12,dtype=bool))['0']
        self.assertFalse(p['accepted']);self.assertEqual(p['segments'],[])


if __name__=='__main__':unittest.main()
