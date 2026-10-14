import unittest
import numpy as np
import torch
from v7.aligned_case_fusion import align_cases,inputs,CaseFusion,tensor_inputs
from v7.benchmark_case_fusion import calibrate,apply


class CaseFusionTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(3);n=4
        x=rng.normal(size=(n,96)).astype(np.float32)
        q=np.tile(np.arange(-4,5)[None,:,None],(n,1,96)).astype(np.float32)
        return dict(raw_x=x,raw_quantiles=q,case_past=rng.normal(size=(n,10,96)),case_future=rng.normal(size=(n,10,96)),
            selected_starts=np.tile(np.arange(10)*192,(n,1)),start=np.arange(n)*192+2500,sid=np.zeros(n,dtype=int),raw_y=rng.normal(size=(n,96)))

    def test_alignment_does_not_amplify_a_flat_old_case(self):
        x=np.tile(np.linspace(10,20,96),(2,1));p=np.ones((2,10,96))*100;y=np.ones_like(p)*1e6
        a,b=align_cases(x,p,y,'wiener')
        np.testing.assert_allclose(a,15.);np.testing.assert_allclose(b,15.)
        p[:]=x[:,None];y[:]=p+2
        a,b=align_cases(x,p,y,'wiener')
        ratio=25/26
        np.testing.assert_allclose(b-a,2*ratio,rtol=1e-5)

    def test_no_query_future_can_change_inputs_and_query_control_ignores_cases(self):
        d=self.data();stats=dict(mean=[0.],std=[1.],memory_std=[1.])
        a,_,_=inputs(d,stats,24,'wiener');d['raw_y'][:]=1e12;b,_,_=inputs(d,stats,24,'wiener')
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        a,_,_=inputs(d,stats,24,'without_retrieval');d['case_past'][:]=np.nan;d['case_future'][:]=1e30;d['selected_starts'][:]=10**9
        b,_,_=inputs(d,stats,24,'without_retrieval')
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_rag_rejects_overlap_and_uninitialized_adapter_preserves_moirai(self):
        d=self.data();stats=dict(mean=[0.],std=[1.]);features,_,_=inputs(d,stats,24,'wiener')
        m=CaseFusion(1);out=m(**tensor_inputs(features,'cpu'))
        np.testing.assert_array_equal(out.detach().numpy(),0.)
        d['selected_starts'][0,0]=d['start'][0]-191
        with self.assertRaises(ValueError):inputs(d,stats,24,'wiener')

    def test_attention_can_learn_to_use_distinct_old_futures_with_bounded_outputs(self):
        d=self.data();stats=dict(mean=[0.],std=[1.]);features,_,_=inputs(d,stats,24,'wiener')
        torch.manual_seed(2);m=CaseFusion(1)
        with torch.no_grad():m.output[-1].weight.normal_(std=.1)
        a=m(**tensor_inputs(features,'cpu')).detach().numpy()
        d['case_future']*=3;other,_,_=inputs(d,stats,24,'wiener')
        b=m(**tensor_inputs(other,'cpu')).detach().numpy()
        self.assertGreater(float(np.max(abs(a-b))),1e-6)
        self.assertTrue(np.all(abs(b)<=other['radius']+1e-6))

    def test_last_validation_block_can_only_reject_the_previously_selected_beta(self):
        n=12;val=dict(sid=np.zeros(n,dtype=int),start=np.arange(n)*192,raw_y=np.ones((n,96)),raw_quantiles=np.zeros((n,9,96)))
        pred=np.ones((n,96))*2;a=calibrate(val,pred)['0']
        self.assertGreater(a['beta'],0.)
        val['raw_y'][8:]=-3;b=calibrate(val,pred)['0']
        self.assertEqual(b['beta'],0.);self.assertEqual(a['proposed_beta'],b['proposed_beta'])
        np.testing.assert_array_equal(apply({'0':b},val,pred),0.)

    def test_five_case_network_and_control_use_exactly_five(self):
        d=self.data();stats=dict(mean=[0.],std=[1.],case_count=5)
        for key in ['case_past','case_future','selected_starts']:d[key]=d[key][:,:5].copy()
        for mode in ['wiener','without_retrieval']:
            features,_,_=inputs(d,stats,24,mode)
            self.assertEqual(features['past'].shape,(4,5,96))
            output,attention=CaseFusion(1)(**tensor_inputs(features,'cpu'),return_attention=True)
            self.assertEqual(tuple(attention.shape),(4,12,61));self.assertEqual(tuple(output.shape),(4,96))

    def test_increment_confirmation_preserves_incumbent_instead_of_resetting_to_moirai(self):
        n=12;val=dict(sid=np.zeros(n,dtype=int),start=np.arange(n)*192,raw_y=np.ones((n,96)),raw_quantiles=np.zeros((n,9,96)))
        incumbent=np.ones((n,96))*.2;proposal=np.ones((n,96))
        a=calibrate(val,proposal,incumbent)['0'];self.assertGreater(a['beta'],0.)
        val['raw_y'][8:]=-5;b=calibrate(val,proposal,incumbent)['0']
        self.assertEqual(a['proposed_beta'],b['proposed_beta']);self.assertEqual(b['beta'],0.)
        np.testing.assert_array_equal(apply({'0':b},val,proposal,incumbent),incumbent)

    def test_training_selector_never_reads_its_own_or_sibling_future(self):
        from types import SimpleNamespace
        from v7.tests.test_future_state_encoder import FakeStore
        from v7.query_future_state_rag import FutureStatePredictor
        from v7.prepare_case_fusion import select_training_cases
        p=object.__new__(FutureStatePredictor);p.store=FakeStore();p.grouped=False
        p.synchronous_values=p.store.x;p.means=np.zeros(2);p.stds=np.ones(2)
        p.base_run=dict(period=24);p.selection=dict(retrieval_weight=1.)
        state=dict(mean=np.zeros(26),scale=np.ones(26),coef=np.random.default_rng(22).normal(scale=.05,size=(27,12)),precision=np.ones(12))
        d=dict(raw_x=p.store.x[0][700:796][None].copy(),raw_quantiles=np.tile(np.arange(-4,5)[None,:,None],(1,1,96)),
            sid=np.array([0]),start=np.array([700]),candidate_starts=(np.arange(100)*4)[None],
            history_scores=np.ones((1,100)),future_scores=np.ones((1,100)),base=np.zeros((1,96)),raw_y=np.zeros((1,96)))
        owner=SimpleNamespace(original=p)
        before=select_training_cases(owner,d,0,p,state,[0,1]);d['raw_y'][:]=1e12
        for values in p.store.x:values[796:]=1e12
        np.testing.assert_array_equal(before,select_training_cases(owner,d,0,p,state,[0,1]))
        d['candidate_starts'][0,-1]=700
        with self.assertRaises(ValueError):select_training_cases(owner,d,0,p,state,[0,1])

    def test_paired_attention_preserves_horizon_and_history_only_scores(self):
        d=self.data();stats=dict(mean=[0.],std=[1.],case_count=5,residual_reference=True,metadata_width=13)
        for key in ['case_past','case_future','selected_starts']:d[key]=d[key][:,:5].copy()
        d['reference']=np.ones((4,96))*.2;d['probe_quantiles']=np.zeros((4,9,32))
        f,_,_=inputs(d,stats,24,'wiener');self.assertEqual(f['metadata'].shape,(4,5,13))
        torch.manual_seed(9);m=CaseFusion(1,conditioned=True,paired=True,metadata_width=13)
        with torch.no_grad():m.output[-1].weight.normal_(std=.1)
        t=tensor_inputs(f,'cpu');a,attention=m(**t,return_attention=True)
        self.assertEqual(tuple(attention.shape),(4,12,6))
        _,scores=m(**t,return_scores=True)
        changed={k:v.clone() for k,v in t.items()};changed['future'][:,:,-8:]+=2
        b,new_scores=m(**changed,return_scores=True)
        torch.testing.assert_close(a[:,:88],b[:,:88],rtol=0,atol=0)
        self.assertGreater(float((a[:,-8:]-b[:,-8:]).abs().max()),1e-6)
        torch.testing.assert_close(scores,new_scores,rtol=0,atol=0)
        changed['quantiles']+=3;changed['reference']+=3;changed['probe_error']+=3
        _,new_scores=m(**changed,return_scores=True)
        torch.testing.assert_close(scores,new_scores,rtol=0,atol=0)

    def test_residual_reference_is_preserved_and_probe_uses_observed_history(self):
        d=self.data();stats=dict(mean=[0.],std=[1.],case_count=5,residual_reference=True)
        for key in ['case_past','case_future','selected_starts']:d[key]=d[key][:,:5].copy()
        d['reference']=np.ones((4,96))*.3;d['probe_quantiles']=np.zeros((4,9,32))
        f,_,scale=inputs(d,stats,24,'without_retrieval')
        np.testing.assert_allclose(f['probe_error'],d['raw_x'][:,64:]/scale[:,None],rtol=1e-5)
        from v7.aligned_case_fusion import forecast
        out=forecast(CaseFusion(1,conditioned=True),tensor_inputs(f,'cpu'),d,scale)
        np.testing.assert_allclose(out,d['reference'])
        old={k:v.copy() for k,v in f.items()};d['raw_y'][:]=1e15
        f,_,_=inputs(d,stats,24,'without_retrieval')
        for key in old:np.testing.assert_array_equal(f[key],old[key])


if __name__=='__main__':unittest.main()
