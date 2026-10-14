import unittest
from types import SimpleNamespace
import numpy as np
from v7.analogue_transport import transport_paths,direct_experts
from v7.query_guarded_bank import GuardedBankPredictor
from v7.query_future_state_rag import FutureStatePredictor
from v7.tests.test_future_state_encoder import FakeStore,FakeAdapter,FakePool
from v7.conditional_recall import recall
from v7.guarded_bank_rag import choose_bank,apply_bank


class TransportTests(unittest.TestCase):
    def test_rejected_old_retrieval_does_not_lock_a_new_branch_on_a_confirmed_base(self):
        n=12;original=dict(sid=np.zeros(n,dtype=int),start=np.arange(n)*192,std=np.ones(n),base=np.zeros((n,96)),
            final_base=np.zeros((n,96)),raw_y=np.ones((n,96)),routed=np.zeros((n,96)),disabled=np.zeros(n,dtype=bool),selected_starts=np.zeros((n,5),dtype=int))
        values={f'expert_{k}':np.ones((n,96))*2 for k in ['future_state_rag','mix_0.5','transport','transport_consensus']};values['selected_starts']=np.ones((n,5),dtype=int)
        old={'0':dict(family='direct',beta=0.,proposed_beta=.5)}
        policy=choose_bank({'new_bank':values},original,old,old)
        self.assertEqual(policy['0']['branch'],'new_bank')
        original['disabled'][:]=True
        rejected=choose_bank({'new_bank':values},original,old,old)
        self.assertEqual(rejected['0']['branch'],'incumbent')

    def test_direct_paths_are_independent_of_rejected_adapter_and_no_rag_ignores_cases(self):
        q=np.tile(np.arange(-4,5)[:,None],(1,96)).astype(float)
        c=dict(encoded_path=np.ones(96),cases=np.ones((5,96))*2,old_encoded_paths=np.zeros((5,96)),quantiles=q,std=1.,base=np.ones(96)*100,final_base=np.ones(96)*200)
        a=direct_experts(c);c.update(base=np.ones(96)*1e8,final_base=np.ones(96)*1e9)
        b=direct_experts(c)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        c['cases']*=1e8;c['old_encoded_paths']-=1e8
        np.testing.assert_array_equal(a['direct_encoder'],direct_experts(c)['direct_encoder'])

    def test_direct_candidate_is_confirmed_as_a_whole_without_reusing_rejected_adapter(self):
        n=12;original=dict(sid=np.zeros(n,dtype=int),start=np.arange(n)*192,std=np.ones(n),base=np.ones((n,96))*100,
            final_base=np.zeros((n,96)),moirai_direct=np.zeros((n,96)),raw_y=np.ones((n,96)),routed=np.zeros((n,96)),
            disabled=np.ones(n,dtype=bool),selected_starts=np.zeros((n,5),dtype=int))
        values={f'expert_{k}':np.ones((n,96))*2 for k in ['direct_copy','direct_transport','direct_consensus']};values['selected_starts']=np.ones((n,5),dtype=int)
        old={'0':dict(family='direct',beta=0.,proposed_beta=.5)}
        policy=choose_bank({'independent':values},original,old,old,direct=True)
        self.assertEqual(policy['0']['branch'],'independent')
        np.testing.assert_allclose(apply_bank({'independent':values},original,policy)[0],1.)

    def test_actual_guarded_direct_inference_ignores_current_and_sibling_future(self):
        f=object.__new__(FutureStatePredictor);f.store=FakeStore();f.means=np.zeros(2);f.stds=np.ones(2);f.grouped=False
        f.synchronous_values=f.store.x;f.adapter=FakeAdapter();f.pools={0:FakePool()}
        f.encoder=dict(mean=np.zeros(26),scale=np.ones(26),coef=np.random.default_rng(61).normal(scale=.05,size=(27,12)),precision=np.ones(12))
        f.base_states={0:dict(mean=np.zeros(28),scale=np.ones(28),coef=np.zeros(29))};f.base_run=dict(period=24);f.meta=dict(window=25)
        f.plan={'0':[1]};f.selection=dict(retrieval_weight=1.);f.base_policies={'0':dict(beta=0.,proposed_beta=0.)}
        f.policies={kind:{'0':dict(family='future_state_rag' if kind=='routed' else kind,beta=.5)} for kind in ['encoder_only','future_state_rag','routed']}
        p=object.__new__(GuardedBankPredictor);p.original=f;p.branches={'original':f};p.policy={'0':dict(branch='original',family='direct_transport',beta=.5)}
        p.no_rag_policy={'0':dict(branch='original',family='direct_encoder',beta=.5)}
        history=f.store.x[0][700:796].copy();a=p.predict(history,0,700)
        for series in f.store.x:series[796:]=1e9
        self.assertEqual(a,p.predict(history,0,700));self.assertEqual(a['weights'],[.2]*5);self.assertEqual(a['auxiliary_cases'],5)

    def test_transport_removes_old_model_state_and_keeps_query_state(self):
        query=np.linspace(0,3,96);old=np.tile(np.linspace(-2,1,96),(5,1));innovation=np.sin(np.arange(96)/4)
        cases=old+innovation
        np.testing.assert_allclose(transport_paths(query,cases,old),np.tile(query+innovation,(5,1)),atol=1e-14)
        np.testing.assert_allclose(transport_paths(query,old,old),np.tile(query,(5,1)),atol=1e-14)

    def test_future_conditioned_recall_separates_identical_histories_and_excludes_unobserved_cases(self):
        starts=np.arange(300);pool=SimpleNamespace(starts=starts,memory_end=600,past=np.zeros((300,96)),future=np.zeros((300,96)),sid=0)
        codes={'values':np.r_[np.full((150,2),-1.),np.ones((150,2))],'mask':np.ones((300,2),dtype=bool)}
        quantiles=np.tile(np.linspace(-1,1,9)[:,None],(1,96))
        a=recall(pool,np.zeros(96),quantiles,np.ones(2),codes,np.ones(2),600)
        self.assertTrue(np.all(a['candidate_starts']>=150));self.assertEqual(len(a['candidate_starts']),100)
        # Ineligible future prototypes cannot affect recall.
        b=recall(pool,np.zeros(96),quantiles,np.ones(2),codes,np.ones(2),291)
        codes['values'][100:]=1e9;pool.future[100:]=1e9
        c=recall(pool,np.zeros(96),quantiles,np.ones(2),codes,np.ones(2),291)
        for key in b:np.testing.assert_array_equal(b[key],c[key])
        self.assertTrue(np.all(b['candidate_starts']+192<=291))

    def test_hybrid_keeps_fifty_history_hits_with_exactly_one_hundred_unique_candidates(self):
        pool=SimpleNamespace(starts=np.arange(300),memory_end=600,past=np.arange(300)[:,None]*np.ones((1,96)),future=np.zeros((300,96)),sid=0)
        codes={'values':np.r_[np.zeros((150,2)),np.ones((150,2))],'mask':np.ones((300,2),dtype=bool)}
        q=np.tile(np.linspace(-1,1,9)[:,None],(1,96))
        result=recall(pool,np.zeros(96),q,np.ones(2),codes,np.ones(2),600,hybrid=True)
        self.assertTrue(set(range(50)).issubset(result['candidate_starts']));self.assertEqual(len(np.unique(result['candidate_starts'])),100)

    def test_confirmation_only_rejects_the_selected_branch_and_never_falls_to_foundation(self):
        n=12;target=np.ones((n,96));original=dict(sid=np.zeros(n,dtype=int),start=np.arange(n)*192,std=np.ones(n),
            base=np.zeros((n,96)),final_base=np.zeros((n,96)),raw_y=target,routed=np.full((n,96),.2),disabled=np.zeros(n,dtype=bool),selected_starts=np.zeros((n,5),dtype=int))
        values={f'expert_{k}':target*2 for k in ['future_state_rag','mix_0.5','transport','transport_consensus']};values['selected_starts']=np.ones((n,5),dtype=int)
        incumbent={'0':dict(family='encoder_only',beta=.1)}
        a=choose_bank({'new':values},original,incumbent,incumbent)['0'];self.assertEqual(a['branch'],'new')
        poisoned=dict(original);poisoned['raw_y']=target.copy();poisoned['raw_y'][8:]=-5
        b=choose_bank({'new':values},poisoned,incumbent,incumbent)['0'];self.assertEqual(b['branch'],'incumbent')
        for k in ['proposed_branch','proposed_family','proposed_beta']:self.assertEqual(a[k],b[k])
        pred,selected=apply_bank({'new':values},original,{'0':b})
        np.testing.assert_array_equal(pred,original['routed']);np.testing.assert_array_equal(selected,original['selected_starts'])


if __name__=='__main__':unittest.main()
