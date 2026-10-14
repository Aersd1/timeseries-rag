import unittest
from types import SimpleNamespace
import numpy as np
from v7.dual_five_rag import RawHistoryBank, fusion, choose_dual, apply_dual


class DualFiveTests(unittest.TestCase):
    def bank(self):
        # Constructing a bank only reads history slices, including for selection.
        x = np.arange(1200, dtype=float)
        store = SimpleNamespace(series=[dict(train_end=900, n=1200, runs=[(0, 1200)])],
            window=lambda sid, start, n:x[start:start+n].copy())
        return RawHistoryBank(store, 0, 900), x

    def test_raw_selection_preserves_level_excludes_first_five_and_future(self):
        bank, x = self.bank()
        query = np.arange(96)+350
        selected, distances = bank.select(query, 700, np.arange(348, 353))
        self.assertNotIn(350, selected)
        self.assertEqual(selected[0], 347)  # ties by earlier start
        self.assertTrue(np.all(selected+192<=700))
        np.testing.assert_allclose(distances, (selected-350)**2)
        # A constant offset is not discarded by independent normalization.
        other, _ = bank.select(query+100, 700, [])
        self.assertEqual(other[0], 450)
        x[700:] = 1e12
        np.testing.assert_array_equal(bank.select(query, 700, np.arange(348, 353))[0], selected)

    def test_early_query_cannot_retrieve_overlapping_bank_histories(self):
        bank, _ = self.bank()
        selected, _ = bank.select(np.arange(96)+350, 240, [])
        self.assertTrue(np.all(selected<=48))
        with self.assertRaises(ValueError): bank.select(np.zeros(96), 194, [])

    def test_ten_case_equal_groups_and_disagreement_do_not_amplify_old_correction(self):
        q = np.tile(np.arange(-4, 5)[None, :, None], (1, 1, 96)).astype(float)
        copied = np.ones((1, 10, 96))
        innov = copied.copy(); innov[:, 5:] = -1
        history = np.linspace(-1, 1, 96)[None]
        paths, state = fusion(np.zeros((1,96)), copied, innov, q, history,
            np.ones(1), np.zeros((1,5)), np.ones((1,96)))
        np.testing.assert_allclose(paths['dual_copy'], 1.)
        np.testing.assert_allclose(paths['dual_transport'], 0.)
        np.testing.assert_allclose(paths['dual_trust'], 0.)
        self.assertEqual(state['agreement'][0], 0.)
        innov[:] = 1
        supported, s = fusion(np.zeros((1,96)), copied, innov, q, history,
            np.ones(1), np.ones((1,5))*100, np.ones((1,96))*1e6)
        self.assertLess(s['support'][0], .01)
        self.assertTrue(np.all(abs(supported['dual_trust'])<=2*history.std()))

    def test_confirmation_cannot_reselect_another_family_using_held_truth(self):
        n = 12
        table = dict(sid=np.zeros(n,dtype=int), start=np.arange(n)*192,
            incumbent=np.zeros((n,96)), moirai_direct=np.zeros((n,96)), raw_y=np.ones((n,96)))
        from v7.dual_five_rag import FAMILIES
        table.update({k:np.ones((n,96))*2 for k in FAMILIES})
        first = choose_dual(table)['0']
        self.assertNotEqual(first['family'], 'incumbent')
        table['raw_y'][8:] = -10
        second = choose_dual(table)['0']
        self.assertEqual(second['family'], 'incumbent')
        for key in ['proposed_family', 'proposed_beta']: self.assertEqual(first[key], second[key])
        np.testing.assert_array_equal(apply_dual(table, {'0':second}), table['incumbent'])

    def test_raw_five_are_not_five_neighbouring_copies_of_one_episode(self):
        bank, _ = self.bank()
        bank.spacing = 96
        selected, _ = bank.select(np.arange(96)+300, 900, [])
        self.assertTrue(np.all(np.diff(np.sort(selected))>=96))

    def test_actual_dual_query_ignores_current_and_sibling_future(self):
        from v7.tests.test_future_state_encoder import FakeStore, FakeAdapter, FakePool
        from v7.query_future_state_rag import FutureStatePredictor
        from v7.query_guarded_bank import GuardedBankPredictor
        from v7.query_dual_five import DualFivePredictor
        f = object.__new__(FutureStatePredictor)
        f.store=FakeStore(); f.means=np.zeros(2); f.stds=np.ones(2); f.grouped=False
        for series in f.store.series: series.update(train_end=650,n=1200,runs=[(0,1200)])
        f.store.window=lambda sid,start,n:f.store.x[sid][start:start+n].copy()
        f.synchronous_values=f.store.x; f.adapter=FakeAdapter(); f.pools={0:FakePool()}
        f.encoder=dict(mean=np.zeros(26),scale=np.ones(26),coef=np.random.default_rng(61).normal(scale=.05,size=(27,12)),precision=np.ones(12))
        f.base_states={0:dict(mean=np.zeros(28),scale=np.ones(28),coef=np.zeros(29))}
        f.base_run=dict(period=24); f.meta=dict(window=25); f.plan={'0':[1]}; f.selection=dict(retrieval_weight=1.)
        f.base_policies={'0':dict(beta=0.,proposed_beta=0.)}
        f.policies={kind:{'0':dict(family='future_state_rag' if kind=='routed' else kind,beta=.5)} for kind in ['encoder_only','future_state_rag','routed']}
        old=object.__new__(GuardedBankPredictor); old.original=f; old.branches={'original':f}
        old.policy={'0':dict(branch='original',family='direct_transport',beta=.5)}; old.no_rag_policy=None
        p=object.__new__(DualFivePredictor); p.previous=old; p.banks={}
        p.contract=dict(limits={'0':600},memory_stride=1,raw_case_spacing=0)
        p.policy={'0':dict(family='increment_transport',beta=.25)}
        history=f.store.x[0][700:796].copy(); before=p.predict(history,0,700)
        for values in f.store.x: values[796:]=1e10
        self.assertEqual(before,p.predict(history,0,700))
        self.assertEqual(before['retrieved_cases'],10); self.assertEqual(before['used_cases'],10)
        self.assertEqual(len(set(before['selected_starts'])),10)


if __name__=='__main__': unittest.main()
