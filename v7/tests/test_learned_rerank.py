import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from v6.data import Windows
from v7.learned_rerank import LearnedMoiraiReranker, history_scores, blend_scores
from v7.evaluate_learned_rerank import select_weight
from v7.tests.test_variants import fixture, tiny_adapter
from v7.train import fit
from v7.retrieval import build


class LearnedRerankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_soft_history_score_changes_ranking(self):
        x = np.arange(8, dtype=float)
        hs = history_scores(x, np.stack([x*3+7, -x]), .1, [.1, .1])
        self.assertAlmostEqual(hs[0], 0.)
        self.assertGreater(hs[1], 3.9)
        future = np.array([.5, .1])
        self.assertEqual(blend_scores(future, hs, 0.).argmin(), 1)
        self.assertEqual(blend_scores(future, hs, .8).argmin(), 0)
        self.assertTrue(np.isfinite(blend_scores([0., 0.], [0., 0.], .5)).all())
        with self.assertRaises(ValueError): blend_scores(future, hs, 1.1)

    def test_selection_refuses_test_and_zero_history(self):
        rows = [dict(split='validation', query=0, sid=0, start=50, horizon=10, method=m, nmse=v)
                for m, v in [('a50_original_order', 2.), ('blend_0.2', 1.), ('blend_0.8', 1.5)]]
        frame = pd.DataFrame(rows)
        self.assertEqual(select_weight({'fixture':frame}, [.2, .8])['history_weight'], .2)
        with self.assertRaises(ValueError): select_weight({'fixture':frame.assign(split='test')}, [.2, .8])
        with self.assertRaises(ValueError): select_weight({'fixture':frame}, [0.])

    def test_top50_matches_exhaustive_and_respects_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, cp, _ = fixture(root)
            with patch('v7.train.MoiraiAdapter.pretrained', return_value=tiny_adapter()):
                fit(root/'store', cp, root/'train')
            build(root/'store', root/'train/best.pt', root/'index')
            engine = LearnedMoiraiReranker(root/'store', root/'train/best.pt', root/'index')
            dataset = Windows(root/'store', c, 'test')
            item = dataset[0]
            try:
                with patch.object(engine.base.library, 'search', wraps=engine.base.library.search) as search:
                    pool = engine.prepare(item['x'], 0, int(item['start']))
                    self.assertEqual(search.call_count, 1)
                self.assertEqual(len(pool['candidates']), 50)
                vector = engine.base.encode(item['x'], 0)[0]['learned'].astype(float)
                exhaustive = []
                for i, shard in enumerate(engine.base.library.meta['shards']):
                    vectors = engine.base.library.array(i, 'learned/vectors.npy').astype(float)
                    starts = np.array(engine.base.library.array(i, 'learned/starts.npy'), copy=True)
                    exhaustive.extend(zip(np.sum((vectors-vector)**2, axis=1), starts))
                expected = sorted(exhaustive, key=lambda r:(r[0], r[1]))[:50]
                self.assertEqual([h['start'] for h in pool['candidates']], [int(r[1]) for r in expected])
                np.testing.assert_allclose([h['distance'] for h in pool['candidates']], [r[0] for r in expected], rtol=1e-10)
                # Original default search still returns non-overlapping histories.
                diverse, _ = engine.base.library.search(vector, sid=0, k=5, capacity=512)
                self.assertTrue(all(abs(a['start']-b['start']) >= c['length'] for i,a in enumerate(diverse) for b in diverse[i+1:]))
                for hit in pool['candidates']:
                    self.assertLessEqual(hit['start']+c['length']+max(c['horizons']), engine.store.series[0]['memory_end'])
                result = engine.rank(pool, .5)
                self.assertEqual(len(result['hits']), 5)
                self.assertAlmostEqual(sum(h['weight'] for h in result['hits']), 1.)
                self.assertTrue(np.isfinite(result['prediction']).all())
                with self.assertRaises(ValueError): engine.prepare(item['x'], 0, 0)
                with patch.object(engine.base.library, 'search', return_value=([], {})):
                    empty = engine.prepare(item['x'], 0, int(item['start']))
                    np.testing.assert_equal(engine.rank(empty)['prediction'], np.full(16, item['x'][-1]))
            finally:
                engine.close()
                dataset.store.close()


if __name__ == '__main__':
    unittest.main()
