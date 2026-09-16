import tempfile
import unittest
from pathlib import Path
import numpy as np
from forecast import ForecastIndex, transfer, aggregate
from model import LearnedShapeModel, znorm
from index import IndexBuilder
from cascade import Cascade


class ForecastTests(unittest.TestCase):
    def test_transfer_affine_and_flat(self):
        x = np.arange(244, dtype=float)
        future = np.arange(244, 268, dtype=float)[None, :]
        q = 3*x+8
        for method in ('mean_std', 'endpoint', 'affine'):
            np.testing.assert_allclose(transfer(q, x[None, :], future, method), 3*future+8, atol=1e-9)
            np.testing.assert_allclose(transfer(q, np.zeros((1,244)), future, method), q[-1])
        pred, eff, _ = aggregate(np.array([[0., 0.], [1., 1.], [99., 99.]]), np.ones(3), 'median')
        np.testing.assert_equal(pred, [1., 1.]); self.assertAlmostEqual(eff, 3)

    def test_self_future_boundary_scope_and_exact_prefix(self):
        rng = np.random.default_rng(9); x = rng.normal(size=1800); other = rng.normal(size=1800)
        m = 244; h = 96; p = 1100; q = x[p:p+m]
        model = LearnedShapeModel().fit(x[rng.integers(0, 500, 30)[:, None]+np.arange(m)])
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/'base'; builder = IndexBuilder(base, model)
            meta = dict(source='x.csv', column='x', device=None, start=0, train_end=len(x))
            builder.add(x, meta); builder.add(other, dict(meta, source='other.csv'))
            builder.finish(); Cascade.build(base, Path(temp)/'cascade', long_size=1024, mid_size=256)
            engine = ForecastIndex(Path(temp)/'cascade')
            try:
                for scope in ('same_series', 'pooled_train'):
                    result = engine.search(q, h, meta, p, scope, capacity=128, k=10)
                    truth = []
                    for sid, series in enumerate((x, other)):
                        if scope == 'same_series' and sid: continue
                        last = p-m-h if sid == 0 else len(series)-m-h
                        windows = np.lib.stride_tricks.sliding_window_view(series, m)[:last+1]
                        ds = np.sum((znorm(windows)-znorm(q))**2, axis=1)
                        truth.extend((float(d), sid, i) for i, d in enumerate(ds))
                    truth.sort()
                    np.testing.assert_allclose([r[0] for r in result['ordinary']], [r[0] for r in truth[:10]], atol=2e-5)
                    self.assertTrue(all(sid != 0 or pos+m+h <= p for _, sid, pos in result['ordinary']))
                    self.assertNotIn((0, p), [(s, pp) for _, s, pp in result['ordinary']])
                    for i, (_, sid, pos) in enumerate(result['episodes']):
                        for _, other_sid, other_pos in result['episodes'][:i]:
                            self.assertTrue(sid != other_sid or abs(pos-other_pos) >= m+h)
                # Last valid future start included, one-past excluded.
                full = engine.search(q, h, meta, p, 'same_series', capacity=2000, k=1000)
                positions = [pp for _, _, pp in full['ordinary']]
                self.assertIn(p-m-h, positions); self.assertNotIn(p-m-h+1, positions)
            finally: engine.close()


if __name__ == '__main__': unittest.main()
