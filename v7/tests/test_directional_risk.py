import unittest
import numpy as np

from v7.directional_risk import inputs, fit, weights, blend
from v7.benchmark_directional_risk import choose


class DirectionalRiskTests(unittest.TestCase):
    def test_optimal_weight_can_reject_or_keep_a_frozen_correction(self):
        n = np.zeros((12, 96)); r = np.ones_like(n); x = np.zeros((12, 3))
        starts = np.repeat(np.arange(4)*192, 3); energy = np.ones(12)
        for direction, expected in [(2., 1.), (-1., 0.), (.4, .4)]:
            model = fit(x, r*direction, n, r, np.ones(12), starts)
            np.testing.assert_allclose(weights(model, x, energy, 0.), expected, atol=1e-7)

    def test_exact_endpoints_and_zero_energy(self):
        n = np.arange(192, dtype=float).reshape(2, 96); r = -n
        np.testing.assert_array_equal(blend(n, r, np.array([0., 1.])), np.stack([n[0], r[1]]))
        model = dict(mean=np.zeros(1), std=np.ones(1), coefficients=np.ones((5, 2)))
        np.testing.assert_array_equal(weights(model, np.zeros((2, 1)), np.zeros(2), 0.), 0.)

    def test_real_histories_are_inputs_and_query_future_is_not(self):
        rng = np.random.default_rng(3); n = 3
        data = dict(raw_x=rng.normal(size=(n, 96)), sid=np.zeros(n, dtype=int),
                    raw_quantiles=rng.normal(size=(n, 9, 96)),
                    case_past=rng.normal(size=(n, 10, 96)), case_future=rng.normal(size=(n, 10, 96)),
                    raw_y=rng.normal(size=(n, 96)))
        stats = dict(std=[1.], mean=[0.]); base = np.zeros((n, 96)); r = base+1
        x = inputs(data, base, r, stats)[0]
        poisoned = dict(data, raw_y=np.full((n, 96), 1e12))
        np.testing.assert_array_equal(x, inputs(poisoned, base, r, stats)[0])
        # Keep all old futures and the candidate forecast fixed; change only old histories.
        changed = dict(data, case_past=data['case_past'][:, :, ::-1])
        self.assertGreater(np.max(abs(x-inputs(changed, base, r, stats)[0])), .01)
        np.testing.assert_array_equal(inputs(data, base, r, stats, True)[0],
                                      inputs(changed, base, r, stats, True)[0])

    def test_confirmation_rejects_without_selecting_a_second_policy(self):
        d = dict(raw_y=np.r_[np.ones((3,96)), np.full((3,96),2.)],
                 sid=np.zeros(6, dtype=int), std=np.ones(6))
        r = np.full((6,96),2.); m = np.zeros((6,96)); a = np.ones(6)
        candidates = {'first':(np.ones((6,96)),a), 'second':(np.full((6,96),2.5),a)}
        policy = choose(candidates, d, dict(std=[1.]), np.arange(3), np.arange(3,6), r, m)
        self.assertEqual(policy['proposed'], 'first')
        self.assertEqual(policy['selected'], 'restored')
        self.assertFalse(policy['confirmed'])


if __name__ == '__main__': unittest.main()
