import unittest
from unittest.mock import Mock

from v7.current_forecast import CurrentPredictor, PRIMARY


class CurrentForecastTests(unittest.TestCase):
    def predictor(self):
        model = object.__new__(CurrentPredictor)
        model.dataset = 'ETTm1'
        model.engine = Mock()
        return model

    def test_fixed_output_is_used_even_when_guard_rejects(self):
        model = self.predictor()
        old = dict(forecasts={PRIMARY: [2.] * 96, 'guarded_dual': [0.] * 96},
                   used_cases=0, retrieved_cases=10, query_future_used=False)
        model.engine.predict.return_value = old
        result = model.predict([1.] * 96, 0, 1000)
        self.assertEqual(result['forecast'], [2.] * 96)
        self.assertEqual(result['primary'], PRIMARY)
        self.assertEqual(result['used_cases'], 10)
        self.assertEqual(old['used_cases'], 0)
        model.engine.predict.assert_called_once_with([1.] * 96, 0, 1000)

    def test_cached_route_uses_same_fixed_output(self):
        model = self.predictor()
        model.engine.predict_cached.return_value = dict(forecasts={PRIMARY: [3.] * 96})
        result = model.predict_cached('history', 1, 1200, 'quantiles', 'probe')
        self.assertEqual(result['forecast'], [3.] * 96)
        model.engine.predict_cached.assert_called_once_with('history', 1, 1200, 'quantiles', 'probe')

    def test_missing_fixed_output_cannot_fall_back_to_another_model(self):
        model = self.predictor()
        with self.assertRaises(KeyError):
            model._primary(dict(forecasts={'guarded_dual': [0.] * 96}))

    def test_close_on_error_and_reject_unsupported_dataset(self):
        model = self.predictor()
        with self.assertRaises(RuntimeError):
            with model:
                raise RuntimeError('query failed')
        model.engine.close.assert_called_once()
        with self.assertRaises(ValueError):
            CurrentPredictor('illness')


if __name__ == '__main__':
    unittest.main()
