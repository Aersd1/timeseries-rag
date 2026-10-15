"""Current forecast: the restored, frozen, fixed-increment 5+5 pipeline."""
import json
from pathlib import Path

DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity', 'traffic')
RUN_ROOT = Path('v7/runs/dual_five_distinct_96_20261014')
STORE_ROOT = Path('v7/runs/all_datasets_20261008')
PRIMARY = 'fixed_increment_transport'


class CurrentPredictor:
    """Use the same restored output for every dataset, without retraining.

    Run from the repository root, as required by the saved source contracts.
    The old 5+5 predictor retains its policy/hash/temporal-boundary validation.
    Its guarded output is diagnostic only here: PRIMARY is the fixed 0.25
    increment, including on variables where guarded_dual rejects that increment.
    """

    def __init__(self, dataset, device='cuda', adapter=None):
        from .query_dual_five import DualFivePredictor
        if dataset not in DATASETS:
            raise ValueError(f'Unsupported dataset: {dataset}; choose from {DATASETS}')
        folder = RUN_ROOT / dataset
        run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        contract = json.loads((folder / 'contract.json').read_text(encoding='utf-8'))
        if not run.get('complete') or PRIMARY not in run.get('metrics', {}):
            raise ValueError('The restored fixed-increment run is incomplete')
        if contract.get('raw_case_spacing') != 192:
            raise ValueError('Expected the original distinct raw-history five-case bank')
        self.dataset = dataset
        self.engine = DualFivePredictor(STORE_ROOT / dataset / 'store', folder, device, adapter)

    def _primary(self, result):
        # Do not accidentally use guarded_dual or a later selector's output.
        return dict(result, forecast=result['forecasts'][PRIMARY], primary=PRIMARY,
                    dataset=self.dataset, scheme='fixed_increment_5plus5',
                    guarded_policy=dict(family=result.get('family'), beta=result.get('beta')),
                    family='increment_transport', beta=0.25,
                    increment_strength=0.25, used_cases=10)

    def predict(self, history, sid, start):
        return self._primary(self.engine.predict(history, sid, start))

    def predict_cached(self, history, sid, start, quantiles, probe):
        return self._primary(self.engine.predict_cached(history, sid, start, quantiles, probe))

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
