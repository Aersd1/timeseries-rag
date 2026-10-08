"""Validate a shared history weight, freeze it, then evaluate held-out A50 reranking."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v6.data import Windows
from .common import fresh_dir, write_json, sha256
from .learned_rerank import LearnedMoiraiReranker
from .__main__ import serializable

WEIGHTS = [.2, .35, .5, .65, .8]


def select_weight(tables, weights=WEIGHTS):
    """Dataset-equal relative NMSE objective; never accept test rows."""
    if not weights or any(not 0 < w < 1 for w in weights):
        raise ValueError('Require nonzero history and future contributions')
    if not tables or any(set(frame['split']) != {'validation'} for frame in tables.values()):
        raise ValueError('Weight selection accepts validation data only')
    objectives = {}
    for weight in weights:
        ratios = []
        for frame in tables.values():
            baseline = frame[frame.method == 'a50_original_order']
            scored = frame[frame.method == f'blend_{weight:g}']
            keys = ['query', 'sid', 'start', 'horizon']
            matched = baseline.merge(scored, on=keys, validate='one_to_one', suffixes=('_base', '_blend'))
            if len(matched) != len(baseline) or len(matched) != len(scored) or not len(matched):
                raise ValueError('Incomplete paired validation coverage')
            ratios.append(float(matched.nmse_blend.mean()/max(matched.nmse_base.mean(), 1e-12)))
        objectives[str(weight)] = float(np.mean(ratios))
    chosen = min(weights, key=lambda w: (objectives[str(w)], abs(w-.5), w))
    return dict(history_weight=chosen, future_weight=1-chosen,
        selection_split='validation', selection_objective='Mean across datasets of blended NMSE / same-pool original-order NMSE',
        candidate_weights=weights, objective_values=objectives, datasets=list(tables))


def evaluate_dataset(source, output, name, split, weights, device):
    src = source/name
    out = fresh_dir(output/name/split)
    engine = LearnedMoiraiReranker(src/'store', src/'train/best.pt', src/'index_a', device)
    dataset = Windows(src/'store', engine.c, split)
    run = dict(complete=False, dataset=name, split=split, queries=len(dataset), weights=weights,
        candidates=50, top_k=5, selection='ordinary', history_metric='separately standardized raw-history NMSE',
        future_metric='multi-horizon quantile pinball', hard_history_gate=False,
        data_id=engine.store.meta['data_id'], config=engine.c,
        checkpoint_sha256=sha256(src/'train/best.pt'), index_sha256=sha256(src/'index_a/manifest.json'))
    write_json(out/'run.json', run)
    metrics, details, timings = [], [], []
    try:
        for qi in range(len(dataset)):
            item = dataset[qi]
            x, y, sid, start = item['x'], item['y'], int(item['sid']), int(item['start'])
            prepared = engine.prepare(x, sid, start)
            original = engine.base.retrieve(x, sid, start, channel='learned')
            forecasts = dict(a_original_diverse=original['prediction'],
                a50_original_order=prepared['baseline_prediction'], moirai_direct=prepared['direct_prediction'],
                persistence=prepared['fallback'])
            selected = {}
            for weight in weights:
                result = engine.rank(prepared, weight)
                method = f'blend_{weight:g}'
                forecasts[method] = result['prediction']
                selected[method] = result['hits']
            for method, prediction in forecasts.items():
                for h in engine.c['horizons']:
                    error = np.asarray(prediction[:h], dtype=float)-y[:h]
                    mse = float(np.mean(error**2))
                    metrics.append(dict(dataset=name, split=split, query=qi, sid=sid, start=start,
                        horizon=h, method=method, mse=mse, mae=float(np.abs(error).mean()),
                        nmse=mse/max(engine.store.series[sid]['memory_std'], 1e-6)**2))
            span = engine.c['length']+max(engine.c['horizons'])
            boundary, independent = -1, 0
            for position in sorted(h['start'] for h in prepared['candidates']):
                if position >= boundary:
                    independent += 1
                    boundary = position+span
            details.append(dict(query=qi, sid=sid, start=start, candidates=prepared['candidates'],
                selected=selected, independent_episodes=independent))
            timings.append(dict(query=qi, **prepared['stats']))
            print(name, split, qi+1, '/', len(dataset), flush=True)
        for label, rows in [('metrics', metrics), ('retrieval', details), ('timing', timings)]:
            with (out/(label+'.jsonl')).open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(serializable(row), allow_nan=False)+'\n')
        run['complete'] = True
        write_json(out/'run.json', run)
        return pd.DataFrame(metrics)
    finally:
        engine.close()
        dataset.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='v7/runs/all_datasets_20261008')
    parser.add_argument('--output', default='v7/runs/a50_rerank_20261008')
    parser.add_argument('--datasets', nargs='+', default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    source, output = Path(args.source), fresh_dir(args.output)
    write_json(output/'protocol.json', dict(datasets=args.datasets, candidates=50, top_k=5,
        validation_weights=WEIGHTS, control_weights=[0., 1.], device=args.device,
        policy='Ordinary top50 learned recall; overlaps allowed. Select one shared positive history weight on validation before test.',
        source=str(source), code_sha256={str(p):sha256(p) for p in [Path(__file__), Path('v7/learned_rerank.py'), Path('v6/index.py')]}))
    tables = {name:evaluate_dataset(source, output, name, 'validation', WEIGHTS, args.device) for name in args.datasets}
    policy = select_weight(tables)
    policy['validation_artifacts'] = {name:sha256(output/name/'validation/metrics.jsonl') for name in args.datasets}
    write_json(output/'policy.json', policy)
    policy_sha = sha256(output/'policy.json')
    print('FROZEN POLICY', policy, flush=True)
    rows = []
    for name in args.datasets:
        scores = evaluate_dataset(source, output, name, 'test', [0., policy['history_weight'], 1.], args.device)
        assert sha256(output/'policy.json') == policy_sha
        rows.append(dict(dataset=name, **scores.groupby('method').nmse.mean().to_dict()))
    pd.DataFrame(rows).to_csv(output/'dataset_metrics.csv', index=False)
    write_json(output/'summary.json', dict(complete=True, policy_sha256=policy_sha, policy=policy, datasets=rows))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
