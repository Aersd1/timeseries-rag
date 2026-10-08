"""Paired held-out evaluation: true query futures are accessed only here."""
import json
from pathlib import Path
import numpy as np
from v6.data import Windows, scale_floor
from .common import config, fresh_dir, write_json, sha256


def evaluate(store, config_path, output, checkpoint_a=None, index_a=None, index_b=None,
             checkpoint_v6=None, index_v6=None, split='test', device='cpu', leaf_budget=0):
    from .retrieval import MoiraiRetriever
    from .rerank import V4MoiraiReranker
    from v6.inference import Retriever
    if split not in ('validation', 'test'):
        raise ValueError('Use validation or test')
    if bool(checkpoint_a) != bool(index_a) or bool(checkpoint_v6) != bool(index_v6):
        raise ValueError('Supply both checkpoint and index for each neural retriever')
    if not index_a and not index_b:
        raise ValueError('Provide at least one variant')
    c = config(config_path); out = fresh_dir(output)
    dataset = Windows(store, c, split)
    engines = []
    run = dict(version=7, complete=False, split=split, config=c, data_id=dataset.store.meta['data_id'],
        queries=len(dataset), leaf_budget=leaf_budget, artifacts={}, timing_note='Per-query latency; model load excluded; B engines preloaded per series; no warmup. First query per series marked.')
    write_json(out/'run.json', run)
    records, timings, details = [], [], []
    try:
        a = MoiraiRetriever(store, checkpoint_a, index_a, device) if index_a else None
        if a: engines.append(a)
        b = V4MoiraiReranker(store, index_b, device) if index_b else None
        if b: engines.append(b)
        old = Retriever(store, checkpoint_v6, index_v6, device) if index_v6 else None
        if old: engines.append(old)
        for name, path in [('checkpoint_a', checkpoint_a), ('index_a', index_a), ('index_b', index_b),
                           ('checkpoint_v6', checkpoint_v6), ('index_v6', index_v6)]:
            if path:
                run['artifacts'][name] = sha256(Path(path)/'manifest.json' if name.startswith('index') else path)
        for engine in engines:
            for key in ('length', 'horizons', 'split', 'normalization'):
                if engine.c[key] != c[key]: raise ValueError(f'Paired evaluation mismatch: {key}')
            if engine.c['evaluation']['scope'] != 'series':
                raise ValueError('Paired evaluation requires same-series scope')
        if a and a.c != c:
            raise ValueError('Use the A checkpoint configuration for paired evaluation')
        if b and b.c != c:
            raise ValueError('Use the B index configuration for paired evaluation')
        if a and old:
            for section, keys in [('index', ('stride','oversample','joint_history_weight')),
                                  ('evaluation', ('k','history_max_nmse','time_budget_ms'))]:
                for key in keys:
                    if a.c[section].get(key) != old.c[section].get(key):
                        raise ValueError(f'A/V6 comparison changes retrieval policy: {section}.{key}')
            if a.library.meta['windows'] != old.library.meta['windows']:
                raise ValueError('A/V6 indices contain different numbers of memory windows')
        run['comparison_policies'] = {name:dict(k=engine.c['evaluation']['k'],
            stride=engine.c['index']['stride'],oversample=engine.c['index']['oversample'])
            for name,engine in [('a',a),('v6',old)] if engine is not None}
        if b:
            run['comparison_policies']['b'] = b.c['rerank']
        seen = set()
        for qi in range(len(dataset)):
            item = dataset[qi]; x, y = item['x'], item['y']
            sid, start = int(item['sid']), int(item['start'])
            s = dataset.store.series[sid]
            forecasts = dict(persistence=np.full(len(y), x[-1]))
            first = sid not in seen; seen.add(sid)
            for name, engine in [('a', a), ('v6', old)]:
                if engine is None: continue
                for channel in ('learned', 'history', 'joint'):
                    if channel not in engine.c['index']['channels']: continue
                    result = engine.retrieve(x, sid, start, channel=channel, leaf_budget=leaf_budget)
                    method = f'{name}_{channel}'
                    forecasts[method] = result['prediction']
                    forecasts[f'{name}_encoder_direct'] = result['forecast']
                    timings.append(dict(query=qi, sid=sid, start=start, method=method, first_query=first, **result['stats']))
                    details.append(dict(query=qi, sid=sid, start=start, method=method, hits=result['hits']))
            if b:
                b.engine(sid)
                result = b.retrieve(x, sid, start)
                forecasts.update(b_reranked=result['prediction'], b_v4_original_order=result['baseline_prediction'],
                    moirai_direct=result['direct_prediction'])
                timings.append(dict(query=qi, sid=sid, start=start, method='b_reranked', first_query=first, **result['stats']))
                details.append(dict(query=qi, sid=sid, start=start, method='b_reranked', hits=result['hits'], candidates=result['candidates']))
            for method, prediction in forecasts.items():
                for h in c['horizons']:
                    error = np.asarray(prediction[:h], dtype=float)-y[:h]
                    mse = float(np.mean(error**2))
                    records.append(dict(query=qi, sid=sid, start=start, horizon=h, method=method, mse=mse,
                        mae=float(np.abs(error).mean()), nmse=mse/max(s['memory_std'], 1e-6)**2,
                        history_nmse=mse/max(float(x.std()), scale_floor(s,c))**2))
            print(dict(query=qi+1, total=len(dataset)), flush=True)
        for name, rows in [('metrics', records), ('timing', timings), ('retrieval', details)]:
            with (out/f'{name}.jsonl').open('w', encoding='utf-8') as f:
                for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
        run['complete'] = True
        write_json(out/'run.json', run)
        summary = summarize(records, timings)
        write_json(out/'summary.json', summary)
        lines = ['# Moirai variants: paired evaluation', '', f'Split: {split}; queries: {len(dataset)}.',
            '', 'NMSE divides by memory-only variance. These means alone do not establish general superiority.', '',
            '|H|Method|N|Mean NMSE|Win rate vs persistence|', '|---:|---|---:|---:|---:|']
        for row in summary['forecast']:
            lines.append(f"|{row['horizon']}|{row['method']}|{row['n']}|{row['nmse']:.6g}|{row['win_rate']:.3f}|")
        lines += ['', 'B original-order and reranked forecasts use the same candidate pool, mapping and K.',
            'The Moirai direct baseline uses its median. No query future enters either retrieval path.',
            'See summary.json for paired changes, per-query outputs for failures/underfill, and run.json for identities.',
            'Model/index initialization is excluded from reported query latency; this is not a strict latency guarantee.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        return summary
    finally:
        for engine in engines: engine.close()
        dataset.store.close()


def summarize(records, timings):
    table = {(r['query'], r['horizon'], r['method']):r for r in records}
    methods = sorted({r['method'] for r in records})
    output = []
    for h in sorted({r['horizon'] for r in records}):
        for method in methods:
            rows = [r for r in records if r['method']==method and r['horizon']==h]
            output.append(dict(horizon=h, method=method, n=len(rows), nmse=float(np.mean([r['nmse'] for r in rows])),
                win_rate=float(np.mean([r['nmse']<table[(r['query'],h,'persistence')]['nmse'] for r in rows]))))
    pairs = []
    for left,right in [('b_reranked','b_v4_original_order'), ('b_reranked','moirai_direct'),
                       ('a_joint','v6_joint'), ('a_learned','v6_learned'), ('a_joint','a_history')]:
        if left not in methods or right not in methods: continue
        for h in sorted({r['horizon'] for r in records}):
            rows = [r for r in records if r['method']==left and r['horizon']==h]
            ds = np.array([table[(r['query'],h,right)]['nmse']-r['nmse'] for r in rows])
            # Resample entire logical series, preserving paired queries and series sizes.
            sids = np.array([r['sid'] for r in rows]); unique = np.unique(sids)
            rng = np.random.default_rng(20261008)
            clusters = [ds[sids==sid] for sid in unique]
            boots = [np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]).mean() for _ in range(1000)]
            pairs.append(dict(left=left, right=right, horizon=h, n=len(ds), mean_improvement=float(ds.mean()),
                win_rate=float(np.mean(ds>0)), series_clusters=len(unique),
                cluster_bootstrap_95=np.quantile(boots,[.025,.975]).tolist() if len(unique)>1 else None))
    latency = []
    for method in sorted({r['method'] for r in timings}):
        rows = [r for r in timings if r['method']==method]
        times = [r['total_ms'] for r in rows]
        latency.append(dict(method=method, p50_ms=float(np.percentile(times,50)), p95_ms=float(np.percentile(times,95)),
            max_ms=float(max(times)), complete_topk_fraction=float(np.mean([r['complete_topk'] for r in rows]))))
    return dict(forecast=output, paired=pairs, timing=latency,
        caveat='Few series imply weak bootstrap evidence; compression fidelity is not measured by forecast error alone.')
