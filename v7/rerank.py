"""Variant B: a single V4 retrieval, then history-only Moirai future scoring."""
import time
from pathlib import Path
import numpy as np
from v5.data import Store
from v6.data import scale_floor
from v6.inference import analog
from .common import config, read_json, write_json, fresh_dir, check_store, sha256
from .moirai import MoiraiAdapter


def v4_api():
    # V4 predates package-relative imports. Fail on conflicting imports instead
    # of silently binding the root/V3 model or kernels. Use a fresh CLI process.
    import sys
    root = Path(__file__).resolve().parents[1]/'v4'
    for name in ('forecast', 'index', 'model', 'cascade', 'kernels', 'tree'):
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module, '__file__', '')).parent.resolve() != root.resolve():
            raise RuntimeError(f'Conflicting legacy module {name}; run v7 in a fresh process')
    from v5.teacher import v4_modules
    return v4_modules()


def build(store_path, config_path, output):
    from v5.teacher import make_oracle
    c = config(config_path)
    v4_api()
    store = Store(store_path); check_store(store, c)
    out = fresh_dir(output)
    meta = dict(version=7, variant='b', complete=False, config=c, data_id=store.meta['data_id'], series={})
    write_json(out/'manifest.json', meta)
    teacher_config = dict(c, teacher=dict(max_points_per_series=c['rerank']['max_points_per_series'],
        chunk_points=c['rerank']['chunk_points']))
    try:
        for s in store.series:
            path = out/f'{s["sid"]:06d}'
            engine = make_oracle(store, s['sid'], teacher_config, path)
            if engine is None:
                raise ValueError(f'Series {s["sid"]}: insufficient V4 training cases')
            engine.close()
            meta['series'][str(s['sid'])] = dict(path=path.name,
                base_sha256=sha256(path/'base/manifest.json'), cascade_sha256=sha256(path/'cascade/manifest.json'))
            write_json(out/'manifest.json', meta)
        meta['complete'] = True
        write_json(out/'manifest.json', meta)
        return meta
    finally:
        store.close()


def aggregate(mapped, scores):
    scores = np.asarray(scores, dtype=float)
    tau = max(float(np.median(scores)), 1e-6)
    weights = np.exp(-(scores-scores.min())/tau); weights /= weights.sum()
    return weights @ mapped, weights


def future_scores(quantiles, levels, candidate_futures, center, scale, horizons, kind='pinball'):
    """Compatibility score, NOT a likelihood; candidates are observed memory futures."""
    q = (np.asarray(quantiles, dtype=float)-center)/scale
    ys = (np.asarray(candidate_futures, dtype=float)-center)/scale
    levels = np.asarray(levels, dtype=float)
    if (q.ndim != 2 or ys.ndim != 2 or q.shape[0] != len(levels) or ys.shape[1] != q.shape[1]
            or scale <= 0 or not np.isfinite(q).all() or not np.isfinite(ys).all()
            or np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(levels) <= 0)
            or not horizons or min(horizons) < 1 or max(horizons) > q.shape[1]):
        raise ValueError('Invalid quantiles, candidates or horizons')
    # Monotone rearrangement fixes quantile crossing; no time-axis pooling/CDF grid.
    q = np.sort(q, axis=0)
    if kind == 'pinball':
        residual = ys[:, None, :]-q[None]
        losses = np.maximum(levels[None, :, None]*residual, (levels[None, :, None]-1)*residual)
        return sum(losses[:, :, :h].mean(axis=(1, 2)) for h in horizons)/len(horizons)
    if kind == 'median_mse':
        median = np.flatnonzero(np.isclose(levels, .5))
        if len(median) != 1:
            raise ValueError('Need a median quantile')
        return sum(((ys[:, :h]-q[median[0], :h])**2).mean(1) for h in horizons)/len(horizons)
    raise ValueError('Unknown future score')


class V4MoiraiReranker:
    def __init__(self, store, index, device='cpu', adapter=None):
        self.path = Path(index)
        self.meta = read_json(self.path/'manifest.json')
        if self.meta.get('variant') != 'b' or not self.meta.get('complete'):
            raise ValueError('Expected complete variant B index')
        self.c = self.meta['config']
        self.store = Store(store); check_store(self.store, self.c)
        if self.store.meta['data_id'] != self.meta['data_id']:
            raise ValueError('Store/index mismatch')
        self.adapter = adapter or MoiraiAdapter.pretrained(self.c['moirai'], device)
        self.engines = {}

    def engine(self, sid):
        if sid not in self.engines:
            entry = self.meta['series'][str(sid)]; path = self.path/entry['path']
            for name in ('base', 'cascade'):
                if sha256(path/name/'manifest.json') != entry[name+'_sha256']:
                    raise ValueError('V4 index manifest changed')
            ForecastIndex, _, _, _ = v4_api()
            self.engines[sid] = ForecastIndex(path/'cascade')
        return self.engines[sid]

    def close(self):
        for engine in self.engines.values():
            engine.close()
        self.store.close()

    def retrieve(self, x, sid, start):
        began = time.perf_counter()
        x = np.asarray(x, dtype=np.float32)
        if not 0 <= sid < len(self.store.series):
            raise ValueError('Unknown sid')
        s = self.store.series[sid]
        if x.shape != (self.c['length'],) or not np.isfinite(x).all():
            raise ValueError('Expected finite configured history')
        if start is None or start < s['memory_end']:
            raise ValueError('Query must follow the fixed memory prefix')
        engine = self.engine(sid); policy = self.c['rerank']; horizon = max(self.c['horizons'])
        # Select the original V4 algorithm even if optional local acceleration exists.
        search = getattr(engine, 'search_baseline', engine.search)
        result = search(x, horizon, s, int(start), 'same_series', capacity=policy['capacity'], k=20)
        hits = []
        for rank, (distance, local_sid, local_start) in enumerate(result[policy['selection']], 1):
            location = engine.idx.manifest['series'][int(local_sid)]
            position = int(location.get('start', 0))+int(local_start)
            if position+self.c['length']+horizon > min(start, s['memory_end']):
                raise AssertionError('Candidate future is not entirely in historical memory')
            hits.append(dict(sid=sid, start=position, distance=float(distance), v4_rank=rank))
        _, mapped = analog(self.store, self.c, x, scale_floor(s, self.c), hits)
        model_began = time.perf_counter()
        quantiles = self.adapter.forecast(x[None], horizon).cpu().numpy()[0]
        model_ms = (time.perf_counter()-model_began)*1000
        levels = np.asarray(self.adapter.module.quantile_levels)
        median_id = int(np.argmin(np.abs(levels-.5)))
        direct = quantiles[median_id]
        k = policy['top_k']
        if hits:
            fs = future_scores(quantiles, levels, mapped, float(x.mean()),
                max(float(x.std()), scale_floor(s, self.c)), self.c['horizons'], policy['score'])
            hs = np.array([hit['distance'] for hit in hits])
            beta = policy['history_weight']
            scores = (1-beta)*fs/max(float(np.median(fs)), 1e-6)+beta*hs/max(float(np.median(hs)), 1e-6)
            order = np.lexsort((np.array([h['start'] for h in hits]), hs, scores))
            for i, hit in enumerate(hits):
                hit.update(history_distance=hit['distance'], future_score=float(fs[i]), rerank_score=float(scores[i]))
            chosen = order[:k]
            prediction, weights = aggregate(mapped[chosen], scores[chosen])
            baseline, _ = aggregate(mapped[:k], hs[:k])
            selected = [dict(hits[i], weight=float(w)) for i, w in zip(chosen, weights)]
            ranked = [hits[i] for i in order]
        else:
            prediction = baseline = np.full(horizon, float(x[-1]))
            selected, ranked = [], []
        return dict(prediction=prediction, baseline_prediction=baseline, direct_prediction=direct,
            hits=selected, candidates=ranked, quantiles=quantiles, quantile_levels=levels.tolist(),
            stats=dict(total_ms=(time.perf_counter()-began)*1000, moirai_ms=model_ms,
                retrieval_ms=result['retrieval_ms'], requested_candidates=20, returned_candidates=len(hits),
                complete_candidates=len(hits)==20, requested_k=k, returned=len(selected),
                complete_topk=len(selected)==k, v4_selection=policy['selection'],
                v4_episode_complete=bool(result['episode_complete']), v4_calls=1))
