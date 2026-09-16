"""Frozen V4 exact engine creates reference-only train/validation/test labels."""
import sys
from pathlib import Path
import numpy as np
from .common import read_json, write_json, fingerprint, valid_ranges, sample_ranges, affine, norm_scale
from .data import Store


def v4_modules():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'v4'))
    from forecast import ForecastIndex
    from index import IndexBuilder
    from model import LearnedShapeModel
    from cascade import Cascade
    return ForecastIndex, IndexBuilder, LearnedShapeModel, Cascade


def make_oracle(store, sid, c, dest):
    ForecastIndex, IndexBuilder, LearnedShapeModel, Cascade = v4_modules()
    s = store.series[sid]; m = c['length']; h = max(c['horizons'])
    ranges = list(valid_ranges(s, m, h, 0, s['memory_end']))
    if not ranges: return None
    points = sum(min(b, s['memory_end'])-a for a, b in s['runs'] if a < s['memory_end'])
    if points > c['teacher']['max_points_per_series']:
        raise ValueError(f"Series {sid}: {points} memory points exceeds explicit RAM guard. Increase teacher.max_points_per_series on a suitably sized server; no silent truncation.")
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    if not (dest/'base'/'manifest.json').exists():
        if (dest/'base').exists():
            raise ValueError(f'Interrupted V4 index build at {dest}/base; use a fresh teacher output/cache directory')
        positions = sample_ranges(ranges, 2048, np.random.default_rng(c['seed']+sid))
        if len(positions) < 10: return None
        samples = np.array([store.window(sid, p, m) for p in positions])
        model = LearnedShapeModel(length=m).fit(samples)
        builder = IndexBuilder(dest/'base', model)
        step = c['teacher']['chunk_points']-m-h+1
        for a, b in s['runs']:
            b = min(b, s['memory_end'])
            for p in range(a, b-m-h+1, step):
                end = min(b, p+c['teacher']['chunk_points'])
                builder.add(np.asarray(store.values(sid)[p:end], dtype=float),
                            dict(source=s['source'], column=s['column'], device=s['device'],
                                 group=s['group'], start=p, train_end=s['memory_end']))
        builder.finish()
    if not (dest/'cascade'/'manifest.json').exists():
        Cascade.build(dest/'base', dest/'cascade')
    return ForecastIndex(dest/'cascade')


def generate(store_path, output):
    store = Store(store_path); c = read_json(Path(store_path)/'config.json')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    identity = dict(data_id=store.meta['data_id'], config_hash=fingerprint(c), version=5)
    if (out/'identity.json').exists():
        if read_json(out/'identity.json') != identity: raise ValueError('Teacher output belongs to another dataset/config')
    else: write_json(out/'identity.json', identity)
    m, horizons, k = c['length'], c['horizons'], c['teacher']['neighbors']
    records, skipped = [], []
    for sid, s in enumerate(store.series):
        folder = out/f'series_{sid:06d}'; done = folder/'done.json'
        if done.exists():
            state = read_json(done); records.extend(state['shards']); skipped.extend(state['skipped']); continue
        folder.mkdir(exist_ok=True)
        engine = make_oracle(store, sid, c, out/'oracle'/f'{sid:06d}')
        local_records, local_skipped = [], []
        if engine is None:
            local_skipped.append(dict(sid=sid, reason='fewer than 10 complete memory cases'))
        else:
            try:
                boundaries = dict(train=(s['memory_end'], s['train_end']),
                                  validation=(s['train_end'], s['validation_end']), test=(s['validation_end'], s['n']))
                for split, (begin, end) in boundaries.items():
                    seed = c['seed']+sid*11+list(boundaries).index(split)
                    starts = sample_ranges(valid_ranges(s, m, max(horizons), begin, end, m+max(horizons)),
                                           c['teacher']['queries_per_series'][split], np.random.default_rng(seed))
                    if not len(starts):
                        local_skipped.append(dict(sid=sid, split=split, reason='no complete disjoint context/future'))
                    buffer = []
                    for number, p in enumerate(starts):
                        q = store.window(sid, p, m)
                        result = engine.search(q, max(horizons), s, int(p), 'same_series', capacity=k, k=k)
                        hits = result['ordinary']
                        if len(hits) < 2: continue
                        cp = np.full(k, -1, dtype=np.int64); dp = np.zeros(k, dtype=np.float32)
                        df = np.zeros((k, len(horizons)), dtype=np.float32); mask = np.zeros(k, dtype=bool)
                        for j, (distance, hs, hp) in enumerate(hits):
                            hs, hp = int(hs), int(hp)
                            p0 = int(engine.idx.manifest['series'][hs]['start'])+hp
                            if p0+m+max(horizons) > min(s['memory_end'], int(p)):
                                raise AssertionError('Candidate future reaches query/test period')
                            cp[j] = p0; dp[j] = np.sqrt(max(0., distance)); mask[j] = True
                        past = np.array([store.window(sid, pos, m) for pos in cp[mask]])
                        scale = norm_scale(q, s['memory_std'])
                        for hi, h in enumerate(horizons):
                            future = np.array([store.window(sid, pos+m, h) for pos in cp[mask]])
                            truth = store.window(sid, int(p)+m, h)
                            df[mask, hi] = np.mean((affine(q, past, future)-truth)**2, axis=1)/(scale*scale)
                        if not np.isfinite(df).all(): raise ValueError('Nonfinite teacher error')
                        hard = mask & (dp <= np.median(dp[mask])) & (df.mean(axis=1) >= np.quantile(df[mask].mean(axis=1), .75))
                        buffer.append(dict(sid=sid, start=int(p), candidates=cp, past_distance=dp, future_nmse=df,
                                           valid=mask, episode_id=np.where(mask, cp//(m+max(horizons)), -1),
                                           hard_negative=hard, retrieval_ms=result['retrieval_ms']))
                        if len(buffer) >= c['teacher']['shard_queries'] or number == len(starts)-1:
                            path = folder/f'{split}_{len(local_records):06d}.npz'
                            np.savez_compressed(path, **{key: np.array([v[key] for v in buffer]) for key in buffer[0]})
                            local_records.append(dict(path=str(path.relative_to(out)), split=split, count=len(buffer), sid=sid)); buffer = []
                    print(f'teacher series={sid} split={split} queries={len(starts)}', flush=True)
            finally: engine.close()
        write_json(done, dict(shards=local_records, skipped=local_skipped))
        records.extend(local_records); skipped.extend(local_skipped)
    manifest = dict(**identity, complete=True, shards=records, skipped=skipped, horizons=horizons, length=m,
                    counts={split:sum(r['count'] for r in records if r['split'] == split) for split in ('train','validation','test')},
                    scope='same logical series, fixed memory prefix, stride-1 exact V4 teacher; arbitrary equal-distance ties')
    write_json(out/'manifest.json', manifest)
    store.close()
    return manifest


class Labels:
    def __init__(self, path, split):
        self.path = Path(path); self.meta = read_json(self.path/'manifest.json')
        if not self.meta['complete']: raise ValueError('Incomplete teacher labels')
        self.shards = [r for r in self.meta['shards'] if r['split'] == split]
        self.edges = np.r_[0, np.cumsum([r['count'] for r in self.shards])]
        self.cached_id, self.cached = None, None

    def __len__(self): return int(self.edges[-1])

    def __getitem__(self, i):
        if not 0 <= i < len(self): raise IndexError(i)
        j = int(np.searchsorted(self.edges, i, side='right')-1)
        if j != self.cached_id:
            with np.load(self.path/self.shards[j]['path'], allow_pickle=False) as z:
                self.cached = {k: z[k] for k in z.files}
            self.cached_id = j
        return {k: v[i-self.edges[j]] for k, v in self.cached.items()}
