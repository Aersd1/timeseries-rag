"""Historical analog forecasting. No query future is accepted by retrieval/transfer."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import time
import numpy as np
from cascade import Cascade
import kernels


def series_key(meta):
    return (os.path.normcase(os.path.normpath(meta['source'])),
            str(meta.get('column')), str(meta.get('device')))


def transfer(q, past, future, method):
    q, past, future = map(lambda a: np.asarray(a, dtype=float), (q, past, future))
    mq, sq = q.mean(), q.std()
    mx, sx = past.mean(axis=1), past.std(axis=1)
    # Flat histories do not identify a scale/slope: conservative persistence.
    valid = (sx > 1e-10) & (sq > 1e-10)
    ratio = np.divide(sq, sx, out=np.zeros_like(sx), where=valid)
    if method == 'mean_std':
        result = mq + ratio[:, None] * (future - mx[:, None])
    elif method == 'endpoint':
        result = q[-1] + ratio[:, None] * (future - past[:, -1, None])
    elif method == 'affine':
        cov = np.mean((past-mx[:, None])*(q-mq), axis=1)
        a = np.divide(cov, sx*sx, out=np.zeros_like(sx), where=valid)
        result = mq + a[:, None]*(future-mx[:, None])
    else:
        raise ValueError('Unknown transfer')
    result[~valid] = q[-1]
    return result


def aggregate(mapped, squared_distances, kind='mean'):
    ds = np.asarray(squared_distances)
    # Query-local temperature depends ONLY on past distances, never test error.
    tau = max(float(np.median(ds)), 1e-8)
    weights = np.exp(-(ds-ds.min())/tau)
    weights /= weights.sum()
    if kind == 'mean':
        pred = weights @ mapped
    elif kind == 'median':
        order = np.argsort(mapped, axis=0, kind='stable')
        cumulative = np.cumsum(weights[order], axis=0)
        rows = np.argmax(cumulative >= .5, axis=0)
        pred = np.take_along_axis(mapped, order, axis=0)[rows, np.arange(mapped.shape[1])]
    else:
        raise ValueError('Unknown aggregation')
    return pred, float(1/(weights @ weights)), tau


class ForecastIndex:
    """Resident V3 symbols, long-block ordering, bounded exact nearest prefix.

    Scope and full-future eligibility are filtered BEFORE building any upper
    bound. No disallowed self-match can prune eligible historical neighbors.
    """
    def __init__(self, path):
        self.base = Cascade(path)
        self.idx = self.base.idx
        self.m = self.idx.model.length
        self.keys = [series_key(s) for s in self.idx.manifest['series']]
        self.block_lo = []
        self.block_hi = []
        for sid, a, b, off in self.base.blocks:
            codes = self.base.codes[off:off+b-a]
            self.block_lo.append(codes.min(axis=0))
            self.block_hi.append(codes.max(axis=0))
        self.block_lo = np.asarray(self.block_lo, dtype=np.uint8)
        self.block_hi = np.asarray(self.block_hi, dtype=np.uint8)

    def search(self, q, horizon, query_meta, query_start, scope='same_series',
               capacity=8192, k=100):
        if horizon < 1 or capacity < k or k < 1:
            raise ValueError('Invalid horizon/candidate count')
        if scope not in ('same_series', 'pooled_train'):
            raise ValueError('Unknown scope')
        started = time.perf_counter()
        model = self.idx.model
        z, ql, qu, _ = model.query(q)
        feature = model.transform(q)
        lookup = np.maximum(np.maximum(model.left-1e-4-feature[:, None],
                                       feature[:, None]-model.right-1e-4), 0.)**2
        lower = kernels.bounds(self.block_lo, self.block_hi, model, ql, qu)
        key = series_key(query_meta)
        best_d = np.empty(0)
        best_s = np.empty(0, dtype=np.int64)
        best_p = np.empty(0, dtype=np.int64)
        eligible = 0
        verified = 0
        for bid in np.argsort(lower, kind='stable'):
            sid, a, b, off = map(int, self.base.blocks[bid])
            meta = self.idx.manifest['series'][sid]
            same = self.keys[sid] == key
            if scope == 'same_series' and not same:
                continue
            # Future must be entirely resident in the training portion.
            b = min(b, meta['n']-self.m-horizon+1)
            if same:
                # Exclusive candidate end <= first query sample.
                b = min(b, int(query_start)-int(meta.get('start', 0))-self.m-horizon+1)
            if b <= a:
                continue
            eligible += b-a
            cutoff = float(best_d[-1]) if len(best_d) >= capacity else np.inf
            if lower[bid] > cutoff+1e-7:
                continue
            codes = self.base.codes[off:off+b-a]
            positions = kernels.filter_codes(codes, lookup, model.pairs*2, cutoff)+a
            if not len(positions):
                continue
            ds = kernels.exact_cached(self.idx.values(sid), positions, z,
                                      self.base.stats[sid], cutoff=cutoff)
            verified += len(ds)
            good = np.flatnonzero(np.isfinite(ds) & (ds <= cutoff))
            if len(good) > capacity:
                good = good[np.argpartition(ds[good], capacity-1)[:capacity]]
            d = np.r_[best_d, ds[good]]
            s = np.r_[best_s, np.full(len(good), sid, dtype=np.int64)]
            p = np.r_[best_p, positions[good]]
            order = np.lexsort((p, s, d))[:capacity]
            best_d, best_s, best_p = d[order], s[order], p[order]
        ordinary = list(zip(best_d[:k], best_s[:k], best_p[:k]))
        episodes = []
        blocked = {}
        for d, sid, p in zip(best_d, best_s, best_p):
            sid, p = int(sid), int(p)
            start = int(self.idx.manifest['series'][sid].get('start', 0))+p
            starts = blocked.setdefault(self.keys[sid], [])
            if any(abs(start-prev) < self.m+horizon for prev in starts):
                continue
            episodes.append((d, sid, p))
            starts.append(start)
            if len(episodes) == k:
                break
        return dict(ordinary=ordinary, episodes=episodes,
                    episode_complete=len(episodes) == k or eligible <= capacity,
                    eligible=eligible, retained=len(best_d), verified=verified,
                    retrieval_ms=(time.perf_counter()-started)*1000)

    def random(self, horizon, query_meta, query_start, scope, rng, k=100):
        key = series_key(query_meta)
        sids, counts = [], []
        for sid, meta in enumerate(self.idx.manifest['series']):
            same = self.keys[sid] == key
            if scope == 'same_series' and not same:
                continue
            n = meta['n']-self.m-horizon+1
            if same:
                n = min(n, query_start-int(meta.get('start', 0))-self.m-horizon+1)
            if n > 0:
                sids.append(sid)
                counts.append(n)
        edges = np.r_[0, np.cumsum(counts)]
        if not edges[-1]:
            return []
        selected = rng.choice(int(edges[-1]), size=min(k, int(edges[-1])), replace=False)
        return [(0., sids[j], int(p-edges[j])) for p in selected
                for j in [int(np.searchsorted(edges, p, side='right')-1)]]

    def arrays(self, hits, horizon):
        past, future = [], []
        for _, sid, p in hits:
            x = self.idx.values(int(sid)); p = int(p)
            past.append(x[p:p+self.m])
            future.append(x[p+self.m:p+self.m+horizon])
        return np.asarray(past), np.asarray(future)

    def close(self):
        self.idx.close()
