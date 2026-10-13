"""Causal trend/residual decomposition and history-only residual retrieval."""
import numpy as np
from .rerank import future_scores, aggregate
from .learned_rerank import diverse_selection

LENGTH = HORIZON = 96


def decompose(values, window=25):
    """Trailing mean, with the first observed value as left padding.

    Each output only depends on its prefix, including for old continuations.
    Passing history+future for supervision cannot change historical features.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1 or not np.isfinite(x).all() or window < 1:
        raise ValueError('Expected finite nonempty time series and positive window')
    padded = np.concatenate([np.repeat(x[..., :1], window-1, axis=-1), x], axis=-1)
    sums = np.concatenate([np.zeros_like(x[..., :1]), np.cumsum(padded, axis=-1)], axis=-1)
    trend = (sums[..., window:]-sums[..., :-window])/window
    return trend.astype(np.float32), (x-trend).astype(np.float32)


def normalize(x, floor):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(-1, keepdims=True)
    scale = np.maximum(x.std(-1, keepdims=True), floor)
    return (x-mean)/scale, mean, scale


class ResidualPool:
    def __init__(self, store, sid, window=25, end=None, stride=1):
        s = store.series[sid]
        limit=s['memory_end'] if end is None else int(end)
        if not 0<limit<=s['n'] or stride<1: raise ValueError('Invalid pool endpoint/stride')
        starts = np.array([p for a,b in s['runs'] for p in range(a, min(b,limit)-192+1,stride)], dtype=np.int64)
        if len(starts) < 100:
            raise ValueError('Require at least 100 complete memory episodes')
        raw = np.stack([store.window(sid,p,192) for p in starts])
        _, residual = decompose(raw, window)
        self.floor = max(s['memory_std']*.01, 1e-6)
        self.past, self.mean, self.scale = normalize(residual[:,:96], self.floor)
        self.future = (residual[:,96:]-self.mean)/self.scale
        self.starts, self.sid, self.memory_end = starts, sid, limit

    def retrieve(self, query, query_start, quantiles, levels):
        """No query future accepted. Known old futures only rerank a history shortlist."""
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (96,) or not np.isfinite(q).all():
            raise ValueError('Expected normalized residual history of length 96')
        allowed = self.starts+192 <= min(query_start,self.memory_end)
        ids = np.flatnonzero(allowed)
        if len(ids) < 100:
            raise ValueError('Insufficient nonoverlapping historical episodes')
        distances = np.mean((self.past[ids]-q)**2, axis=-1)
        order = np.lexsort((self.starts[ids],distances))[:100]
        ids, hs = ids[order], distances[order]
        fs = future_scores(quantiles, levels, self.future[ids], 0., 1., [96])
        score = .8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)
        hits = [dict(sid=self.sid,start=int(p),distance=float(d)) for p,d in zip(self.starts[ids],hs)]
        chosen = diverse_selection(hits,score,5,192,1.)
        analog, weights = aggregate(self.future[ids[chosen]],score[chosen])
        examples = np.concatenate([self.past[ids[chosen]],self.future[ids[chosen]]],axis=-1)
        return dict(examples=examples.astype(np.float32),weights=weights.astype(np.float32),
            valid=np.ones(5,dtype=bool),selected_starts=self.starts[ids[chosen]],
            candidate_starts=self.starts[ids],history_scores=hs,future_scores=fs,
            analog=analog.astype(np.float32))
