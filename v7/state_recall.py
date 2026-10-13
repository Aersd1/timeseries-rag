"""State-aware shortlist; only observed histories enter the recall distance."""
import numpy as np
from .trend_rag import decompose, normalize
from .rerank import future_scores, aggregate
from .learned_rerank import diverse_selection

STATE_WEIGHTS = [0., .05, .2, 1.]


def history_state(history, memory_std):
    x = np.asarray(history, dtype=np.float64)
    if x.shape[-1] != 96 or not np.isfinite(x).all():
        raise ValueError('Expected finite96-point histories')
    std = max(float(memory_std), 1e-6)
    t = np.arange(96, dtype=float)-47.5
    trend = (x @ t)/(t @ t)*95
    return np.stack([x.mean(-1)/std,
        np.log(np.maximum(x.std(-1), std*.1)/std), trend/std], axis=-1)


def state_distance(query_state, candidate_states, query_start, starts, period):
    distance = np.sum((candidate_states-query_state)**2, axis=-1)
    # For weekly data there is no identifiable within-day phase.
    if period is not None and period >= 2:
        phase = 2*np.pi*((query_start-np.asarray(starts)) % period)/period
        distance += 1-np.cos(phase)
        return distance/4
    return distance/3


class StateRecall:
    def __init__(self, pool, raw_histories, memory_std, period):
        self.pool = pool
        self.states = history_state(raw_histories, memory_std)
        self.std, self.period = memory_std, period
        if len(self.states) != len(pool.starts):
            raise ValueError('Pool/state alignment mismatch')

    def retrieve(self, history, quantiles, start, weight, window=25):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('State weight must be finite and nonnegative')
        x = np.asarray(history, dtype=np.float32)
        quantiles = np.asarray(quantiles, dtype=np.float32)
        if x.shape != (96,) or quantiles.shape != (9,96) or not np.isfinite(quantiles).all():
            raise ValueError('Expected history and nine predicted quantiles')
        pool = self.pool
        _, r = decompose(x, window)
        q, mean, scale = normalize(r, pool.floor)
        paths = np.concatenate([np.repeat(x[None], 9, axis=0), quantiles], axis=1)
        _, forecast_residual = decompose(paths, window)
        predicted = (forecast_residual[:,96:]-mean)/scale
        ids = np.flatnonzero(pool.starts+192 <= min(start, pool.memory_end))
        if len(ids) < 100:
            raise ValueError('Need100 complete nonoverlapping historical episodes')
        shape = np.mean((pool.past[ids]-q)**2, axis=-1)
        state = state_distance(history_state(x, self.std), self.states[ids], start, pool.starts[ids], self.period)
        # Exactly reproduce the old float32 distance and tie order at weight0.
        distance = shape if weight == 0 else shape+weight*state
        chosen = np.lexsort((pool.starts[ids], distance))[:100]
        ids, hs, ds = ids[chosen], shape[chosen], state[chosen]
        fs = future_scores(predicted, np.arange(1,10)/10, pool.future[ids], 0., 1., [96])
        prior = .8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)
        hits = [dict(sid=pool.sid, start=int(p), distance=float(d)) for p,d in zip(pool.starts[ids],hs)]
        selected = diverse_selection(hits, prior, 5, 192, 1.)
        _, weights = aggregate(pool.future[ids[selected]], prior[selected])
        return dict(candidate_starts=pool.starts[ids], history_scores=hs, future_scores=fs,
            state_scores=ds, prior=prior, selected_starts=pool.starts[ids[selected]], weights=weights)
