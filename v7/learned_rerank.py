"""A learned recall, then Moirai/history scoring and optional overlap penalty."""
import time
import numpy as np
import torch
from v6.data import scale_floor
from v6.inference import analog
from .retrieval import MoiraiRetriever
from .rerank import aggregate, future_scores


def history_scores(x, histories, query_floor, candidate_floors):
    """Raw shape NMSE after separate observed-history normalization."""
    x, histories = np.asarray(x, dtype=float), np.asarray(histories, dtype=float)
    if x.ndim != 1 or histories.ndim != 2 or histories.shape[1] != len(x):
        raise ValueError('Invalid histories')
    floors = np.asarray(candidate_floors, dtype=float)
    if floors.shape != (len(histories),) or query_floor <= 0 or np.any(floors <= 0):
        raise ValueError('Invalid scale floors')
    if not np.isfinite(x).all() or not np.isfinite(histories).all():
        raise ValueError('Nonfinite histories')
    q = (x-x.mean())/max(float(x.std()), query_floor)
    z = (histories-histories.mean(1)[:, None])/np.maximum(histories.std(1), floors)[:, None]
    return np.mean((z-q)**2, axis=1)


def blend_scores(future, history, history_weight):
    """Normalize score scales within the pool; smaller blended score is better."""
    future, history = np.asarray(future, dtype=float), np.asarray(history, dtype=float)
    if future.ndim != 1 or future.shape != history.shape or not len(future):
        raise ValueError('Invalid score shapes')
    if not 0 <= history_weight <= 1 or not np.isfinite(future).all() or not np.isfinite(history).all():
        raise ValueError('Invalid scores or weight')
    if np.any(future < 0) or np.any(history < 0):
        raise ValueError('Scores must be nonnegative')
    return ((1-history_weight)*future/max(float(np.median(future)), 1e-6)
            + history_weight*history/max(float(np.median(history)), 1e-6))


class LearnedMoiraiReranker:
    def __init__(self, store, checkpoint, index, device='cpu', candidates=100, top_k=5):
        if not isinstance(candidates, int) or not isinstance(top_k, int) or not 1 <= top_k <= candidates:
            raise ValueError('Expected integer 1 <= top_k <= candidates')
        self.base = MoiraiRetriever(store, checkpoint, index, device)
        self.c, self.store = self.base.c, self.base.store
        self.adapter = self.base.model.prior.adapter
        self.candidates, self.top_k = candidates, top_k

    def close(self):
        self.base.close()

    def prepare(self, x, sid, start):
        """Accept history only; reuse one recall and one forecast for weight sweeps."""
        return self.prepare_batch(np.asarray(x,dtype=np.float32)[None],[sid],[start])[0]

    def prepare_batch(self, histories, sids, starts):
        """History-only batch inference for gate training; no query labels accepted."""
        histories=np.asarray(histories,dtype=np.float32)
        if histories.ndim!=2 or histories.shape[1]!=self.c['length'] or not np.isfinite(histories).all():
            raise ValueError('Invalid history batch')
        if len(histories)!=len(sids) or len(sids)!=len(starts) or not len(histories):
            raise ValueError('Batch identities differ')
        for sid,start in zip(sids,starts):
            if not 0<=sid<len(self.store.series) or start is None or start<self.store.series[sid]['memory_end']:
                raise ValueError('Batch query must follow fixed historical memory')
        began=time.perf_counter()
        with torch.inference_mode():
            x=torch.as_tensor(histories,device=self.base.device)
            floors=torch.tensor([scale_floor(self.store.series[s],self.c) for s in sids],device=self.base.device)
            vectors=self.base.model(x,floors)['learned'].cpu().numpy()
        encode_ms=(time.perf_counter()-began)*1000/len(histories)
        began=time.perf_counter()
        quantiles=self.adapter.forecast(histories,max(self.c['horizons'])).cpu().numpy()
        forecast_ms=(time.perf_counter()-began)*1000/len(histories)
        return [self._prepare_encoded(x,sid,start,vector,q,encode_ms,forecast_ms)
            for x,sid,start,vector,q in zip(histories,sids,starts,vectors,quantiles)]

    def _prepare_encoded(self, x, sid, start, vector, quantiles, encode_ms, forecast_ms):
        began = time.perf_counter()
        if not 0 <= sid < len(self.store.series):
            raise ValueError('Unknown sid')
        s = self.store.series[sid]
        if start is None or start < s['memory_end']:
            raise ValueError('Query must follow fixed historical memory')
        x = np.asarray(x, dtype=np.float32)
        hits, stats = self.base.library.search(vector, 'learned', self.candidates,
            sid=sid, query_sid=sid, query_start=start, capacity=self.candidates, selection='ordinary')
        m, horizon = self.c['length'], max(self.c['horizons'])
        for rank, hit in enumerate(hits, 1):
            if hit['sid'] != sid or hit['start']+m+horizon > min(start, s['memory_end']):
                raise AssertionError('Candidate outside historical memory')
            hit['learned_rank'] = rank
        _, mapped = analog(self.store, self.c, x, scale_floor(s, self.c), hits)
        levels = np.asarray(self.adapter.module.quantile_levels)
        direct = quantiles[int(np.flatnonzero(np.isclose(levels, .5))[0])]
        fs = hs = np.empty(0)
        if hits:
            fs = future_scores(quantiles, levels, mapped, float(x.mean()),
                max(float(x.std()), scale_floor(s, self.c)), self.c['horizons'], 'pinball')
            past = np.stack([self.store.window(sid, hit['start'], m) for hit in hits])
            hs = history_scores(x, past, scale_floor(s, self.c), [scale_floor(s, self.c)]*len(hits))
            for hit, f, h in zip(hits, fs, hs):
                hit.update(future_score=float(f), history_nmse=float(h))
        baseline, _ = analog(self.store, self.c, x, scale_floor(s, self.c), hits[:self.top_k])
        return dict(candidates=hits, mapped=mapped, future_scores=fs, history_scores=hs,
            quantiles=quantiles,quantile_levels=levels,
            baseline_prediction=baseline, direct_prediction=direct,
            fallback=np.full(horizon, float(x[-1])),
            stats=dict(stats, recall_calls=1, encode_ms=encode_ms, moirai_forecast_ms=forecast_ms,
                preparation_ms=(time.perf_counter()-began)*1000+encode_ms+forecast_ms,
                requested_candidates=self.candidates, returned_candidates=len(hits),
                complete_candidates=len(hits)==self.candidates))

    def rank(self, prepared, history_weight=.2, diversity_weight=0.):
        return rank_prepared(prepared,history_weight,self.top_k,
            self.c['length']+max(self.c['horizons']),diversity_weight)

    def retrieve(self, x, sid, start, history_weight=.2, diversity_weight=0.):
        began = time.perf_counter()
        prepared = self.prepare(x, sid, start)
        result = self.rank(prepared, history_weight, diversity_weight)
        result.update(candidates=prepared['candidates'], baseline_prediction=prepared['baseline_prediction'],
            direct_prediction=prepared['direct_prediction'],
            stats=dict(prepared['stats'], total_ms=(time.perf_counter()-began)*1000,
                requested_k=self.top_k, returned=len(result['hits']), complete_topk=len(result['hits'])==self.top_k))
        return result


def diverse_selection(hits,scores,k,span,strength):
    """Greedy soft full-episode overlap penalty; always keep k when available.

    Base compatibility remains the aggregation score. This penalty changes only
    selection and never overrides query/candidate chronological eligibility.
    """
    scores=np.asarray(scores,dtype=float)
    if span<1 or not np.isfinite(strength) or strength<0 or k<1 or len(hits)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid diversity inputs')
    starts=np.asarray([h['start'] for h in hits]); sids=np.asarray([h['sid'] for h in hits])
    order=np.lexsort((starts,np.asarray([h['distance'] for h in hits]),scores))
    if not strength: return order[:k]
    selected=[]; remaining=set(range(len(hits)))
    for _ in range(min(k,len(hits))):
        def key(i):
            overlap=max((max(0.,1-abs(float(starts[i]-starts[j]))/span)
                if sids[i]==sids[j] else 0. for j in selected),default=0.)
            return (scores[i]+strength*overlap,scores[i],hits[i]['distance'],starts[i])
        chosen=min(remaining,key=key); selected.append(chosen); remaining.remove(chosen)
    return np.asarray(selected,dtype=np.int64)


def rank_prepared(prepared,history_weight,top_k,span,diversity_weight=0.):
    """Pure history-only ranking; cached policies need not reload Moirai."""
    if not 0<=history_weight<=1 or not np.isfinite(diversity_weight) or diversity_weight<0:
        raise ValueError('Invalid ranking weights')
    hits=prepared['candidates']
    if not hits:
        return dict(prediction=prepared['fallback'],hits=[],history_weight=history_weight,diversity_weight=diversity_weight)
    scores=blend_scores(prepared['future_scores'],prepared['history_scores'],history_weight)
    chosen=diverse_selection(hits,scores,top_k,span,diversity_weight)
    prediction,weights=aggregate(prepared['mapped'][chosen],scores[chosen])
    return dict(prediction=prediction,history_weight=history_weight,diversity_weight=diversity_weight,
        hits=[dict(hits[i],rerank_score=float(scores[i]),weight=float(w)) for i,w in zip(chosen,weights)])
