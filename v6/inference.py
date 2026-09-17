"""This module accepts history only; candidate futures are historical memory."""
import time
import copy
import numpy as np
import torch
from v5.data import Store
from .common import sha256, check_store, read_json
from pathlib import Path
from .data import scale_floor
from .train import load
from .index import Library


def analog(store, c, x, floor, hits):
    m, h = c['length'], max(c['horizons'])
    center, scale = float(np.mean(x)), max(float(np.std(x)), floor)
    mapped = []
    for hit in hits:
        s = store.series[hit['sid']]
        past = store.window(hit['sid'], hit['start'], m).astype(float)
        future = store.window(hit['sid'], hit['start']+m, h).astype(float)
        # Stable positive scale mapping, not an unconstrained least-squares slope.
        cs = max(float(past.std()), scale_floor(s, c))
        mapped.append(center + scale * (future-past.mean()) / cs)
    if not mapped:
        return np.full(h, float(x[-1])), np.empty((0, h))
    distances = np.array([hit['distance'] for hit in hits])
    temperature = max(float(np.median(distances)), 1e-6)
    weights = np.exp(-(distances-distances.min())/temperature)
    weights /= weights.sum()
    return weights @ np.array(mapped), np.array(mapped)


class Retriever:
    def __init__(self, store, checkpoint, index, device='cpu', fusion_path=None):
        self.model, self.state = load(checkpoint, device, 'encoder')
        self.c, self.device = copy.deepcopy(self.state['config']), device
        self.store, self.library = Store(store), Library(index)
        check_store(self.store, self.c)
        if self.store.meta['data_id'] != self.state['data_id'] or self.library.meta['data_id'] != self.state['data_id']:
            raise ValueError('Index/checkpoint/store identity mismatch')
        if self.library.meta['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Index was built with different weights')
        self.c['index']['channels'] = list(self.library.c['index']['channels'])
        self.c['index']['joint_history_weight']=self.library.c['index'].get('joint_history_weight',.5)
        self.model.c['index']['joint_history_weight']=self.c['index']['joint_history_weight']
        self.c['evaluation']['history_max_nmse']=self.library.c['evaluation'].get('history_max_nmse',.5)
        self.index_sha=sha256(Path(index)/'manifest.json')
        self.fusion=read_json(fusion_path) if fusion_path else None

    def close(self):
        self.library.close(); self.store.close()

    def encode(self, x, sid):
        if not 0 <= sid < len(self.store.series):
            raise ValueError('Unknown logical series ID')
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self.c['length'],) or not np.isfinite(x).all():
            raise ValueError('Query must be finite history of the configured length')
        floor = scale_floor(self.store.series[sid], self.c)
        started = time.perf_counter()
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x[None]).to(self.device), torch.tensor([floor], device=self.device))
            # CPU copies synchronize CUDA; elapsed includes transfer and computation.
            vectors = {name: out[name][0].cpu().numpy() for name in self.c['index']['channels']}
            prior = {k: v[0].cpu().numpy() for k, v in out['prior'].items()}
            forecast = out['forecast'][0].cpu().numpy() * prior['scale'] + prior['center']
        return vectors, prior, forecast, (time.perf_counter()-started)*1000

    def retrieve(self, x, sid, start=None, channel=None, leaf_budget=0):
        began = time.perf_counter()
        channel=channel or (self.fusion['method'].split('_leaves')[0] if self.fusion else
                            ('joint' if 'joint' in self.c['index']['channels'] else 'learned'))
        vectors, prior, forecast, encode_ms = self.encode(x, sid)
        hits, stats, prediction, mapped = self.search_encoded(x, sid, start, vectors[channel], channel, leaf_budget)
        raw_prediction=prediction
        if self.fusion:
            from .fusion import check, apply
            name=f'{channel}_leaves{leaf_budget}'
            check(self.fusion,self.state['data_id'],self.library.meta['checkpoint_sha256'],self.index_sha,self.c,name)
            prediction=apply(self.fusion,dict(persistence=np.full(len(prediction),float(x[-1])),
                encoder_forecast=forecast,probabilistic_prior=prior['mean']*prior['scale']+prior['center'],**{name:raw_prediction}))
        stats.update(encode_ms=encode_ms, total_ms=(time.perf_counter()-began)*1000)
        return dict(hits=hits, stats=stats, prediction=prediction, raw_prediction=raw_prediction, mapped=mapped, prior=prior, forecast=forecast)

    def search_encoded(self, x, sid, start, vector, channel, leaf_budget):
        e = self.c['evaluation']; scope = e['scope']
        began = time.perf_counter()
        gate_stats={}
        def history_filter(hits):
            # Raw observed history only. No candidate/query future is examined.
            q=np.asarray(x,dtype=float)
            qz=(q-q.mean())/max(float(q.std()),scale_floor(self.store.series[sid],self.c))
            selected=[]; checked=0; limit=e['history_max_nmse']; m=self.c['length']
            for a in range(0,len(hits),64):
                batch=[hit for hit in hits[a:a+64] if all(hit['sid']!=old['sid'] or abs(hit['start']-old['start'])>=m for old in selected)]
                if not batch: continue
                past=np.empty((len(batch),m),dtype=np.float64)
                ids=np.array([hit['sid'] for hit in batch]); positions=np.array([hit['start'] for hit in batch],dtype=np.int64)
                for source_sid in np.unique(ids):
                    mask=ids==source_sid
                    # Read a bounded batch of histories, not thousands of Python window calls.
                    past[mask]=self.store.values(int(source_sid))[positions[mask,None]+np.arange(m)[None,:]]
                if not np.isfinite(past).all(): raise ValueError('Indexed history is no longer finite')
                floors=np.array([scale_floor(self.store.series[hit['sid']],self.c) for hit in batch])
                z=(past-past.mean(1)[:,None])/np.maximum(past.std(1),floors)[:,None]
                ds=np.mean((z-qz)**2,axis=1); checked+=len(ds)
                for hit,d in zip(batch,ds):
                    if d<=limit and all(hit['sid']!=old['sid'] or abs(hit['start']-old['start'])>=m for old in selected):
                        selected.append(dict(hit,history_nmse=float(d)))
                        if len(selected)==e['k']: break
                if len(selected)==e['k']: break
            gate_stats.update(history_checked=checked,history_threshold=limit)
            return selected
        hits, stats = self.library.search(vector, channel, e['k'],
            sid=sid if scope == 'series' else None,
            group=self.store.series[sid]['group'] if scope == 'group' else None,
            query_sid=sid, query_start=start, leaf_budget=leaf_budget,
            time_budget_ms=e['time_budget_ms'],**({'candidate_filter':history_filter} if channel=='joint' else {}))
        stats.update(gate_stats)
        prediction, mapped = analog(self.store, self.c, x, scale_floor(self.store.series[sid], self.c), hits)
        stats['search_and_fetch_ms'] = (time.perf_counter()-began)*1000
        return hits, stats, prediction, mapped
