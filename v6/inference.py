"""This module accepts history only; candidate futures are historical memory."""
import time
import numpy as np
import torch
from v5.data import Store
from .common import sha256, check_store
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
    def __init__(self, store, checkpoint, index, device='cpu'):
        self.model, self.state = load(checkpoint, device, 'encoder')
        self.c, self.device = self.state['config'], device
        self.store, self.library = Store(store), Library(index)
        check_store(self.store, self.c)
        if self.store.meta['data_id'] != self.state['data_id'] or self.library.meta['data_id'] != self.state['data_id']:
            raise ValueError('Index/checkpoint/store identity mismatch')
        if self.library.meta['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Index was built with different weights')

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

    def retrieve(self, x, sid, start=None, channel='learned', leaf_budget=0):
        began = time.perf_counter()
        vectors, prior, forecast, encode_ms = self.encode(x, sid)
        hits, stats, prediction, mapped = self.search_encoded(x, sid, start, vectors[channel], channel, leaf_budget)
        stats.update(encode_ms=encode_ms, total_ms=(time.perf_counter()-began)*1000)
        return dict(hits=hits, stats=stats, prediction=prediction, mapped=mapped, prior=prior, forecast=forecast)

    def search_encoded(self, x, sid, start, vector, channel, leaf_budget):
        e = self.c['evaluation']; scope = e['scope']
        began = time.perf_counter()
        hits, stats = self.library.search(vector, channel, e['k'],
            sid=sid if scope == 'series' else None,
            group=self.store.series[sid]['group'] if scope == 'group' else None,
            query_sid=sid, query_start=start, leaf_budget=leaf_budget,
            time_budget_ms=e['time_budget_ms'])
        prediction, mapped = analog(self.store, self.c, x, scale_floor(self.store.series[sid], self.c), hits)
        stats['search_and_fetch_ms'] = (time.perf_counter()-began)*1000
        return hits, stats, prediction, mapped
