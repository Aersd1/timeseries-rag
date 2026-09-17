"""Temporal shards -> enclosing boxes -> leaf embeddings; bounded top-M heap.

Exact mode proves nearest stored float32 vectors, not nearest real futures.
Budget mode is approximate and never sets certified=True after early stopping.
"""
import heapq
import time
import copy
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from v5.data import Store
from v5.common import valid_ranges
from .common import write_json, read_json, sha256, fresh_dir, check_store
from .data import scale_floor
from .train import load


def spatial_order(vectors, leaf_size):
    """Balanced, leaf-aligned median splits; bounded by one temporal shard."""
    order = np.arange(len(vectors), dtype=np.int64)
    pending = [(0, len(order))]
    while pending:
        a, b = pending.pop()
        if b-a <= leaf_size:
            continue
        ids = order[a:b]
        values = np.asarray(vectors[ids])
        axis = int(np.ptp(values, axis=0).argmax())
        left = ((b-a + leaf_size-1)//leaf_size//2)*leaf_size
        order[a:b] = ids[np.argpartition(values[:, axis], left)]
        pending.extend([(a, a+left), (a+left, b)])
    return order


def repack(index_path, output, channels=None, joint_history_weight=None, history_max_nmse=None):
    """Rebuild geometry from existing embeddings. No model fit or re-encoding."""
    source = Library(index_path)
    meta = copy.deepcopy(source.meta)
    selected = channels or meta['config']['index']['channels']
    available=set(meta['config']['index']['channels'])
    if {'learned','history'} <= available: available.add('joint')
    if not selected or set(selected)-available:
        raise ValueError('Unknown channel')
    weight=meta['config']['index'].get('joint_history_weight',.5) if joint_history_weight is None else joint_history_weight
    if not 0<weight<1: raise ValueError('Joint history weight must be between 0 and 1')
    if 'joint' in meta['config']['index']['channels'] and weight!=meta['config']['index'].get('joint_history_weight',.5):
        raise ValueError('To change joint weights, repack the original learned/history channels')
    out = fresh_dir(output); began = time.perf_counter()
    meta.update(complete=False, layout='spatial', source_manifest_sha256=sha256(Path(index_path)/'manifest.json'))
    meta['config']['index'].update(channels=list(selected), layout='spatial')
    meta['config']['index']['joint_history_weight']=weight
    if 'joint' in selected:
        gate=meta['config']['evaluation'].get('history_max_nmse',.5) if history_max_nmse is None else history_max_nmse
        if gate<=0: raise ValueError('Positive raw-history threshold required')
        meta['config']['evaluation']['history_max_nmse']=gate
    write_json(out/'manifest.json', meta)
    for shard_id, shard in enumerate(meta['shards']):
        path = out/shard['path']; path.mkdir()
        np.save(path/'starts.npy', source.array(shard_id, 'starts.npy'))
        levels = {}
        for channel in selected:
            cp = path/channel; cp.mkdir()
            base_channel='learned' if channel=='joint' and channel not in source.c['index']['channels'] else channel
            v = source.array(shard_id, f'{base_channel}/vectors.npy')
            start_file = f'{base_channel}/starts.npy' if source.meta.get('layout') == 'spatial' else 'starts.npy'
            starts = source.array(shard_id, start_file)
            if channel=='joint' and channel not in source.c['index']['channels']:
                hs=source.array(shard_id,'history/starts.npy' if source.meta.get('layout')=='spatial' else 'starts.npy')
                order_h=np.argsort(hs); positions=np.searchsorted(hs[order_h],starts)
                if np.any(positions>=len(hs)) or not np.array_equal(hs[order_h][positions],starts):
                    raise ValueError('Learned/history windows differ')
                hv=source.array(shard_id,'history/vectors.npy')[order_h[positions]]
                v=np.concatenate([np.float32(np.sqrt(1-weight))*v,np.float32(np.sqrt(weight))*hv],axis=1).astype('<f4')
            order = spatial_order(v, meta['config']['index']['leaf_size'])
            dest = np.lib.format.open_memmap(cp/'vectors.npy', mode='w+', dtype='<f4', shape=v.shape)
            for a in range(0, len(order), 4096):
                dest[a:a+4096] = v[order[a:a+4096]]
            dest.flush(); np.save(cp/'starts.npy', starts[order])
            levels[channel] = boxes(cp, dest, meta['config']['index']['leaf_size'], meta['config']['index']['fanout'])
        shard['levels'] = levels
        print(dict(repacked_shard=shard_id+1, total=len(meta['shards'])), flush=True)
    source.close()
    meta.update(complete=True, repack_seconds=time.perf_counter()-began,
                payload_bytes=sum(p.stat().st_size for p in out.rglob('*.npy')))
    write_json(out/'manifest.json', meta)
    return meta


def boxes(path, vectors, leaf_size, fanout):
    """Bounded construction workspace: one level of boxes, not all vectors."""
    level, levels = 0, []
    current = vectors
    block = leaf_size
    while True:
        n = (len(current) + block - 1) // block
        lo = np.lib.format.open_memmap(path / f'lo{level}.npy', mode='w+', dtype='<f4', shape=(n, vectors.shape[1]))
        hi = np.lib.format.open_memmap(path / f'hi{level}.npy', mode='w+', dtype='<f4', shape=lo.shape)
        for i in range(n):
            a, b = i * block, min((i + 1) * block, len(current))
            if level == 0:
                lo[i] = np.nextafter(current[a:b].min(0), np.float32(-np.inf))
                hi[i] = np.nextafter(current[a:b].max(0), np.float32(np.inf))
            else:
                lo[i] = previous_lo[a:b].min(0)
                hi[i] = previous_hi[a:b].max(0)
        lo.flush(); hi.flush(); levels.append(n)
        if n == 1:
            return levels
        previous_lo, previous_hi = lo, hi
        current, block, level = lo, fanout, level + 1


def build(store_path, checkpoint, output, device='cpu'):
    model, state = load(checkpoint, device, 'encoder')
    c = state['config']; store = Store(store_path); check_store(store, c)
    if store.meta['data_id'] != state['data_id']:
        raise ValueError('Checkpoint/store mismatch')
    out = fresh_dir(output)
    meta = dict(version=6, complete=False, data_id=state['data_id'], checkpoint_sha256=sha256(checkpoint),
                config=c, shards=[], windows=0, raw_points=store.meta['total_points'])
    meta['layout']=c['index'].get('layout','temporal')
    write_json(out / 'manifest.json', meta)
    started = time.perf_counter()
    try:
        for s in store.series:
            for lo, end, stride in valid_ranges(s, c['length'], max(c['horizons']), 0, s['memory_end'], c['index']['stride']):
                count = (end - lo + stride - 1) // stride
                for offset in range(0, count, c['index']['shard_windows']):
                    n = min(c['index']['shard_windows'], count - offset)
                    sid = len(meta['shards']); path = out / f'{sid:06d}'; path.mkdir()
                    starts = lo + (offset + np.arange(n, dtype=np.int64)) * stride
                    np.save(path / 'starts.npy', starts)
                    arrays = {}; levels = {}
                    with torch.inference_mode():
                        for a in range(0, n, c['index']['batch_size']):
                            batch = starts[a:a+c['index']['batch_size']]
                            x = torch.from_numpy(np.stack([store.window(s['sid'], p, c['length']) for p in batch])).to(device)
                            result = model(x, torch.full((len(x),), scale_floor(s, c), device=device))
                            for channel in c['index']['channels']:
                                v = result[channel].cpu().numpy().astype('<f4')
                                if not np.isfinite(v).all():
                                    raise FloatingPointError('Nonfinite embedding')
                                if channel not in arrays:
                                    cp = path / channel; cp.mkdir()
                                    arrays[channel] = np.lib.format.open_memmap(cp / 'vectors.npy', mode='w+', dtype='<f4', shape=(n, v.shape[1]))
                                arrays[channel][a:a+len(v)] = v
                    for channel, v in arrays.items():
                        v.flush()
                        if meta['layout']=='spatial':
                            order=spatial_order(v,c['index']['leaf_size'])
                            shuffled=np.array(v[order],copy=True)
                            v[:]=shuffled; v.flush(); del shuffled
                            np.save(path/channel/'starts.npy',starts[order])
                        levels[channel] = boxes(path / channel, v, c['index']['leaf_size'], c['index']['fanout'])
                    meta['shards'].append(dict(path=path.name, sid=s['sid'], group=s['group'], n=n, levels=levels))
                    meta['windows'] += n
                    write_json(out / 'manifest.json', meta)
                    print(dict(shard=sid, windows=meta['windows']), flush=True)
    finally:
        store.close()
    if not meta['windows']:
        raise ValueError('No complete memory windows')
    meta.update(complete=True, build_seconds=time.perf_counter()-started,
                payload_bytes=sum(p.stat().st_size for p in out.rglob('*.npy')))
    write_json(out / 'manifest.json', meta)
    return meta


def lower_bound(q, lo, hi):
    delta = np.maximum(np.maximum(np.asarray(lo, dtype=np.float64)-q, q-np.asarray(hi, dtype=np.float64)), 0)
    value = np.sum(delta * delta, axis=-1)
    # Conservative float64 arithmetic slack plus outward rounded float32 boxes.
    return np.maximum(0, value - 1e-12 * (1 + value))


def diverse(hits, k, length):
    selected = []
    for hit in hits:
        if all(hit['sid'] != old['sid'] or abs(hit['start']-old['start']) >= length for old in selected):
            selected.append(hit)
            if len(selected) == k:
                break
    return selected


class Candidates:
    """Vectorized, bounded top-M; deterministic distance/sid/start tie breaking."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.distances = np.empty(0, dtype=np.float64)
        self.sids = np.empty(0, dtype=np.int64)
        self.starts = np.empty(0, dtype=np.int64)
        self.threshold = float('inf')

    def add(self, distances, sid, starts):
        keep = distances <= self.threshold
        if not np.any(keep): return
        ds = np.r_[self.distances, distances[keep]]
        ss = np.r_[self.sids, np.full(int(keep.sum()),sid,dtype=np.int64)]
        ps = np.r_[self.starts, starts[keep]]
        if len(ds)>self.capacity:
            partition = np.argpartition(ds,self.capacity-1)
            pivot = ds[partition[self.capacity-1]]
            strict = np.flatnonzero(ds<pivot)
            tied = np.flatnonzero(ds==pivot)
            tied = tied[np.lexsort((ps[tied],ss[tied]))[:self.capacity-len(strict)]]
            chosen = np.r_[strict,tied]
            ds,ss,ps = ds[chosen],ss[chosen],ps[chosen]
        self.distances,self.sids,self.starts = ds,ss,ps
        if len(ds)==self.capacity: self.threshold=float(ds.max())

    def hits(self):
        order=np.lexsort((self.starts,self.sids,self.distances))
        return [dict(distance=float(self.distances[i]),sid=int(self.sids[i]),start=int(self.starts[i])) for i in order]

    def __len__(self): return len(self.distances)


class Library:
    def __init__(self, path, cache_size=128):
        self.path = Path(path); self.meta = read_json(self.path / 'manifest.json')
        if self.meta.get('version') != 6 or not self.meta['complete']:
            raise ValueError('Incomplete or incompatible index')
        self.c = self.meta['config']; self.cache = OrderedDict(); self.cache_size = cache_size

    def array(self, shard, file):
        key = (shard, file)
        if key not in self.cache:
            self.cache[key] = np.load(self.path / self.meta['shards'][shard]['path'] / file, mmap_mode='r')
            # Do not forcibly close evicted arrays: callers may still reference them.
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def close(self):
        self.cache.clear()

    def search(self, query, channel='learned', k=10, sid=None, group=None, query_start=None, query_sid=None,
               leaf_budget=0, time_budget_ms=0, exclude_span=None, capacity=None, candidate_filter=None):
        """One pass with initial redundancy, no all-distance cache or adaptive rescans.

        An underfilled diverse result is explicitly marked incomplete. Increase
        oversample on validation if needed. Time budgets are soft, checked per node.
        """
        began = time.perf_counter(); q = np.asarray(query, dtype=np.float64)
        if q.ndim != 1 or not np.isfinite(q).all() or k < 1 or leaf_budget < 0 or time_budget_ms < 0:
            raise ValueError('Invalid query or search budget')
        if query_start is not None and query_sid is None and sid is None:
            raise ValueError('Provide query_sid to exclude the query in a cross-series search')
        if channel not in self.c['index']['channels']:
            raise ValueError('Channel was not built')
        capacity = max(k, int(capacity or k * self.c['index']['oversample']))
        length = self.c['length']; span = exclude_span or length + max(self.c['horizons'])
        fanout, leaf_size = self.c['index']['fanout'], self.c['index']['leaf_size']
        queue, best, leaves, visited, scored, eligible = [], Candidates(capacity), 0, 0, 0, 0
        for i, shard in enumerate(self.meta['shards']):
            if sid is not None and shard['sid'] != sid:
                continue
            if group is not None and shard['group'] != group:
                continue
            eligible += shard['n']
            level = len(shard['levels'][channel]) - 1
            lo = self.array(i, f'{channel}/lo{level}.npy')[0]
            hi = self.array(i, f'{channel}/hi{level}.npy')[0]
            if q.shape != lo.shape:
                raise ValueError('Query dimension mismatch')
            heapq.heappush(queue, (float(lower_bound(q, lo, hi)), i, level, 0))
        stopped = False
        while queue:
            if time_budget_ms and (time.perf_counter()-began)*1000 >= time_budget_ms:
                stopped = True; break
            bound, shard_id, level, node = heapq.heappop(queue)
            threshold = best.threshold
            if bound > threshold:
                break
            visited += 1
            shard = self.meta['shards'][shard_id]
            if level:
                a = node * fanout; b = min(a + fanout, shard['levels'][channel][level-1])
                bounds = lower_bound(q, self.array(shard_id, f'{channel}/lo{level-1}.npy')[a:b],
                                     self.array(shard_id, f'{channel}/hi{level-1}.npy')[a:b])
                for child, lb in enumerate(bounds, a):
                    if lb <= threshold:
                        heapq.heappush(queue, (float(lb), shard_id, level-1, child))
            else:
                if leaf_budget and leaves >= leaf_budget:
                    stopped = True; break
                leaves += 1
                a = node * leaf_size; b = min(a + leaf_size, shard['n'])
                starts_file = f'{channel}/starts.npy' if self.meta.get('layout') == 'spatial' else 'starts.npy'
                starts = self.array(shard_id, starts_file)[a:b]
                allowed = np.ones(len(starts), dtype=bool)
                excluded_sid = sid if query_sid is None else query_sid
                if query_start is not None and shard['sid'] == excluded_sid:
                    allowed &= np.abs(starts-query_start) >= span
                values = np.asarray(self.array(shard_id, f'{channel}/vectors.npy')[a:b], dtype=np.float64)
                ds = np.sum((values-q)**2, axis=1); scored += len(ds)
                best.add(ds[allowed],shard['sid'],starts[allowed])
        hits = best.hits()
        if candidate_filter is not None:
            hits=candidate_filter(hits)
        selected = diverse(hits, k, length)
        return selected, dict(search_ms=(time.perf_counter()-began)*1000, leaves=leaves, nodes=visited,
                              scored=scored, eligible=eligible, candidates_retained=len(best),
                              capacity=capacity, certified=not stopped, complete_topk=len(selected)==k,
                              exact_diverse_topk=not stopped and len(selected)==k,
                              returned=len(selected), leaf_budget=leaf_budget, time_budget_ms=time_budget_ms)
