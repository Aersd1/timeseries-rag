"""Variant A shares V6 geometry, raw-history gate and analog aggregation."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from v5.common import valid_ranges
from v6.index import boxes, spatial_order, Library
from v6.inference import Retriever
from v6.data import scale_floor
from .common import fresh_dir, write_json, sha256, check_store
from .model import load


def build(store_path, checkpoint, output, device='cpu'):
    model, state = load(checkpoint, device)
    c = state['config']
    store = Store(store_path)
    check_store(store, c)
    if store.meta['data_id'] != state['data_id']:
        raise ValueError('Checkpoint/store mismatch')
    out = fresh_dir(output)
    meta = dict(version=6, encoder_family='moirai2_v7a', complete=False, data_id=state['data_id'],
        checkpoint_sha256=sha256(checkpoint), config=c, shards=[], windows=0,
        raw_points=store.meta['total_points'], layout='spatial')
    write_json(out/'manifest.json', meta)
    began = time.perf_counter()
    try:
        for s in store.series:
            for lo, end, stride in valid_ranges(s, c['length'], max(c['horizons']), 0, s['memory_end'], c['index']['stride']):
                count = (end-lo+stride-1)//stride
                for offset in range(0, count, c['index']['shard_windows']):
                    n = min(c['index']['shard_windows'], count-offset)
                    path = out/f'{len(meta["shards"]):06d}'; path.mkdir()
                    starts = lo+(offset+np.arange(n, dtype=np.int64))*stride
                    np.save(path/'starts.npy', starts)
                    arrays = {}
                    with torch.inference_mode():
                        for a in range(0, n, c['index']['batch_size']):
                            positions = starts[a:a+c['index']['batch_size']]
                            x = torch.from_numpy(np.stack([store.window(s['sid'], p, c['length']) for p in positions])).to(device)
                            encoded = model(x, torch.full((len(x),), scale_floor(s, c), device=device))
                            for channel in c['index']['channels']:
                                values = encoded[channel].cpu().numpy().astype('<f4')
                                if not np.isfinite(values).all():
                                    raise FloatingPointError('Nonfinite index vector')
                                if channel not in arrays:
                                    (path/channel).mkdir()
                                    arrays[channel] = np.lib.format.open_memmap(path/channel/'vectors.npy',
                                        mode='w+', dtype='<f4', shape=(n, values.shape[1]))
                                arrays[channel][a:a+len(values)] = values
                    levels = {}
                    for channel, values in arrays.items():
                        order = spatial_order(values, c['index']['leaf_size'])
                        values[:] = np.array(values[order], copy=True); values.flush()
                        np.save(path/channel/'starts.npy', starts[order])
                        levels[channel] = boxes(path/channel, values, c['index']['leaf_size'], c['index']['fanout'])
                    meta['shards'].append(dict(path=path.name, sid=s['sid'], group=s['group'], n=n, levels=levels))
                    meta['windows'] += n
                    write_json(out/'manifest.json', meta)
                    print(dict(windows=meta['windows'], shard=path.name), flush=True)
                    arrays.clear()
        if not meta['windows']:
            raise ValueError('No complete memory windows')
        meta.update(complete=True, build_seconds=time.perf_counter()-began,
            payload_bytes=sum(p.stat().st_size for p in out.rglob('*.npy')))
        write_json(out/'manifest.json', meta)
        return meta
    finally:
        store.close()


class MoiraiRetriever(Retriever):
    """Use inherited V6 encode/search/analog without its Gaussian-only evaluation."""
    def __init__(self, store, checkpoint, index, device='cpu'):
        self.model, self.state = load(checkpoint, device)
        self.c, self.device = copy.deepcopy(self.state['config']), device
        self.store, self.library = Store(store), Library(index)
        check_store(self.store, self.c)
        if self.library.meta.get('encoder_family') != 'moirai2_v7a':
            raise ValueError('Rebuild the index with variant A; old V6 geometry cannot be reused')
        if self.store.meta['data_id'] != self.state['data_id'] or self.library.meta['data_id'] != self.state['data_id']:
            raise ValueError('Store/checkpoint/index identity mismatch')
        if self.library.meta['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Index built from different weights')
        if self.library.c != self.c:
            raise ValueError('Index policy differs from the A checkpoint')
        self.index_sha = sha256(Path(index)/'manifest.json')
        self.fusion = None

    def retrieve(self, x, sid, start=None, channel=None, leaf_budget=0):
        if not 0 <= sid < len(self.store.series):
            raise ValueError('Unknown sid')
        if start is None or start < self.store.series[sid]['memory_end']:
            raise ValueError('Provide a query start at/after the fixed memory prefix')
        if channel not in (None, 'learned', 'history', 'joint'):
            raise ValueError('Unsupported A retrieval channel')
        return super().retrieve(x, sid, start, channel, leaf_budget)
