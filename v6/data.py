"""Reuse the CSV parser; separate prior fit, encoder fit, validation and test."""
import copy
import tempfile
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from v5.data import Store, ingest as ingest_v5
from v5.common import valid_ranges, sample_ranges
from .common import config, read_json, write_json, fingerprint, check_store


def ingest(config_path, output):
    c = config(config_path)
    # Adapter affects parser settings only; no V5 source changes or training.
    base = read_json(Path(__file__).parents[1] / 'v5/configs/server.json')
    for key in ('length', 'horizons', 'data', 'split', 'seed'):
        base[key] = copy.deepcopy(c[key])
    with tempfile.TemporaryDirectory() as tmp:
        adapter = Path(tmp) / 'adapter.json'
        write_json(adapter, base)
        meta = ingest_v5(adapter, output)
    meta.update(version=6, config_hash=fingerprint(c))
    meta.pop('data_id')
    meta['data_id'] = fingerprint(meta)
    write_json(Path(output) / 'config.json', c)
    write_json(Path(output) / 'catalog.json', meta)
    return meta


def scale_floor(series, c):
    return max(series['memory_std'] * c['normalization']['memory_std_floor'], 1e-6)


class Windows(Dataset):
    def __init__(self, store_path, c, split):
        self.store = Store(store_path)
        check_store(self.store, c)
        self.c, self.split, self.refs = c, split, []
        m, h = c['length'], max(c['horizons'])
        rng = np.random.default_rng(c['seed'] + {'prior': 0, 'train': 1, 'validation': 2, 'test': 3}[split])
        count = c['training']['samples_per_series'] if split in ('prior', 'train') else c['evaluation']['queries_per_series']
        for s in self.store.series:
            begin, end = {'prior': (0, s['memory_end']), 'train': (s['memory_end'], s['train_end']),
                          'validation': (s['train_end'], s['validation_end']),
                          'test': (s['validation_end'], s['n'])}[split]
            # Evaluation queries have disjoint context+future spans.
            stride = c['training']['sample_stride'] if split in ('prior', 'train') else m + h
            starts = sample_ranges(valid_ranges(s, m, h, begin, end, stride), count, rng)
            self.refs.extend((s['sid'], int(p)) for p in starts)
        if not self.refs:
            raise ValueError(f'No complete {split} windows; adjust data size/splits')

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        sid, start = self.refs[i]
        m = self.c['length']
        return dict(x=self.store.window(sid, start, m),
                    y=self.store.window(sid, start + m, max(self.c['horizons'])),
                    floor=np.float32(scale_floor(self.store.series[sid], self.c)),
                    sid=np.int64(sid), start=np.int64(start))
