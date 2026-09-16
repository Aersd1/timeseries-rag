"""Streaming CSV ingestion with relative paths and finite-run boundaries."""
import glob
import hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from .common import fresh_dir, read_json, write_json, fingerprint, validate_config


class Store:
    def __init__(self, path, cache_size=16):
        self.path = Path(path)
        self.meta = read_json(self.path/'catalog.json')
        if not self.meta['complete']: raise ValueError('Incomplete store')
        self.series = self.meta['series']; self.cache = OrderedDict(); self.cache_size = cache_size

    def values(self, sid):
        sid = int(sid)
        if sid not in self.cache:
            self.cache[sid] = np.memmap(self.path/self.series[sid]['file'], mode='r', dtype='<f4')
            if len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        self.cache.move_to_end(sid)
        return self.cache[sid]

    def window(self, sid, start, length):
        values = np.array(self.values(sid)[int(start):int(start)+length], dtype=np.float32)
        if len(values) != length or not np.isfinite(values).all():
            raise ValueError('Invalid/gapped window reference')
        return values

    def close(self):
        for values in self.cache.values(): values._mmap.close()
        self.cache.clear()


def csv_chunks(path, **kwargs):
    with pd.read_csv(path, **kwargs) as reader:
        yield from reader


def finite_runs(x, chunk=1_000_000):
    runs, began = [], None
    for offset in range(0, len(x), chunk):
        finite = np.isfinite(x[offset:offset+chunk])
        changes = np.flatnonzero(np.r_[True, finite[1:] != finite[:-1]])
        for j in changes:
            pos = offset+int(j)
            if finite[j] and began is None: began = pos
            elif not finite[j] and began is not None:
                runs.append([began, pos]); began = None
    if began is not None: runs.append([began, len(x)])
    return runs


def ingest(config_path, output):
    c = validate_config(read_json(config_path)); out = fresh_dir(output)
    (out/'values').mkdir(); series, seen, digests = [], set(), {}
    for spec in c['data']['sources']:
        paths = sorted(glob.glob(spec['glob'], recursive=True))
        if not paths: raise ValueError(f"No CSV files matched {spec['glob']}")
        for path in paths:
            path = str(Path(path).resolve())
            columns = spec['columns']; device_col = spec.get('device_column'); time_col = spec.get('time_column')
            usecols = list(dict.fromkeys(columns+([device_col] if device_col else [])+([time_col] if time_col else [])))
            ids, last_time = {}, {}
            for frame in csv_chunks(path, usecols=usecols, chunksize=c['data']['chunk_rows'],
                                     dtype={device_col:'string'} if device_col else None,
                                     encoding=spec.get('encoding', 'utf-8'), sep=spec.get('delimiter', ',')):
                if device_col and frame[device_col].isna().any(): raise ValueError('Missing device ID')
                groups = frame.groupby(device_col, sort=False) if device_col else [(None, frame)]
                for device, part in groups:
                    device = None if device is None else str(device)
                    if time_col:
                        if spec.get('time_kind', 'datetime') == 'numeric':
                            t = pd.to_numeric(part[time_col], errors='raise').to_numpy(dtype=float)
                        else:
                            dt = pd.to_datetime(part[time_col], errors='raise', utc=True)
                            if dt.isna().any(): raise ValueError('Missing timestamp')
                            t = dt.astype('int64').to_numpy(dtype=float)/1e9
                        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or (device in last_time and t[0] <= last_time[device]):
                            raise ValueError('Timestamps must increase per device; sort/deduplicate upstream')
                        if 'expected_interval' in spec:
                            diff = np.diff(np.r_[last_time[device], t]) if device in last_time else np.diff(t)
                            if not np.allclose(diff, spec['expected_interval'], rtol=1e-6, atol=1e-6):
                                raise ValueError('Timestamp gap: resample or split the file before ingest')
                        last_time[device] = t[-1]
                    for col in columns:
                        key = (path, col, device)
                        if key not in ids:
                            if key in seen: raise ValueError('Duplicate file/column/device in config')
                            seen.add(key); sid = len(series); ids[key] = sid
                            digests[sid] = hashlib.sha256()
                            series.append(dict(sid=sid, file=f'values/{sid:06d}.bin', source=path,
                                               column=col, device=device, group=spec.get('group', 'custom'), n=0))
                        sid = ids[key]
                        x = pd.to_numeric(part[col], errors='coerce').to_numpy(dtype=np.float32)
                        x = x.astype('<f4')
                        with (out/series[sid]['file']).open('ab') as f: x.tofile(f)
                        digests[sid].update(x.tobytes())
                        series[sid]['n'] += len(x)
    for s in series:
        s['values_sha256'] = digests[s['sid']].hexdigest()
        x = np.memmap(out/s['file'], mode='r', dtype='<f4')
        s['runs'] = finite_runs(x)
        s['memory_end'] = int(s['n']*c['split']['memory'])
        s['train_end'] = int(s['n']*c['split']['train'])
        s['validation_end'] = int(s['n']*c['split']['validation'])
        # Stable streaming moments on MEMORY only, no train-query/test labels.
        total, mean, m2 = 0, 0., 0.
        for a in range(0, s['memory_end'], c['data']['chunk_rows']):
            v = np.asarray(x[a:min(s['memory_end'], a+c['data']['chunk_rows'])], dtype=float)
            v = v[np.isfinite(v)]
            if not len(v): continue
            n = len(v); mu = float(v.mean()); diff = mu-mean
            m2 += float(np.sum((v-mu)**2))+diff*diff*total*n/(total+n)
            mean += diff*n/(total+n); total += n
        s.update(memory_mean=mean, memory_std=float(np.sqrt(m2/max(total, 1))))
        del x
    catalog = dict(version=5, complete=True, dtype='<f4', series=series,
                   config_hash=fingerprint(c), total_points=sum(s['n'] for s in series))
    catalog['data_id'] = fingerprint(catalog)
    write_json(out/'catalog.json', catalog); write_json(out/'config.json', c)
    return catalog
