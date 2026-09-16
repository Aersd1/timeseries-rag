import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    return dict(python=platform.python_version(), platform=platform.platform(), git_commit=commit)


def affine(q, past, future):
    q, past, future = (np.asarray(v, dtype=np.float64) for v in (q, past, future))
    mx = past.mean(axis=-1); xc = past-mx[:, None]
    variance = np.mean(xc*xc, axis=-1)
    a = np.divide(np.mean(xc*(q-q.mean()), axis=-1), variance,
                  out=np.zeros(len(past)), where=variance > 1e-20)
    result = q.mean()+a[:, None]*(future-mx[:, None])
    result[variance <= 1e-20] = q[-1]
    return result


def norm_scale(q, memory_std):
    return max(float(np.std(q)), float(memory_std)*1e-4, 1e-6)


def weighted_analog(q, past, future, squared_distances):
    mapped = affine(q, past, future)
    ds = np.asarray(squared_distances)
    tau = max(float(np.median(ds)), 1e-8)
    w = np.exp(-(ds-ds.min())/tau); w /= w.sum()
    return w @ mapped


def valid_ranges(series, length, future, begin, end, stride=1):
    """All complete context+future windows in finite runs and one split."""
    if stride < 1: raise ValueError('stride must be positive')
    for a, b in series['runs']:
        lo, last = max(a, begin), min(b, end)-length-future
        if lo <= last:
            yield lo, last+1, stride


def sample_ranges(ranges, count, rng):
    ranges = list(ranges)
    counts = [(b-a+s-1)//s for a, b, s in ranges]
    edges = np.r_[0, np.cumsum(counts)]
    if not edges[-1]: return np.empty(0, dtype=np.int64)
    ids = rng.choice(int(edges[-1]), size=min(count, int(edges[-1])), replace=False)
    return np.sort(np.array([ranges[j][0]+(i-edges[j])*ranges[j][2]
                     for i in ids for j in [int(np.searchsorted(edges, i, side='right')-1)]], dtype=np.int64))


def validate_config(c):
    if c.get('version') != 5: raise ValueError('Expected config version 5')
    split = c['split']
    if not 0 < split['memory'] < split['train'] < split['validation'] < 1:
        raise ValueError('Require 0 < memory < train < validation < 1')
    m, hs = c['length'], c['horizons']
    if m < 34 or not hs or any(h < 1 for h in hs) or len(set(hs)) != len(hs):
        raise ValueError('Invalid context length or horizons')
    model = c['model']
    if model['primitive'] not in (4, 8) or not 2 <= model['vocab'] <= 65536:
        raise ValueError('primitive must be 4/8, vocabulary 2..65536')
    if c['teacher']['neighbors'] < 2 or c['teacher']['chunk_points'] < m+max(hs):
        raise ValueError('Invalid teacher neighbors/chunk size')
    if min(c['teacher']['queries_per_series'].values()) < 1:
        raise ValueError('Each split needs a positive sampling target')
    if any(v < 0 for v in c['loss_weights'].values()): raise ValueError('Negative loss weight')
    if any(c['temperatures'][k] <= 0 for k in ('shape', 'future', 'student', 'assignment')):
        raise ValueError('Temperatures must be positive')
    for k in ('dim','hidden'):
        if model[k] < 1: raise ValueError('Invalid model size')
    for k in ('epochs','batch_size','candidates'):
        if c['training'][k] < 1: raise ValueError('Invalid training count')
    if c['training']['candidates'] < 2: raise ValueError('Need at least two training candidates')
    if c['data']['chunk_rows'] < 1 or c['teacher']['shard_queries'] < 1:
        raise ValueError('Chunk sizes must be positive')
    if c['index']['stride'] < 1 or c['index']['batch_size'] < 1:
        raise ValueError('Invalid token index settings')
    if not 0 < c['evaluation']['candidate_fraction'] <= 1 or c['evaluation']['analog_k'] < 1:
        raise ValueError('Invalid evaluation budget')
    if not c['evaluation']['topk'] or min(c['evaluation']['topk']) < 1:
        raise ValueError('Need positive evaluation topk')
    return c
