"""Shared configuration and artifact identity checks."""
import hashlib
from pathlib import Path
from v5.common import read_json, write_json, fingerprint, environment, fresh_dir


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config(path):
    c = read_json(path)
    if c['version'] != 6 or c['length'] < 8:
        raise ValueError('Expected V6 config with length >= 8')
    if not c['horizons'] or min(c['horizons']) < 1:
        raise ValueError('Positive horizons required')
    s = c['split']
    if not 0 < s['memory'] < s['train'] < s['validation'] < 1:
        raise ValueError('Invalid chronological split')
    for key in ('hidden', 'dim', 'components', 'belief_bins', 'history_bins'):
        if c['model'][key] < 1:
            raise ValueError(key)
    for key in ('batch_size', 'prior_epochs', 'encoder_epochs', 'samples_per_series'):
        if c['training'][key] < 1:
            raise ValueError(key)
    if c['training']['batch_size'] < 3:
        raise ValueError('Contrastive training needs batch_size >= 3')
    for key in ('stride', 'leaf_size', 'shard_windows', 'batch_size'):
        if c['index'][key] < 1:
            raise ValueError(key)
    if c['index']['oversample'] < 1 or c['training']['sample_stride'] < 1 or c['data']['chunk_rows'] < 1:
        raise ValueError('Positive sampling/chunk/oversample settings required')
    if c['evaluation']['queries_per_series'] < 1 or c['evaluation']['bootstrap_samples'] < 1:
        raise ValueError('Positive evaluation sample counts required')
    if c['evaluation']['audit_queries'] < 0 or c['evaluation']['max_examples'] < 0:
        raise ValueError('Negative evaluation limit')
    if c['evaluation']['time_budget_ms'] < 0:
        raise ValueError('Negative time budget')
    if c['index']['fanout'] < 2 or not c['index']['channels']:
        raise ValueError('Invalid hierarchy')
    if set(c['index']['channels']) - {'learned', 'history', 'belief'}:
        raise ValueError('Unknown channel')
    if c['normalization']['memory_std_floor'] <= 0:
        raise ValueError('Positive memory scale floor required')
    if c['training']['temperature'] <= 0 or c['training']['target_temperature'] <= 0:
        raise ValueError('Positive temperatures required')
    if c['evaluation']['scope'] not in ('series', 'group', 'all'):
        raise ValueError('Unknown search scope')
    if c['evaluation']['k'] < 1 or not c['evaluation']['leaf_budgets'] or min(c['evaluation']['leaf_budgets']) < 0:
        raise ValueError('Invalid evaluation budgets')
    return c


def check_store(store, c):
    old = read_json(Path(store.path) / 'config.json')
    for key in ('length', 'horizons', 'split'):
        if old[key] != c[key]:
            raise ValueError(f'Store/config mismatch: {key}')


def save_torch(path, state):
    import torch
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)
