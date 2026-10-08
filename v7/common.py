"""Configuration shared by the two explicitly separated experiments."""
from v6.common import config as v6_config, read_json, write_json, sha256, fresh_dir, check_store


def config(path):
    c = v6_config(path)  # Dataset/index format remains V6-compatible.
    if c.get('experiment') != 'moirai2_dual':
        raise ValueError('Expected a v7 Moirai experiment configuration')
    if c['evaluation']['scope'] != 'series':
        raise ValueError('These experiments enforce same-series chronological retrieval')
    if not c['model'].get('use_belief', True):
        raise ValueError('Variant A replaces the belief branch; use_belief must be true')
    if c['index']['layout'] != 'spatial':
        raise ValueError('Variant A builds a spatial layout')
    if c['index']['channels'] != ['learned', 'history', 'joint']:
        raise ValueError('A uses learned/history/joint; latent belief is a training target only')
    r = c['rerank']
    if r['candidates'] != 20 or not 1 <= r['top_k'] <= 20:
        raise ValueError('B retrieves 20 candidates; top_k must be in [1,20]')
    if r['selection'] not in ('ordinary', 'episodes') or r['score'] not in ('pinball', 'median_mse'):
        raise ValueError('Invalid rerank policy')
    if not 0 <= r['history_weight'] <= 1:
        raise ValueError('history_weight must be in [0,1]')
    if r['capacity'] < 20 or r['chunk_points'] < c['length']+max(c['horizons']):
        raise ValueError('Insufficient V4 capacity/chunk size')
    return c
