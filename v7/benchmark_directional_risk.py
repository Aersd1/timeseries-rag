"""Freeze all risk policies on VAL before opening any TEST in this experiment."""
import argparse
import json
from pathlib import Path
import numpy as np

from .common import sha256, write_json
from .current_forecast import DATASETS, RUN_ROOT, PRIMARY
from .directional_risk import inputs, fit, weights, blend
from .benchmark_guarded_bank import table
from .guarded_bank_rag import apply_bank
from .selective_rag import chronological_split, rows
from .benchmark_state_recall import metric, interval

FEATURE_ROOT = Path('v7/runs/learned_case_fusion_96_20261014')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def npz(path):
    with np.load(path, allow_pickle=False) as f: return {k:f[k].copy() for k in f.files}


def load(name, split):
    folder = RUN_ROOT/name; run = read(folder/'run.json'); contract = read(folder/'contract.json')
    feature = FEATURE_ROOT/name; meta = read(feature/'preparation.json')
    if sha256(folder/'contract.json') != run['contract_sha256']:
        raise ValueError('Changed restored contract')
    if sha256(folder/'run.json') != meta['source_run_sha256']:
        raise ValueError('Changed frozen source')
    feature_hash = (read(feature/'run.json')['test_info']['sha256'] if split == 'test'
                    else meta['files'][split]['sha256'])
    if sha256(feature/f'{split}.npz') != feature_hash:
        raise ValueError('Changed actual case inputs')
    expected = contract['source_validation_sha256'] if split == 'validation' else run['test_trace_sha256']
    if sha256(folder/f'{split}_trace.npz') != expected: raise ValueError('Changed frozen trace')
    d = npz(feature/f'{split}.npz'); t = npz(folder/f'{split}_trace.npz')
    for key in ('sid', 'start', 'raw_y', 'selected_starts'):
        np.testing.assert_array_equal(d[key], t[key])
    limits = np.array([contract['limits'][str(int(s))] for s in d['sid']])
    if np.any(d['selected_starts']+192 > np.minimum(d['start'], limits)[:, None]):
        raise ValueError('Self/overlapping/future case')
    restored = t['incumbent']+.25*(t['increment_transport']-t['incumbent'])
    source = Path(contract['source']); c = read(source/'contract.json'); source_run = read(source/'run.json')
    if sha256(source/'contract.json') != contract['source_contract_sha256']:
        raise ValueError('Changed no-retrieval source contract')
    if sha256(source/'run.json') != contract['source_run_sha256']:
        raise ValueError('Changed no-retrieval source run')
    policy_path = source/'no_rag_policy.json'
    if sha256(policy_path) != source_run['no_rag_policy_sha256']:
        raise ValueError('Changed no-retrieval policy')
    original = Path(c['original'])
    manifest = read(original/'direct_manifest.json')
    if sha256(original/f'direct_{split}_trace.npz') != manifest['files'][split]['direct_trace_sha256']:
        raise ValueError('Changed original no-retrieval trace')
    v = table(original, split, True)
    for k in ('sid', 'start', 'raw_y'): np.testing.assert_array_equal(v[k], d[k])
    no_rag = apply_bank({original.parent.name:v}, dict(v, routed=v['encoder_only']), read(policy_path))[0]
    if split == 'test':
        if sha256(folder/'predictions.npz') != run['predictions_sha256']: raise ValueError('Changed saved forecasts')
        saved = npz(folder/'predictions.npz')
        np.testing.assert_array_equal(restored, saved[PRIMARY])
        np.testing.assert_allclose(no_rag, saved['no_rag'], rtol=0, atol=0)
    if sha256(feature/'stats.json') != meta['stats_sha256']: raise ValueError('Changed TRAIN statistics')
    return d, no_rag, restored, read(feature/'stats.json')


def metrics(pred, data, stats):
    m = metric(pred, data)
    sd = np.maximum(np.asarray(stats['std'])[data['sid']], 1e-6)
    m['train_nmse'] = float(np.mean(((pred-data['raw_y'])/sd[:, None])**2))
    return m


def safe(candidate, reference):
    return all(candidate[k] <= reference[k]*(1+1e-10) for k in ('mse', 'mae', 'train_nmse'))


def options(model, x, energy, no_rag, restored, foundation):
    result = {'moirai': (foundation, np.zeros(len(x))), 'no_rag': (no_rag, np.zeros(len(x)))}
    for a in (.25, .5, .75, 1.):
        alpha = np.full(len(x), a); result[f'static_{a}'] = (blend(no_rag, restored, alpha), alpha)
    for risk in (0., .5, 1., 2.):
        alpha = weights(model, x, energy, risk)
        result[f'dynamic_{risk}'] = (blend(no_rag, restored, alpha), alpha)
    return result


def choose(candidates, d, stats, cal, confirm, restored, foundation):
    refs = {k:metrics(v[cal], rows(d, cal), stats) for k, v in [('restored', restored), ('moirai', foundation)]}
    accepted = []
    for key, (forecast, _) in candidates.items():
        m = metrics(forecast[cal], rows(d, cal), stats)
        if all(safe(m, r) for r in refs.values()) and m['mse'] < refs['restored']['mse']*(1-1e-5):
            accepted.append((m['mse']/max(refs['restored']['mse'], 1e-20)+
                             m['train_nmse']/max(refs['restored']['train_nmse'], 1e-20), key))
    if not accepted: return dict(selected='restored', proposed=None, confirmed=False)
    _, key = min(accepted); forecast = candidates[key][0]
    cm = metrics(forecast[confirm], rows(d, confirm), stats)
    held_refs = {k:metrics(v[confirm], rows(d, confirm), stats) for k, v in [('restored', restored), ('moirai', foundation)]}
    ok = all(safe(cm, r) for r in held_refs.values()) and cm['mse'] < held_refs['restored']['mse']*(1-1e-5)
    return dict(selected=key if ok else 'restored', proposed=key, confirmed=bool(ok),
                confirmation=cm, confirmation_references=held_refs)


def fit_dataset(name, out):
    d, n, r, stats = load(name, 'validation')
    meta = read(FEATURE_ROOT/name/'preparation.json')
    catalog = read(Path(meta['store'])/'catalog.json')
    fit_ids, cal, confirm = chronological_split(d, max(s['train_end'] for s in catalog['series']),
                                               min(s['validation_end'] for s in catalog['series']))
    out.mkdir(parents=True, exist_ok=False); policies = {}; model_hashes = {}
    for arm in ('full', 'query_only'):
        x, scale, energy = inputs(d, n, r, stats, arm == 'query_only')
        model = fit(x[fit_ids], d['raw_y'][fit_ids], n[fit_ids], r[fit_ids], scale[fit_ids], d['start'][fit_ids])
        candidates = options(model, x, energy, n, r, d['raw_quantiles'][:, 4])
        policies[arm] = choose(candidates, d, stats, cal, confirm, r, d['raw_quantiles'][:, 4])
        np.savez_compressed(out/f'{arm}.npz', **model); model_hashes[arm] = sha256(out/f'{arm}.npz')
    write_json(out/'policy.json', policies)
    write_json(out/'contract.json', dict(dataset=name, primary='full', model_hashes=model_hashes,
        policy_sha256=sha256(out/'policy.json'), validation_sha256=sha256(FEATURE_ROOT/name/'validation.npz'),
        restored_run_sha256=sha256(RUN_ROOT/name/'run.json'),
        split={k:dict(rows=len(ids), first_start=int(d['start'][ids].min()),
                      last_label_end=int(d['start'][ids].max()+192)) for k, ids in [('fit',fit_ids),('cal',cal),('confirm',confirm)]},
        rule='First VAL block fits risk; second selects one policy; third confirms or rejects; rejection preserves restored 5+5',
        caveats=['Previously explored TEST', 'Upstream choices reused VAL: not independent full-pipeline confirmation',
                 'Bootstrap spread is model sensitivity, not calibrated predictive confidence',
                 'query_only controls extra gate case inputs, not all retrieval in the frozen R forecast']))
    print('RISK FROZEN', name, {k:v['selected'] for k,v in policies.items()}, flush=True)


def evaluate(name, out):
    contract = read(out/'contract.json')
    assert sha256(out/'policy.json') == contract['policy_sha256']
    assert sha256(RUN_ROOT/name/'run.json') == contract['restored_run_sha256']
    d, n, r, stats = load(name, 'test'); foundation = d['raw_quantiles'][:, 4]
    forecasts = dict(moirai=foundation, no_rag=n, restored=r); diagnostics = {}
    for arm, policy in read(out/'policy.json').items():
        assert sha256(out/f'{arm}.npz') == contract['model_hashes'][arm]
        model = npz(out/f'{arm}.npz')
        x, scale, energy = inputs(d, n, r, stats, arm == 'query_only')
        poisoned = dict(d, raw_y=np.full_like(d['raw_y'], 1e12))
        np.testing.assert_array_equal(x, inputs(poisoned, n, r, stats, arm == 'query_only')[0])
        candidates = options(model, x, energy, n, r, foundation)
        key = policy['selected']
        forecast, alpha = (r, np.ones(len(r))) if key == 'restored' else candidates[key]
        forecasts[arm] = forecast
        forecasts[arm+'_raw_dynamic'] = candidates['dynamic_1.0'][0]
        diagnostics[arm] = dict(policy=key, mean_alpha=float(alpha.mean()),
                                closed_fraction=float(np.mean(alpha == 0)),
                                full_fraction=float(np.mean(alpha == 1)))
    result = dict(dataset=name, queries=len(r), metrics={k:metrics(v, d, stats) for k,v in forecasts.items()},
        diagnostics=diagnostics, gain_vs_restored_interval=interval(r, forecasts['full'], d),
        gain_vs_no_rag_interval=interval(n, forecasts['full'], d),
        primary_gain_vs_moirai_percent=float(100*(1-np.mean((forecasts['full']-d['raw_y'])**2)/np.mean((foundation-d['raw_y'])**2))))
    np.savez_compressed(out/'predictions.npz', **forecasts, sid=d['sid'], start=d['start'])
    write_json(out/'result.json', result)
    print('RISK RESULT', name, result['primary_gain_vs_moirai_percent'], diagnostics, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--evaluate-only', action='store_true', help='Replay already frozen policies without fitting')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS)); a = p.parse_args()
    root = Path(a.output)
    if a.evaluate_only:
        frozen = read(root/'frozen.json')
        for name in a.datasets:
            if sha256(root/name/'contract.json') != frozen[name]: raise ValueError('Changed frozen policy contract')
    else:
        if root.exists(): raise FileExistsError('Choose a fresh experiment directory')
        root.mkdir(parents=True)
        for name in a.datasets: fit_dataset(name, root/name)
        # All dataset policies exist before any new TEST is opened.
        write_json(root/'frozen.json', {name:sha256(root/name/'contract.json') for name in a.datasets})
    results = [evaluate(name, root/name) for name in a.datasets]
    write_json(root/'summary.json', dict(complete=True, primary='full', results=results,
        note='Experimental only; current forecast remains restored fixed-increment 5+5'))


if __name__ == '__main__': main()
