"""Paired 5 versus raw-history 5+5, with policy frozen before TEST is read."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256, write_json
from .query_future_state_rag import FutureStatePredictor
from .benchmark_guarded_bank import table
from .guarded_bank_rag import apply_bank
from .benchmark_error_correction import load
from .benchmark_state_recall import metric, interval
from .dual_five_rag import RawHistoryBank, encoded_cases, fusion, choose_dual, apply_dual, FAMILIES, incumbent_reference


def collect(p, folder, split, spacing=192):
    contract = json.loads((folder/'contract.json').read_text())
    policy = json.loads((folder/'policy.json').read_text())
    original = table(Path(contract['original']), split, True)
    branches = {}
    for key, cfg in contract['branch_runs'].items():
        path = Path(cfg['path'])
        if sha256(path/'run.json')!=cfg['run_sha256'] or sha256(path/'direct_manifest.json')!=cfg['direct_manifest_sha256']:
            raise ValueError('Changed branch metadata')
        manifest = json.loads((path/'direct_manifest.json').read_text())
        if sha256(path/f'direct_{split}_trace.npz')!=manifest['files'][split]['direct_trace_sha256']:
            raise ValueError('Changed branch trace')
        branches[key] = table(path, split, True)
    incumbent, selected = apply_bank(branches, original, policy)
    data = load(p.cache, split)
    for key in ['sid', 'start', 'raw_y']:
        np.testing.assert_array_equal(original[key], data[key])
    all_selected = np.zeros((len(selected), 10), dtype=np.int64)
    distances = np.zeros_like(selected, dtype=float)
    outputs = {k: np.zeros_like(incumbent) for k in FAMILIES}
    support = np.zeros(len(selected)); agreement = support.copy(); limits = {}
    for sid in np.unique(data['sid']):
        sid = int(sid); rows = np.flatnonzero(data['sid']==sid)
        choice = policy[str(sid)]; scope = 'prefix'
        if choice['branch']!='incumbent':
            cfg = contract['branch_runs'][choice['branch']]
            branch_run = json.loads((Path(cfg['path'])/'run.json').read_text())
            scope = branch_run.get('memory_scope', 'prefix')
        series = p.store.series[sid]
        end = int(series['train_end'] if scope=='train' else series['memory_end'])
        limits[str(sid)] = end
        bank = RawHistoryBank(p.store, sid, end, p.meta.get('memory_stride', 1), spacing)
        for row in rows:
            raw, distance = bank.select(data['raw_x'][row], int(data['start'][row]), selected[row])
            all_selected[row] = np.r_[selected[row], raw]
            distances[row] = distance
        if np.any(all_selected[rows]+192>np.minimum(data['start'][rows, None], end)):
            raise ValueError('Self/overlapping or not fully observed auxiliary episode')
        encoded, copied, innovation = encoded_cases(p, sid, data['raw_x'][rows], data['start'][rows], all_selected[rows])
        paths, state = fusion(encoded, copied, innovation, data['raw_quantiles'][rows],
            data['raw_x'][rows], data['std'][rows], distances[rows], incumbent[rows])
        for name, value in paths.items(): outputs[name][rows] = value
        support[rows] = state['support']; agreement[rows] = state['agreement']
        if sid%100==0: print('DUAL TRACE', folder.name, split, sid, flush=True)
    return dict(**outputs, incumbent=incumbent, moirai_direct=data['base'],
        **{k:data[k] for k in ['sid', 'start', 'raw_y', 'std']}, selected_starts=all_selected,
        history_distances=distances, support=support, agreement=agreement), limits


def experiment(source, dataset, out, spacing=192):
    out.mkdir(parents=True, exist_ok=False)
    source_run = json.loads((source/'run.json').read_text())
    if sha256(source/'policy.json')!=source_run['policy_sha256'] or sha256(source/'contract.json')!=source_run['contract_sha256']:
        raise ValueError('Changed source policy')
    contract = json.loads((source/'contract.json').read_text())
    original_run = json.loads((Path(contract['original'])/'run.json').read_text())
    checkpoint = Path(original_run['source_checkpoint'])
    p = FutureStatePredictor(dataset/'store', checkpoint, adapter=False)
    try:
        validation, limits = collect(p, source, 'validation', spacing)
        np.savez_compressed(out/'validation_trace.npz', **validation)
        policy = choose_dual(validation); write_json(out/'policy.json', policy)
        write_json(out/'contract.json', dict(source=str(source), source_run_sha256=sha256(source/'run.json'),
            checkpoint=str(checkpoint), limits=limits, memory_stride=p.meta.get('memory_stride', 1),
            raw_case_spacing=spacing,
            source_validation_sha256=sha256(out/'validation_trace.npz'), policy_sha256=sha256(out/'policy.json'),
            source_contract_sha256=sha256(source/'contract.json'), raw_selection='Raw96 MSE, tie by start; exclude first five; future-free',
            blend='Five routed + five raw-history cases; equal case weights',
            rule='First two VAL blocks select; third confirms/rejects; TEST loaded only after policy is saved'))
        test, test_limits = collect(p, source, 'test', spacing)
        if limits!=test_limits: raise ValueError('Changed frozen bank')
        np.savez_compressed(out/'test_trace.npz', **test)
        if sha256(source/'predictions.npz')!=source_run['predictions_sha256']:
            raise ValueError('Changed original paired forecasts')
        with np.load(source/'predictions.npz', allow_pickle=False) as f:
            np.testing.assert_allclose(test['incumbent'], f['guarded_rag'], rtol=2e-5, atol=2e-4)
            control = f['guarded_no_rag'].copy()
        forecasts = dict(moirai_direct=test['moirai_direct'], previous_five=test['incumbent'], no_rag=control,
            guarded_dual=apply_dual(test, policy))
        for family in FAMILIES:
            beta = 1. if family=='dual_trust' else .25
            base = test['incumbent'] if incumbent_reference(family) else test['moirai_direct']
            forecasts['fixed_'+family] = base+beta*(test[family]-base)
        truth = {k:test[k] for k in ['sid', 'start', 'std', 'raw_y']}
        np.savez_compressed(out/'predictions.npz', **forecasts, **truth, selected_starts=test['selected_starts'])
        result = dict(complete=True, dataset=source.name, queries=len(test['sid']), metrics={k:metric(v, truth) for k,v in forecasts.items()},
            adopted_variables=[int(k) for k,v in policy.items() if v['family']!='incumbent'],
            policy_sha256=sha256(out/'policy.json'), contract_sha256=sha256(out/'contract.json'),
            predictions_sha256=sha256(out/'predictions.npz'), test_trace_sha256=sha256(out/'test_trace.npz'),
            gain_interval=interval(test['incumbent'], forecasts['guarded_dual'], truth),
            caveat='Development periods reused; ten retrieved does not mean ten used when validation rejects; no new Moirai fine-tuning')
        write_json(out/'run.json', result)
        print('DUAL RESULT', source.name, result['metrics'], result['adopted_variables'], flush=True)
    finally: p.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--data', default='v7/runs/all_datasets_20261008')
    parser.add_argument('--datasets', nargs='+', required=True); args = parser.parse_args()
    for name in args.datasets:
        experiment(Path(args.source)/name, Path(args.data)/name, Path(args.output)/name)
