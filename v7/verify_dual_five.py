"""Audit policy, raw-only ranking, physical outcomes and actual query inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256, write_json
from .dual_five_rag import RawHistoryBank, choose_dual, apply_dual, FAMILIES, incumbent_reference
from .query_dual_five import DualFivePredictor
from .benchmark_error_correction import load
from .benchmark_state_recall import metric, interval


def read(path):
    with np.load(path, allow_pickle=False) as f: return {k:f[k].copy() for k in f.files}


def verify(root, source, names, live=False):
    records = []
    for name in names:
        folder = Path(root)/name
        run = json.loads((folder/'run.json').read_text())
        contract = json.loads((folder/'contract.json').read_text())
        for file, key in [('policy.json','policy_sha256'), ('contract.json','contract_sha256'),
            ('predictions.npz','predictions_sha256'), ('test_trace.npz','test_trace_sha256')]:
            assert sha256(folder/file)==run[key]
        assert sha256(folder/'validation_trace.npz')==contract['source_validation_sha256']
        val = read(folder/'validation_trace.npz'); test = read(folder/'test_trace.npz')
        policy = choose_dual(val)
        assert policy==json.loads((folder/'policy.json').read_text())
        forecasts = read(folder/'predictions.npz')
        np.testing.assert_array_equal(apply_dual(test, policy), forecasts['guarded_dual'])
        for family in FAMILIES:
            beta = 1. if family=='dual_trust' else .25
            base = test['incumbent'] if incumbent_reference(family) else test['moirai_direct']
            np.testing.assert_array_equal(base+beta*(test[family]-base), forecasts['fixed_'+family])
        for key, value in run['metrics'].items(): assert metric(forecasts[key],forecasts)==value
        assert interval(forecasts['previous_five'],forecasts['guarded_dual'],forecasts)==run['gain_interval']
        store = Store(Path(source)/name/'store')
        pairs = 0
        try:
            for values in [val, test]:
                for sid in np.unique(values['sid']):
                    sid = int(sid); rows = np.flatnonzero(values['sid']==sid)
                    end = int(contract['limits'][str(sid)])
                    bank = RawHistoryBank(store, sid, end, contract['memory_stride'], contract['raw_case_spacing'])
                    for row in rows:
                        start = int(values['start'][row]); selected = values['selected_starts'][row]
                        assert len(np.unique(selected))==10
                        assert np.all(selected+192<=min(start,end))
                        history = store.window(sid,start,96)
                        raw, distance = bank.select(history,start,selected[:5])
                        np.testing.assert_array_equal(raw,selected[5:])
                        np.testing.assert_allclose(distance,values['history_distances'][row],rtol=1e-12,atol=1e-12)
                        np.testing.assert_array_equal(values['raw_y'][row],store.window(sid,start+96,96))
                        pairs += 10
        finally: store.close()
        predictor = DualFivePredictor(Path(source)/name/'store',folder,'cuda' if live else 'cpu',adapter=None if live else False)
        try:
            p = predictor.previous.original
            data = load(p.cache,'test'); probes = Path(p.base_run['source_probes'])
            manifest = json.loads((probes/'manifest.json').read_text())
            assert sha256(probes/'manifest.json')==p.base_run['source_probe_manifest_sha256']
            assert sha256(probes/'test.npz')==manifest['files']['test']
            probe = read(probes/'test.npz')['quantiles']
            adopted = np.flatnonzero(np.isin(data['sid'],run['adopted_variables']))
            ids = np.unique([0,int(adopted[0]) if len(adopted) else len(data['sid'])//2,len(data['sid'])-1])
            for row in ids:
                sid, start = int(data['sid'][row]),int(data['start'][row])
                output = predictor.predict(data['raw_x'][row],sid,start) if live else predictor.predict_cached(
                    data['raw_x'][row],sid,start,data['raw_quantiles'][row:row+1],probe[row:row+1])
                for key in ['guarded_dual']+['fixed_'+family for family in FAMILIES]:
                    np.testing.assert_allclose(output['forecasts'][key],forecasts[key][row],rtol=2e-5,atol=2e-4)
                np.testing.assert_array_equal(output['selected_starts'],forecasts['selected_starts'][row])
                assert output['retrieved_cases']==10 and not output['query_future_used']
        finally: predictor.close()
        record = dict(dataset=name,old_case_pairs=pairs,queries=len(test['sid']),raw_rank_and_targets_rebuilt=True,
            policy_and_metrics_rebuilt=True,deployed_queries=len(ids),actual_gpu=live)
        records.append(record); print('VERIFIED DUAL',record,flush=True)
    write_json(Path(root)/('gpu_verification.json' if live else 'verification.json'),dict(complete=True,records=records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    parser.add_argument('--source',default='v7/runs/all_datasets_20261008')
    parser.add_argument('--datasets',nargs='+',required=True); parser.add_argument('--live',action='store_true')
    args=parser.parse_args(); verify(args.root,args.source,args.datasets,args.live)
