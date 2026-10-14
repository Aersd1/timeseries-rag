"""Reconstruct calibration policies, paired predictions, and chronology checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .benchmark_component_rag import fit_policies,forecasts
from .benchmark_state_recall import metric,interval
from .query_component_rag import ComponentPredictor
from .query_future_state_rag import FutureStatePredictor


def verify(root,source,names):
    records=[]
    for name in names:
        folder=Path(root)/name;run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        contract=json.loads((folder/'contract.json').read_text(encoding='utf-8'));checkpoint=Path(run['source_checkpoint'])
        assert sha256(folder/'policy.json')==run['policy_sha256']==contract['policy_sha256']
        assert sha256(folder/'validation_trace.npz')==contract['validation_sha256']
        assert sha256(folder/'contract.json')==run['contract_sha256'];assert sha256(folder/'predictions.npz')==run['predictions_sha256']
        with np.load(folder/'validation_trace.npz',allow_pickle=False) as f:val={k:f[k].copy() for k in f.files}
        old=json.loads((checkpoint/'policy.json').read_text(encoding='utf-8'));policies=fit_policies(val,old)
        assert policies==json.loads((folder/'policy.json').read_text(encoding='utf-8'))
        with np.load(folder/'test_trace.npz',allow_pickle=False) as f:test={k:f[k].copy() for k in f.files}
        pred=forecasts(test,policies)
        with np.load(folder/'predictions.npz',allow_pickle=False) as f:
            for kind in pred:np.testing.assert_array_equal(pred[kind],f[kind]);assert metric(pred[kind],f)==run['metrics'][kind]
            assert interval(pred['segmented_no_rag'],pred['segmented_rag'],f)==run['rag_gain_interval']
        p=ComponentPredictor(Path(source)/name/'store',folder,adapter=False)
        try:
            # Actual deployment component calculation against cached full forecasts.
            data=load(p.cache,'test');probes=Path(p.base_run['source_probes'])
            with np.load(probes/'test.npz',allow_pickle=False) as f:probe=f['quantiles'].copy()
            ids=np.unique(np.array([0,len(data['sid'])//2,len(data['sid'])-1]))
            for i in ids:
                sid=int(data['sid'][i]);start=int(data['start'][i]);meta={k:data[k][i] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']}
                assert np.all(meta['candidate_starts']+192<=min(start,p.store.series[sid]['memory_end']))
                result=p.predict_cached(data['raw_x'][i],sid,start,data['raw_quantiles'][i:i+1],probe[i:i+1],meta)
                for kind in pred:np.testing.assert_allclose(result['forecasts'][kind],pred[kind][i],atol=2e-4,rtol=2e-5)
        finally:p.close()
        records.append(dict(dataset=name,policies_reconstructed=True,predictions_and_metrics_reconstructed=True,deployment_cases=len(ids)))
        print('VERIFIED COMPONENT',records[-1],flush=True)
    write_json(Path(root)/'verification.json',dict(complete=True,records=records))


def verify_reference(folder,source):
    folder=Path(folder);run=json.loads((folder/'run.json').read_text(encoding='utf-8'));contract=json.loads((folder/'contract.json').read_text(encoding='utf-8'))
    for name,key in [('contract.json','contract_sha256'),('predictions.npz','predictions_sha256'),('foundation.npz','foundation_sha256')]:assert sha256(folder/name)==run[key]
    assert sha256(run['references'])==contract['references_sha256']
    assert sha256(Path(run['component_checkpoint'])/'run.json')==contract['component_run_sha256']
    assert sha256(Path(run['resolution_checkpoint'])/'run.json')==contract['resolution_run_sha256']
    a=ComponentPredictor(Path(source)/'store',run['component_checkpoint'],adapter=False)
    b=FutureStatePredictor(Path(source)/'store',run['resolution_checkpoint'],adapter=False)
    try:
        with np.load(folder/'predictions.npz',allow_pickle=False) as f,np.load(folder/'foundation.npz',allow_pickle=False) as cached:
            np.testing.assert_array_equal(f['sid'],contract['sid']);np.testing.assert_array_equal(f['start'],contract['start'])
            for method,m in run['metrics'].items():assert metric(f[method],f)==m
            assert interval(f['routed'],f['segmented_rag'],f)==run['segment_vs_legacy_interval']
            assert interval(f['segmented_no_rag'],f['segmented_rag'],f)==run['segment_rag_gain_interval']
            for i,(sid,start) in enumerate(zip(f['sid'],f['start'])):
                sid,start=int(sid),int(start);np.testing.assert_array_equal(f['raw_y'][i],a.store.window(sid,start+96,96))
                for k in ['selected_starts','resolution_selected_starts']:assert np.all(f[k][i]+192<=min(start,a.store.series[sid]['memory_end']))
                if i in [0,len(f['sid'])//2,len(f['sid'])-1]:
                    history=a.store.window(sid,start,96);q=cached['quantiles'][i:i+1];probe=cached['probe'][i:i+1]
                    ra=a.predict_cached(history,sid,start,q,probe);rb=b.predict_cached(history,sid,start,q,probe)
                    for kind,value in ra['forecasts'].items():np.testing.assert_allclose(value,f[kind][i],atol=2e-4,rtol=2e-5)
                    for kind in ['encoder_only','future_state_rag','routed']:np.testing.assert_allclose(rb['forecasts'][kind],f[f'resolution_{kind}'][i],atol=2e-4,rtol=2e-5)
            report=dict(complete=True,queries=len(f['sid']),reference_cohort_verified=True,outcomes_and_metrics_verified=True,
                candidate_boundaries_verified=True,deployment_replays=3,caveat='Previously evaluated cohort, not blind')
        write_json(folder/'verification.json',report);print('VERIFIED REFERENCE',report,flush=True)
    finally:a.close();b.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--datasets',nargs='+');p.add_argument('--reference',action='store_true')
    a=p.parse_args()
    if a.reference:verify_reference(a.root,a.source)
    else:verify(a.root,a.source,a.datasets)
