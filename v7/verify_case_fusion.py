"""Audit neural training provenance, every old episode and actual deployment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .aligned_case_fusion import MODES,inputs,tensor_inputs,forecast
from .benchmark_case_fusion import load_trace,calibrate,apply,parts,fixed_alignment
from .benchmark_state_recall import metric,interval
from .query_case_fusion import LearnedCasePredictor
from .dual_five_rag import RawHistoryBank


def verify(root,names,device='cpu',live=False):
    records=[]
    for name in names:
        folder=Path(root)/name;run=json.loads((folder/'run.json').read_text());contract=json.loads((folder/'contract.json').read_text())
        meta=json.loads((folder/'preparation.json').read_text());selection=json.loads((folder/'selection.json').read_text())
        count=meta.get('case_count',10)
        for filename,key in [('contract.json','contract_sha256'),('predictions.npz','predictions_sha256'),('seed_predictions.npz','seed_predictions_sha256')]:
            assert sha256(folder/filename)==run[key]
        for filename,key in [('preparation.json','preparation_sha256'),('stats.json','stats_sha256'),('selection.json','selection_sha256'),('policy.json','policy_sha256'),('validation_forecasts.npz','validation_predictions_sha256')]:
            assert sha256(folder/filename)==contract[key]
        values={split:load_trace(folder,split,run['test_info']['sha256'] if split=='test' else meta['files'][split]['sha256']) for split in ['train','validation','test']}
        store=Store(meta['store']);case_pairs=0
        try:
            for split,data in values.items():
                for series in store.series:
                    sid=series['sid'];rows=np.flatnonzero(data['sid']==sid);selected=data['selected_starts'][rows];start=data['start'][rows]
                    limit=int(meta['limits'][str(sid)])
                    assert np.all(selected+192<=np.minimum(start[:,None],limit))
                    assert all(len(np.unique(v))==count for v in selected)
                    if count==10:assert np.all(np.diff(np.sort(selected[:,5:],axis=1),axis=1)>=192)
                    if split=='train':assert np.all(start+192<=series['train_end'])
                    else:assert np.all(start>=series['train_end'] if split=='validation' else start>=series['validation_end'])
                    raw=store.values(sid)
                    np.testing.assert_array_equal(data['case_past'][rows],raw[selected[:,:,None]+np.arange(96)[None,None]])
                    np.testing.assert_array_equal(data['case_future'][rows],raw[selected[:,:,None]+96+np.arange(96)[None,None]])
                    np.testing.assert_array_equal(data['raw_x'][rows],raw[start[:,None]+np.arange(96)])
                    np.testing.assert_array_equal(data['raw_y'][rows],raw[start[:,None]+96+np.arange(96)])
                    if split=='train' and count==10:
                        bank=RawHistoryBank(store,sid,limit,
                            json.loads((Path(meta['source'])/'contract.json').read_text())['memory_stride'],192)
                        for row in rows:
                            raw_starts,_=bank.select(data['raw_x'][row],int(data['start'][row]),data['selected_starts'][row,:5])
                            np.testing.assert_array_equal(raw_starts,data['selected_starts'][row,5:])
                    case_pairs+=len(rows)*count
            for series in store.series:
                train=values['train'];val=values['validation'];sid=series['sid']
                assert np.max(train['start'][train['sid']==sid]+192)<=np.min(val['start'][val['sid']==sid])
        finally:store.close()
        predictor=LearnedCasePredictor(meta['store'],folder,device,adapter=None if live else False)
        try:
            val=values['validation'];test=values['test'];policy=json.loads((folder/'policy.json').read_text())
            with np.load(folder/'validation_forecasts.npz',allow_pickle=False) as vf,np.load(folder/'seed_predictions.npz',allow_pickle=False) as sf,np.load(folder/'predictions.npz',allow_pickle=False) as saved:
                for mode,models in predictor.models.items():
                    np.testing.assert_array_equal(np.mean([vf[f'{mode}_{seed}'] for seed in selection['settings']['seeds']],axis=0).astype(np.float32),vf[mode])
                    assert calibrate(val,vf[mode])==policy[mode]
                    raw=np.mean([sf[f'{mode}_{seed}'] for seed in selection['settings']['seeds']],axis=0).astype(np.float32)
                    np.testing.assert_array_equal(raw,saved['raw_'+mode]);np.testing.assert_array_equal(apply(policy[mode],test,raw),saved['calibrated_'+mode])
                    for seed in selection['settings']['seeds']:assert metric(sf[f'{mode}_{seed}'],test)==run['individual_seed_metrics'][f'{mode}_{seed}']
                    features,_,scale=inputs(val,predictor.stats,predictor.period,mode)
                    rebuilt=forecast(models[0],tensor_inputs(features,device),val,scale)
                    np.testing.assert_allclose(rebuilt,vf[f'{mode}_{selection["settings"]["seeds"][0]}'],rtol=2e-5,atol=2e-4)
                if count==5 and not meta.get('residual_reference'):
                    with np.load(Path(meta['source'])/'validation_trace.npz',allow_pickle=False) as old:
                        incumbent_val=old['incumbent'].copy()
                    for mode in MODES:
                        proposal=incumbent_val+vf[mode]-val['raw_quantiles'][:,4]
                        assert calibrate(val,proposal,incumbent_val)==policy['increment_'+mode]
                        proposal=saved['previous_previous_five']+saved['raw_'+mode]-saved['moirai_direct']
                        np.testing.assert_array_equal(apply(policy['increment_'+mode],test,proposal,saved['previous_previous_five']),saved['increment_'+mode])
                    proposal=incumbent_val+vf['wiener']-vf['without_retrieval']
                    assert calibrate(val,proposal,incumbent_val)==policy['increment_difference']
                for key,value in fixed_alignment(test,predictor.stats).items():np.testing.assert_array_equal(value,saved[key])
                for key,metrics in run['metrics'].items():assert metric(saved[key],saved)==metrics
                assert interval(saved['moirai_direct'],saved['calibrated_wiener'],saved)==run['vs_moirai_interval']
                assert interval(saved['calibrated_without_retrieval'],saved['calibrated_wiener'],saved)==run['rag_interval']
                p=predictor.original;probes=Path(p.base_run['source_probes'])
                manifest=json.loads((probes/'manifest.json').read_text())
                assert sha256(probes/'manifest.json')==p.base_run['source_probe_manifest_sha256'];assert sha256(probes/'test.npz')==manifest['files']['test']
                with np.load(probes/'test.npz',allow_pickle=False) as f:probe=f['quantiles'].copy()
                active=np.flatnonzero(np.isin(test['sid'],run['active_variables']['wiener']))
                ids=np.unique([0,int(active[0]) if len(active) else len(test['sid'])//2,len(test['sid'])-1])
                for row in ids:
                    sid,start=int(test['sid'][row]),int(test['start'][row])
                    output=predictor.predict(test['raw_x'][row],sid,start) if live else predictor.predict_cached(test['raw_x'][row],sid,start,test['raw_quantiles'][row:row+1],probe[row:row+1])
                    for mode in MODES:
                        for prefix in ['raw_','calibrated_']:np.testing.assert_allclose(output['forecasts'][prefix+mode],saved[prefix+mode][row],rtol=2e-5,atol=2e-4)
                        if count==5 and not meta.get('residual_reference'):np.testing.assert_allclose(output['forecasts']['increment_'+mode],saved['increment_'+mode][row],rtol=2e-5,atol=2e-4)
                    np.testing.assert_array_equal(output['selected_starts'],test['selected_starts'][row])
                    assert not output['query_future_used']
                # Poison current and sibling outcomes in memory, then replay the
                # entire actual retrieval+adapter path with identical forecast cache.
                row=int(ids[0]);sid,start=int(test['sid'][row]),int(test['start'][row])
                before=predictor.predict_cached(test['raw_x'][row],sid,start,test['raw_quantiles'][row:row+1],probe[row:row+1])
                bank_predictor=predictor.previous if count==5 else predictor.previous.previous
                for branch in bank_predictor.branches.values():
                    for series_values in branch.synchronous_values:series_values[start+96:start+192]=1e15
                after=predictor.predict_cached(test['raw_x'][row],sid,start,test['raw_quantiles'][row:row+1],probe[row:row+1])
                assert before==after
        finally:predictor.close()
        record=dict(dataset=name,queries=len(test['sid']),old_case_pairs=case_pairs,
            all_physical_sources_and_training_boundaries=True,raw_history_train_selection_rebuilt=count==10,
            policies_predictions_metrics_rebuilt=True,deployed_queries=len(ids),current_and_sibling_future_poison_invariant=True,actual_gpu=live)
        records.append(record);print('VERIFIED LEARNED CASE',record,flush=True)
    write_json(Path(root)/('gpu_verification.json' if live else 'verification.json'),dict(complete=True,records=records))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--datasets',nargs='+',required=True);parser.add_argument('--live',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4);verify(args.root,args.datasets,'cuda' if args.live else 'cpu',args.live)
