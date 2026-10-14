"""Audit frozen policies, five-case chronology, paired forecasts and deployment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .benchmark_guarded_bank import table
from .guarded_bank_rag import choose_bank,apply_bank
from .benchmark_state_recall import metric,interval
from .benchmark_error_correction import load
from .query_guarded_bank import GuardedBankPredictor


def verify(root,source,names,device='cpu',live=False):
    records=[]
    for name in names:
        folder=Path(root)/name;run=json.loads((folder/'run.json').read_text());contract=json.loads((folder/'contract.json').read_text());direct=contract.get('direct',False)
        assert sha256(folder/'contract.json')==run['contract_sha256'];assert sha256(folder/'policy.json')==run['policy_sha256']==contract['policy_sha256']
        assert sha256(folder/'predictions.npz')==run['predictions_sha256']
        original=Path(contract['original']);assert sha256(original/'run.json')==contract['original_run_sha256']
        original_run=json.loads((original/'run.json').read_text());checkpoint=Path(original_run['source_checkpoint'])
        old_policy=json.loads((checkpoint/'policy.json').read_text());base_run=json.loads((Path(json.loads((checkpoint/'run.json').read_text())['source_checkpoint'])/'run.json').read_text())
        assert sha256(original/f"{'direct_' if direct else ''}validation_trace.npz")==contract['original_validation_sha256']
        branches={key:Path(value['path']) for key,value in contract['branch_runs'].items()}
        for key,value in contract['branch_runs'].items():
            path=Path(value['path']);assert sha256(path/'run.json')==value['run_sha256']
            assert sha256(path/f"{'direct_' if direct else ''}validation_trace.npz")==value['validation_sha256']
            if direct:assert sha256(path/'direct_manifest.json')==value['direct_manifest_sha256']
        val=table(original,'validation',direct);validation={k:table(v,'validation',direct) for k,v in branches.items()}
        rebuilt=choose_bank(validation,val,old_policy['routed'],old_policy['encoder_only'],direct=direct)
        assert rebuilt==json.loads((folder/'policy.json').read_text())
        test=table(original,'test',direct);testing={k:table(v,'test',direct) for k,v in branches.items()};pred,selected=apply_bank(testing,test,rebuilt)
        if direct:
            assert sha256(folder/'no_rag_policy.json')==run['no_rag_policy_sha256']
            no=choose_bank({original.parent.name:val},dict(val,routed=val['encoder_only']),old_policy['encoder_only'],old_policy['encoder_only'],direct=True,no_retrieval=True)
            assert no==json.loads((folder/'no_rag_policy.json').read_text())
            no_pred=apply_bank({original.parent.name:test},dict(test,routed=test['encoder_only']),no)[0]
        store=Store(Path(source)/name/'store')
        try:
            pairs=0
            for branch,path in branches.items():
                br=json.loads((path/'run.json').read_text())
                for split in ['validation','test']:
                    values=table(path,split,direct)
                    for s in store.series:
                        ids=values['sid']==s['sid'];limit=s['train_end'] if br.get('memory_scope')=='train' else s['memory_end']
                        assert np.all(values['selected_starts'][ids]+192<=np.minimum(values['start'][ids,None],limit));pairs+=int(ids.sum()*5)
            with np.load(folder/'predictions.npz',allow_pickle=False) as f:
                np.testing.assert_array_equal(pred,f['guarded_rag']);np.testing.assert_array_equal(selected,f['selected_starts'])
                if direct:np.testing.assert_array_equal(no_pred,f['guarded_no_rag'])
                for method,m in run['metrics'].items():assert metric(f[method],f)==m
                assert interval(f['routed'],f['guarded_rag'],f)==run['gain_interval']
                assert interval(f['guarded_no_rag'] if direct else f['encoder_only'],f['guarded_rag'],f)==run['rag_interval']
                for sid,start,y in zip(f['sid'],f['start'],f['raw_y']):np.testing.assert_array_equal(y,store.window(int(sid),int(start)+96,96))
        finally:store.close()
        predictor=GuardedBankPredictor(Path(source)/name/'store',folder,device,adapter=None if live else False)
        try:
            data=load(predictor.original.cache,'test');probes=Path(predictor.original.base_run['source_probes'])
            with np.load(probes/'test.npz',allow_pickle=False) as f:probe=f['quantiles'].copy()
            adopted=np.flatnonzero(np.isin(data['sid'],run['adopted_variables']));ids=np.unique(np.r_[0,adopted[0] if len(adopted) else len(data['sid'])//2,len(data['sid'])-1])
            with np.load(folder/'predictions.npz',allow_pickle=False) as expected:
                for i in ids:
                    sid=int(data['sid'][i]);start=int(data['start'][i])
                    result=predictor.predict(data['raw_x'][i],sid,start) if live else predictor.predict_cached(data['raw_x'][i],sid,start,data['raw_quantiles'][i:i+1],probe[i:i+1])
                    for k in ['guarded_rag','guarded_no_rag'] if direct else ['guarded_rag']:np.testing.assert_allclose(result['forecasts'][k],expected[k][i],rtol=2e-5,atol=2e-4)
                    np.testing.assert_array_equal(result['selected_starts'],expected['selected_starts'][i]);assert len(result['selected_starts'])==5
        finally:predictor.close()
        record=dict(dataset=name,policy_predictions_metrics_rebuilt=True,observed_candidate_pairs=pairs,deployment_cases=len(ids),actual_gpu=live)
        records.append(record);print('VERIFIED GUARDED BANK',record,flush=True)
    write_json(Path(root)/('gpu_verification.json' if live else 'verification.json'),dict(complete=True,records=records))


def verify_cohort(folder,source):
    folder=Path(folder);run=json.loads((folder/'run.json').read_text());contract=json.loads((folder/'contract.json').read_text())
    for name,key in [('contract.json','contract_sha256'),('predictions.npz','predictions_sha256'),('foundation.npz','foundation_sha256')]:assert sha256(folder/name)==run[key]
    assert sha256(run['references'])==contract['references_sha256'];assert sha256(Path(run['source_checkpoint'])/'run.json')==contract['source_run_sha256']
    p=GuardedBankPredictor(Path(source)/'store',run['source_checkpoint'],adapter=False)
    try:
        with np.load(folder/'predictions.npz',allow_pickle=False) as f,np.load(folder/'foundation.npz',allow_pickle=False) as q,np.load(run['references'],allow_pickle=False) as prior:
            np.testing.assert_array_equal(f['sid'],contract['sid']);np.testing.assert_array_equal(f['start'],contract['start'])
            if contract['offset']:
                for sid in np.unique(f['sid']):assert np.all(abs(f['start'][f['sid']==sid,None]-prior['start'][prior['sid']==sid][None,:])>=96)
            for i,(sid,start) in enumerate(zip(f['sid'],f['start'])):
                sid,start=int(sid),int(start);s=p.original.store.series[sid];np.testing.assert_array_equal(f['raw_y'][i],p.original.store.window(sid,start+96,96))
                policy=p.policy[str(sid)]
                limit=s['memory_end'] if policy['branch']=='incumbent' or policy['branch'].startswith('analogue_transport') else p._predictor(policy['branch']).memory_limit(s)
                assert np.all(f['selected_starts'][i]+192<=min(start,limit))
                if i in [0,len(f['sid'])//2,len(f['sid'])-1]:
                    result=p.predict_cached(p.original.store.window(sid,start,96),sid,start,q['quantiles'][i:i+1],q['probe'][i:i+1])
                    for k,v in result['forecasts'].items():np.testing.assert_allclose(v,f[k][i],rtol=2e-5,atol=2e-4)
            for method,m in run['metrics'].items():assert metric(f[method],f)==m
            assert interval(f['routed'],f['guarded_rag'],f)==run['gain_interval'];assert interval(f['guarded_no_rag'],f['guarded_rag'],f)==run['rag_interval']
            report=dict(complete=True,queries=len(f['sid']),outcomes_metrics_and_boundaries_verified=True,shifted_target_overlap=0 if contract['offset'] else None,deployment_replays=3)
        write_json(folder/'verification.json',report);print('VERIFIED GUARDED COHORT',report,flush=True)
    finally:p.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--datasets',nargs='+');p.add_argument('--live',action='store_true');p.add_argument('--cohort',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    if a.cohort:verify_cohort(a.root,a.source)
    else:verify(a.root,a.source,a.datasets,'cuda' if a.live else 'cpu',a.live)
