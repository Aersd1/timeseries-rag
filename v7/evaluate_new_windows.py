"""Frozen checkpoint evaluation on windows disjoint from inspected test episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from v5.common import valid_ranges
from .common import sha256,write_json
from .benchmark_error_correction import load
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool
from .error_correction import retrieve
from .past_error_adapter import feature,rerank
from .query_past_error_adapter import scale_data
from .benefit_rerank import covariance
from .safe_error_policy import apply_policy,bound_delta
from .benchmark_state_recall import metric,interval


def evaluate(folder,source,out,count,device):
    out.mkdir(parents=True,exist_ok=False);run=json.loads((folder/'run.json').read_text(encoding='utf-8'));cache=Path(run['source_cache'])
    if run.get('online'):raise ValueError('Fresh-window evaluation requires a frozen head')
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    assert sha256(cache/'manifest.json')==run['source_manifest_sha256']
    for name,key in [('policy.json','policy_sha256'),('selection.json','selection_sha256'),('context_plan.json','context_plan_sha256')]:assert sha256(folder/name)==run[key]
    policy=json.loads((folder/'policy.json').read_text(encoding='utf-8'));selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
    plan=json.loads((folder/'context_plan.json').read_text(encoding='utf-8'));old=load(cache,'test');store=Store(source/'store')
    assert store.meta['data_id']==run['source_data_id'];records=[];references={}
    try:
        for s in store.series:
            sid=s['sid'];ranges=valid_ranges(s,96,96,s['validation_end'],s['n'],192)
            starts=np.array([p for a,b,step in ranges for p in range(a,b,step)],dtype=np.int64)
            prior=old['start'][old['sid']==sid]
            starts=starts[np.all(abs(starts[:,None]-prior[None,:])>=192,axis=1)]
            if len(starts)>count:starts=starts[np.linspace(0,len(starts)-1,count,dtype=int)]
            if not len(starts):raise ValueError('No disjoint new windows')
            references[str(sid)]=starts.tolist()
        # Freeze references and the selected checkpoint hashes before seeing new outcomes.
        write_json(out/'evaluation_contract.json',dict(checkpoint_run_sha256=sha256(folder/'run.json'),references=references,
            rule='Full192-point new episodes are pairwise disjoint and do not overlap any original test episode',
            caveat='New windows share the already inspected calendar period and are not a new year/dataset blind test'))
        adapter=MoiraiAdapter.pretrained(meta['moirai'],device)
        for s in store.series:
            sid=s['sid'];starts=np.asarray(references[str(sid)],dtype=np.int64);n=len(starts)
            x=np.stack([store.window(sid,p,96) for p in starts]);q=adapter.forecast(x,96).cpu().numpy()
            probe=adapter.forecast(x[:,:64],32).cpu().numpy();error=x[:,64:]-probe[:,4]
            pool=ResidualPool(store,sid,meta['window'],stride=meta.get('memory_stride',1))
            memories=[retrieve(h,z,pool,int(p),meta['window'])[1] for h,z,p in zip(x,q,starts)]
            path=cache/f'memory_errors_{sid}.npz';assert sha256(path)==meta['memory_files'][str(sid)]
            with np.load(path,allow_pickle=False) as f:ms=f['starts'].copy();me=f['errors'].copy()
            data=dict(raw_x=x,raw_quantiles=q,base=q[:,4],std=np.full(n,max(s['memory_std'],1e-6),dtype=np.float32),sid=np.full(n,sid),start=starts)
            data.update({k:np.stack([m[k] for m in memories]) for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']})
            data['errors']=me[np.searchsorted(ms,data['selected_starts'])]
            assert np.all(data['candidate_starts']+192<=starts[:,None])
            metric_std=data['std'].copy()
            if run.get('annual_calendar'):data['annual_calendar']=True
            if run.get('adaptive_scale'):
                scale_data(data);sd=np.array([store.window(sid,int(p),96).std() for p in ms])
                me=me*(max(s['memory_std'],1e-6)/np.maximum(sd,.1*max(s['memory_std'],1e-6)))[:,None]
            preds={'moirai_direct':q[:,4]}
            for kind in policy:
                name=f'past_{sid}_{kind}.npz';assert sha256(folder/name)==run['past_hashes'][name]
                with np.load(folder/name,allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
                r=rerank(data,np.arange(n),error,ms,me,selection[str(sid)][kind]['error_state_weight']) if kind=='with_rag' else None
                features=feature(store,data,np.arange(n),s,run['period'],plan,error,kind=='with_rag',r)
                delta=bound_delta(covariance(state,features)*data['std'][:,None],q)/data['std'][:,None]
                preds['past_'+kind]=q[:,4]+apply_policy(policy[kind],{kind:delta},data['sid'])*data['std'][:,None]
            # Outcomes are fetched only after both candidate forecasts have been fixed.
            records.append(dict(**preds,raw_y=np.stack([store.window(sid,p+96,96) for p in starts]),std=metric_std,sid=data['sid'],start=starts))
            if sid%40==0 or sid==len(store.series)-1:print('NEW WINDOWS',folder.name,sid+1,'/',len(store.series),flush=True)
        table={k:np.concatenate([r[k] for r in records]) for k in records[0]};np.savez_compressed(out/'predictions.npz',**table)
        metrics={k:metric(table[k],table) for k in ['moirai_direct','past_with_rag','past_without_rag']}
        result=dict(complete=True,dataset=folder.name,queries=len(table['sid']),variables=len(store.series),metrics=metrics,
            gain_vs_direct_percent=100*(1-metrics['past_with_rag']['nmse']/metrics['moirai_direct']['nmse']),
            rag_gain_vs_ablation_percent=100*(1-metrics['past_with_rag']['nmse']/metrics['past_without_rag']['nmse']),
            gain_interval=interval(table['moirai_direct'],table['past_with_rag'],table),
            contract_sha256=sha256(out/'evaluation_contract.json'),predictions_sha256=sha256(out/'predictions.npz'),
            source_checkpoint=str(folder),frozen_before_outcomes=True,overlap_with_original_test=0,
            caveat='New windows in the same historical period; checkpoint was developed using other windows in that period')
        write_json(out/'run.json',result);print('NEW WINDOW RESULT',result,flush=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-folder',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=24);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    evaluate(Path(a.checkpoint_folder),Path(a.source),Path(a.output),a.count,a.device)
