"""Deploy the explored adapter with a history-only probe and causal feedback."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .moirai import MoiraiAdapter,MODEL_ID,MODEL_REVISION
from .trend_rag import ResidualPool
from .error_correction import retrieve
from .past_error_adapter import feature,rerank,fit_completed
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,apply_policy


def scale_data(data):
    memory_std=data['std'].copy();data['std']=np.maximum(data['raw_x'].std(1),.1*memory_std)
    data['scale_ratio']=data['std']/memory_std
    if 'raw_y' in data:data['target']=(data['raw_y']-data['base'])/data['std'][:,None]
    return data


def predict(store_path,cache_path,folder,history,sid,start,device='cuda',use_rag=True):
    folder,cache=Path(folder),Path(cache_path);run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if sha256(cache/'manifest.json')!=run['source_manifest_sha256']:raise ValueError('Changed cache')
    if meta['model_identity']['model_id']!=MODEL_ID or meta['model_identity']['revision']!=MODEL_REVISION:raise ValueError('Wrong pinned model')
    for name,key in [('policy.json','policy_sha256'),('selection.json','selection_sha256'),('context_plan.json','context_plan_sha256')]:
        if sha256(folder/name)!=run[key]:raise ValueError('Changed checkpoint metadata')
    kind='with_rag' if use_rag else 'without_rag';name=f'past_{sid}_{kind}.npz'
    if sha256(folder/name)!=run['past_hashes'][name]:raise ValueError('Changed weights')
    with np.load(folder/name,allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
    plan=json.loads((folder/'context_plan.json').read_text(encoding='utf-8'))
    selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))[str(sid)][kind]
    policy=json.loads((folder/'policy.json').read_text(encoding='utf-8'))[kind];store=Store(store_path)
    try:
        if store.meta['data_id']!=run['source_data_id']:raise ValueError('Wrong store')
        s=store.series[sid]
        if start<s['validation_end']:raise ValueError('All policy validation outcomes must already be observed')
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        adapter=MoiraiAdapter.pretrained(meta['moirai'],device)
        q=adapter.forecast(x[None],96).cpu().numpy();probe_q=adapter.forecast(x[None,:64],32).cpu().numpy()
        error=x[None,64:]-probe_q[:,4]
        pool=ResidualPool(store,sid,meta['window'],stride=meta.get('memory_stride',1))
        _,m=retrieve(x,q[0],pool,start,meta['window'])
        path=cache/f'memory_errors_{sid}.npz'
        if sha256(path)!=meta['memory_files'][str(sid)]:raise ValueError('Changed memory')
        with np.load(path,allow_pickle=False) as f:
            memory_starts=f['starts'].copy();memory_errors=f['errors'].copy()
        np.testing.assert_array_equal(pool.starts,memory_starts)
        std=np.float32(max(s['memory_std'],1e-6));positions=np.searchsorted(memory_starts,m['selected_starts'])
        data=dict(raw_x=x[None],raw_quantiles=q,base=q[:,4],start=np.array([start]),sid=np.array([sid]),std=np.array([std]),
            errors=memory_errors[positions][None],**{k:m[k][None] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']})
        if run.get('annual_calendar'):data['annual_calendar']=True
        if run.get('adaptive_scale'):
            scale_data(data)
            old_std=np.array([store.window(sid,int(p),96).std() for p in memory_starts])
            memory_errors=memory_errors*(float(std)/np.maximum(old_std,.1*float(std)))[:,None]
        weight=selection['error_state_weight']
        ranked=rerank(data,[0],error,memory_starts,memory_errors,weight) if use_rag else None
        current=feature(store,data,[0],s,run['period'],plan,error,use_rag,ranked)
        feedback_count=0
        if run.get('online'):
            probes=Path(run['source_probes']);manifest=json.loads((probes/'manifest.json').read_text(encoding='utf-8'))
            if sha256(probes/'manifest.json')!=run['source_probe_manifest_sha256']:raise ValueError('Changed probes')
            xs=[];ys=[];ps=[]
            for split in ['train','validation','test']:
                old=load(cache,split);ids=np.flatnonzero((old['sid']==sid)&(old['start']+192<=start))
                if run.get('annual_calendar'):old['annual_calendar']=True
                if not len(ids):continue
                if sha256(probes/f'{split}.npz')!=manifest['files'][split]:raise ValueError('Changed past probes')
                with np.load(probes/f'{split}.npz',allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['sid'],old['sid']);np.testing.assert_array_equal(f['start'],old['start']);e=f['error'][ids]
                if run.get('adaptive_scale'):scale_data(old)
                ranked_old=rerank(old,ids,e,memory_starts,memory_errors,weight) if use_rag else None
                xs.append(feature(store,old,ids,s,run['period'],plan,e,use_rag,ranked_old))
                ys.append(old['target'][ids]);ps.append(old['start'][ids])
            feedback_count=sum(map(len,ps))
            state=fit_completed(np.concatenate(xs),np.concatenate(ys),np.concatenate(ps),start,selection['penalty'])
        scale=data['std'][:,None]
        delta=bound_delta(covariance(state,current)*scale,q)/scale
        final=q[:,4]+apply_policy(policy,{kind:delta},np.array([sid]))*scale
        return dict(prediction=final[0].tolist(),moirai_prediction=q[0,4].tolist(),selected_starts=(ranked or m)['selected_starts'].reshape(-1).tolist(),
            beta=policy[str(sid)]['beta'],family=policy[str(sid)]['family'],length=96,horizon=96,history_only_probe=True,
            feedback_episodes=feedback_count,adaptive_scale=run.get('adaptive_scale',False),online=run.get('online',False),foundation_frozen=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--cache',required=True);p.add_argument('--checkpoint-folder',required=True)
    p.add_argument('--history',required=True);p.add_argument('--sid',type=int,required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--without-rag',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    write_json(a.output,predict(a.store,a.cache,a.checkpoint_folder,np.load(a.history,allow_pickle=False),a.sid,a.start,a.device,not a.without_rag))
