"""Train a joint future-state encoder and test information beyond its own decoder."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .query_past_error_adapter import scale_data
from .past_error_adapter import feature
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,choose_policy,apply_policy
from .future_state_encoder import history_features,future_state,prototype_state,train,predict,train_resolution,decode
from .learned_rerank import diverse_selection
from .rerank import aggregate
from .complementary_rag import candidate_innovations
from .benchmark_state_recall import metric,interval

WEIGHTS=[0.,.25,1.,4.]


def retrieve_forecast(data,i,base,query_code,precision,starts,codes,past,future,library_starts,weight,channel_index=None):
    candidates=data['candidate_starts'][i];sid=int(data['sid'][i]);p=np.searchsorted(library_starts,candidates)
    np.testing.assert_array_equal(library_starts[p],candidates)
    if np.any(candidates+192>data['start'][i]):raise ValueError('Overlapping/self candidate')
    mask=codes['mask'][p];weighted=mask*precision
    distance=np.sum((codes['values'][p]-query_code)**2*weighted,axis=1)/np.maximum(weighted.sum(1),1e-6)
    fs=data['future_scores'][i];hs=data['history_scores'][i]
    score=.8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)+weight*distance/max(float(np.median(distance)),1e-6)
    hits=[dict(sid=sid,start=int(p),distance=float(h)) for p,h in zip(candidates,hs)]
    selected=diverse_selection(hits,score,5,192,1.)
    channel=sid if channel_index is None else int(channel_index)
    values=candidate_innovations(data['raw_x'][i],base,past[p[selected],:,channel],future[p[selected],:,channel],data['metric_std'][i])
    delta,w=aggregate(values,score[selected])
    return delta.astype(np.float32),candidates[selected],w


def experiment(checkpoint,complementary,source,out,optimize_resolution=False):
    out.mkdir(parents=True,exist_ok=False);run=json.loads((checkpoint/'run.json').read_text(encoding='utf-8'))
    if run.get('online'):raise ValueError('Frozen baseline required')
    cache=Path(run['source_cache']);probes=Path(run['source_probes']);meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    assert sha256(cache/'manifest.json')==run['source_manifest_sha256']
    assert sha256(probes/'manifest.json')==run['source_probe_manifest_sha256']
    plan=json.loads((checkpoint/'context_plan.json').read_text(encoding='utf-8'));policy_base=json.loads((checkpoint/'policy.json').read_text(encoding='utf-8'))['without_rag']
    store=Store(source/'store');assert store.meta['data_id']==run['source_data_id']
    if len(store.series)>32 or len({s['source'] for s in store.series})!=1:raise ValueError('This joint encoder pilot requires <=32 synchronous channels from one source')
    matrix=np.stack([np.array(store.values(s['sid']),dtype=np.float32) for s in store.series],axis=1)
    means=np.array([s['memory_mean'] for s in store.series]);stds=np.array([max(s['memory_std'],1e-6) for s in store.series]);channels=len(stds)
    def histories(starts):return np.stack([matrix[int(p):int(p)+96] for p in starts])
    def futures(starts):return np.stack([matrix[int(p)+96:int(p)+192] for p in starts])
    data={sp:load(cache,sp) for sp in ['train','validation','test']};probe={}
    for sp,current in data.items():
        current['metric_std']=current['std'].copy()
        if run.get('annual_calendar'):current['annual_calendar']=True
        if run.get('adaptive_scale'):scale_data(current)
        with np.load(probes/f'{sp}.npz',allow_pickle=False) as f:probe[sp]=f['error'].copy()
    unique=np.unique(data['train']['start']);unique=unique[np.array([np.isfinite(matrix[p:p+192]).all() for p in unique])]
    x=histories(unique);y=futures(unique);h=history_features(x,unique,means,stds,run['period'] or 1);code=future_state(x,y,stds)
    state,selection,held,oof=train_resolution(x,y,unique,means,stds,run['period'] or 1) if optimize_resolution else train(h,code,unique)
    width=int(state.get('block_width',16));history_width=int(state.get('history_width',16));np.savez(out/'future_encoder.npz',**state)
    lookup={int(unique[j]):oof[k] for k,j in enumerate(held)}
    memory_end=min(s['memory_end'] for s in store.series)
    library_starts=np.unique(np.concatenate([d['candidate_starts'].reshape(-1) for d in data.values()]))
    if np.any(library_starts+192>memory_end):raise ValueError('Unknown prototype futures')
    past=histories(library_starts);future=futures(library_starts);codes=prototype_state(past,future,means,stds,width)
    losses={str(w):[] for w in WEIGHTS};calibration_rows=0
    for s in store.series:
        sid=s['sid']
        with np.load(complementary/f'outer_train_{sid}.npz',allow_pickle=False) as f:rows=f['rows'];bases=f['base']
        for i,base in zip(rows,bases):
            p=int(data['train']['start'][i])
            if p not in lookup:continue
            target=(data['train']['raw_y'][i]-base)/data['train']['metric_std'][i];calibration_rows+=1
            for w in WEIGHTS:
                delta,_,_=retrieve_forecast(data['train'],int(i),base,lookup[p],state['precision'],None,codes,past,future,library_starts,w)
                bounded=bound_delta((delta*data['train']['metric_std'][i])[None],data['train']['raw_quantiles'][i:i+1])[0]/data['train']['metric_std'][i]
                losses[str(w)].append(float(np.mean((target-.25*bounded)**2)))
    if calibration_rows<32:raise ValueError('Insufficient shared forward calibration queries')
    weight=min(WEIGHTS,key=lambda w:(np.mean(losses[str(w)]),w))
    write_json(out/'selection.json',dict(encoder=selection,retrieval_weight=weight,calibration_queries=calibration_rows,
        retrieval_losses={k:float(np.mean(v)) for k,v in losses.items()},selected_before_validation_test=True))
    bases={};deltas={kind:{} for kind in ['encoder_only','future_state_rag']};traces={}
    try:
        for sp in ['validation','test']:
            current=data[sp];bases[sp]=np.zeros_like(current['base']);a=np.zeros_like(current['base']);b=np.zeros_like(current['base']);selected=np.zeros((len(a),5),dtype=np.int64)
            joint=histories(current['start']);predicted=predict(state,history_features(joint,current['start'],means,stds,run['period'] or 1,block_width=history_width))
            for s in store.series:
                sid=s['sid'];rows=np.flatnonzero(current['sid']==sid);beta=policy_base[str(sid)].get('proposed_beta',0.)
                with np.load(checkpoint/f'past_{sid}_without_rag.npz',allow_pickle=False) as f:base_state={k:f[k].copy() for k in f.files}
                f=feature(store,current,rows,s,run['period'],plan,probe[sp][rows],False)
                bases[sp][rows]=current['base'][rows]+beta*bound_delta(covariance(base_state,f)*current['std'][rows,None],current['raw_quantiles'][rows])
                for i in rows:
                    scale=max(float(current['raw_x'][i].std()),.1*stds[sid]);center=float(current['raw_x'][i,-16:].mean())
                    path=center+decode(predicted[i],channels,width)[:,sid]*scale
                    a[i]=(path-bases[sp][i])/current['metric_std'][i]
                    b[i],selected[i],_=retrieve_forecast(current,int(i),bases[sp][i],predicted[i],state['precision'],None,codes,past,future,library_starts,weight)
                print('JOINT FUTURE',checkpoint.name,sp,sid+1,'/',channels,flush=True)
            for kind,delta in [('encoder_only',a),('future_state_rag',b)]:
                deltas[kind][sp]=bound_delta(delta*current['metric_std'][:,None],current['raw_quantiles'])/current['metric_std'][:,None]
            traces[sp]=selected
        val=data['validation'];test=data['test'];target=(val['raw_y']-bases['validation'])/val['metric_std'][:,None]
        policies={kind:choose_policy(target,{kind:values['validation']},val['sid'],val['start'],require_mae=True) for kind,values in deltas.items()}
        experts={kind:dict(values) for kind,values in deltas.items()}
        for fraction in [.25,.5,.75]:
            experts[f'mix_{fraction}']={sp:(1-fraction)*deltas['encoder_only'][sp]+fraction*deltas['future_state_rag'][sp] for sp in ['validation','test']}
        policies['routed']=choose_policy(target,{k:v['validation'] for k,v in experts.items()},val['sid'],val['start'],require_mae=True)
        final_base=bases['test'].copy()
        for sid,p in policy_base.items():
            if p['beta']==0 and p.get('proposed_beta',0.)>0:
                final_base[test['sid']==int(sid)]=test['base'][test['sid']==int(sid)]
                for policy in policies.values():policy[sid].update(beta=0.,family='direct',reason='base expert rejected on final validation block')
        write_json(out/'policy.json',policies);forecasts=dict(moirai_direct=test['base'],no_rag=final_base)
        for kind,values in deltas.items():forecasts[kind]=final_base+apply_policy(policies[kind],{kind:values['test']},test['sid'])*test['metric_std'][:,None]
        forecasts['routed']=final_base+apply_policy(policies['routed'],{k:v['test'] for k,v in experts.items()},test['sid'])*test['metric_std'][:,None]
        truth={k:test[k] for k in ['raw_y','sid','start']};truth['std']=test['metric_std']
        np.savez_compressed(out/'predictions.npz',**forecasts,**truth,selected_starts=traces['test'])
        metrics={k:metric(v,truth) for k,v in forecasts.items()}
        result=dict(complete=True,optimize_resolution=optimize_resolution,dataset=checkpoint.name,metrics=metrics,queries=len(test['sid']),variables=channels,
            gain_vs_direct_percent=100*(1-metrics['future_state_rag']['nmse']/metrics['moirai_direct']['nmse']),
            rag_vs_no_rag_percent=100*(1-metrics['future_state_rag']['nmse']/metrics['no_rag']['nmse']),
            rag_vs_encoder_only_percent=100*(1-metrics['future_state_rag']['nmse']/metrics['encoder_only']['nmse']),
            routed_vs_encoder_only_percent=100*(1-metrics['routed']['nmse']/metrics['encoder_only']['nmse']),
            retrieval_weight=weight,source_checkpoint=str(checkpoint),source_complementary=str(complementary),
            selection_sha256=sha256(out/'selection.json'),encoder_sha256=sha256(out/'future_encoder.npz'),policy_sha256=sha256(out/'policy.json'),
            predictions_sha256=sha256(out/'predictions.npz'),gain_interval=interval(final_base,forecasts['future_state_rag'],truth),
            mechanism='Joint history predicts six16-point future state blocks of all variables; prototypes use known old futures; train-only precision and recall weight',
            caveat='Exploratory reused test periods; compare with the same learned future encoder without retrieval')
        write_json(out/'run.json',result);print('JOINT FUTURE RESULT',checkpoint.name,metrics,flush=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True);p.add_argument('--complementary-root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args()
    for name in a.datasets:experiment(Path(a.checkpoint_root)/name,Path(a.complementary_root)/name,Path(a.source)/name,Path(a.output)/name)
