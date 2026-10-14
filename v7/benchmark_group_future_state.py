"""Scale future-state encoders to many variables through prefix-defined groups."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .contextual_adapter import context_plan
from .query_past_error_adapter import scale_data
from .past_error_adapter import feature
from .complementary_rag import forward_base
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,choose_policy,apply_policy
from .future_state_encoder import history_features,future_state,prototype_state,train,predict,train_resolution,decode
from .benchmark_future_state_rag import retrieve_forecast,WEIGHTS
from .benchmark_state_recall import metric,interval


def experiment(checkpoint,source,out,optimize_resolution=False):
    out.mkdir(parents=True,exist_ok=False);run=json.loads((checkpoint/'run.json').read_text(encoding='utf-8'))
    if run.get('online'):raise ValueError('Frozen base required')
    cache=Path(run['source_cache']);probes=Path(run['source_probes']);store=Store(source/'store')
    assert store.meta['data_id']==run['source_data_id'];assert sha256(cache/'manifest.json')==run['source_manifest_sha256']
    assert sha256(probes/'manifest.json')==run['source_probe_manifest_sha256']
    if len({(s['source'],s['n'],s['memory_end']) for s in store.series})!=1:raise ValueError('Synchronous same-source channels required')
    plan=json.loads((checkpoint/'context_plan.json').read_text(encoding='utf-8'));base_policy=json.loads((checkpoint/'policy.json').read_text(encoding='utf-8'))['without_rag']
    neighbors=context_plan(store,7);groups={str(s['sid']):[s['sid']]+neighbors[str(s['sid'])] for s in store.series}
    write_json(out/'groups.json',groups)
    matrix=np.stack([np.array(store.values(s['sid']),dtype=np.float32) for s in store.series],axis=1)
    memory_mean=np.array([s['memory_mean'] for s in store.series]);memory_std=np.array([max(s['memory_std'],1e-6) for s in store.series])
    data={sp:load(cache,sp) for sp in ['train','validation','test']};probe={}
    for sp,current in data.items():
        current['metric_std']=current['std'].copy()
        if run.get('annual_calendar'):current['annual_calendar']=True
        if run.get('adaptive_scale'):scale_data(current)
        with np.load(probes/f'{sp}.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['sid'],current['sid']);np.testing.assert_array_equal(f['start'],current['start']);probe[sp]=f['error'].copy()
    bases={sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']}
    deltas={kind:{sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']} for kind in ['encoder_only','future_state_rag']}
    selections={};selected_starts=np.zeros((len(data['test']['sid']),5),dtype=np.int64)
    try:
        for s in store.series:
            sid=s['sid'];group=np.array(groups[str(sid)]);local=matrix[:,group];means=memory_mean[group];stds=memory_std[group];period=run['period'] or 1
            group_rows=np.flatnonzero(np.isin(data['train']['sid'],group));unique=np.unique(data['train']['start'][group_rows])
            def episodes(starts):return local[np.asarray(starts)[:,None]+np.arange(192)[None,:]]
            ep=episodes(unique);valid=np.isfinite(ep).all((1,2));unique=unique[valid];ep=ep[valid]
            h=history_features(ep[:,:96],unique,means,stds,period,weekly=True);labels=future_state(ep[:,:96],ep[:,96:],stds)
            state,selection,held,oof=train_resolution(ep[:,:96],ep[:,96:],unique,means,stds,period,weekly=True) if optimize_resolution else train(h,labels,unique)
            width=int(state.get('block_width',16));history_width=int(state.get('history_width',16));np.savez(out/f'encoder_{sid}.npz',**state)
            lookup={int(unique[j]):oof[k] for k,j in enumerate(held)}
            ids=np.flatnonzero(data['train']['sid']==sid);beta=base_policy[str(sid)].get('proposed_beta',0.)
            name=f'past_{sid}_without_rag.npz';assert sha256(checkpoint/name)==run['past_hashes'][name]
            with np.load(checkpoint/name,allow_pickle=False) as f:base_state={k:f[k].copy() for k in f.files}
            x=feature(store,data['train'],ids,s,period,plan,probe['train'][ids],False)
            outer_rows,outer_base,base_folds=forward_base(x,data['train']['target'][ids],data['train']['start'][ids],data['train']['base'][ids],data['train']['raw_quantiles'][ids],data['train']['std'][ids],beta)
            all_rows={sp:np.flatnonzero(d['sid']==sid) for sp,d in data.items()}
            library=np.unique(np.concatenate([d['candidate_starts'][all_rows[sp]].reshape(-1) for sp,d in data.items()]))
            if np.any(library+192>s['memory_end']):raise ValueError('Unknown prototype future')
            old=episodes(library);past=old[:,:96];future=old[:,96:];codes=prototype_state(past,future,means,stds,width)
            losses={str(w):[] for w in WEIGHTS};count=0
            for row,base in zip(ids[outer_rows],outer_base):
                p=int(data['train']['start'][row])
                if p not in lookup:continue
                count+=1;target=(data['train']['raw_y'][row]-base)/data['train']['metric_std'][row]
                for w in WEIGHTS:
                    delta,_,_=retrieve_forecast(data['train'],int(row),base,lookup[p],state['precision'],None,codes,past,future,library,w,channel_index=0)
                    raw=bound_delta((delta*data['train']['metric_std'][row])[None],data['train']['raw_quantiles'][row:row+1])[0]/data['train']['metric_std'][row]
                    losses[str(w)].append(float(np.mean((target-.25*raw)**2)))
            weight=min(WEIGHTS,key=lambda w:(np.mean(losses[str(w)]),w)) if count>=4 else 0.
            selections[str(sid)]=dict(encoder=selection,retrieval_weight=weight,calibration_queries=count,base_folds=base_folds,
                retrieval_losses={k:float(np.mean(v)) if v else None for k,v in losses.items()},weekly=True,group=group.tolist())
            for sp in ['validation','test']:
                current=data[sp];rows=all_rows[sp];qep=local[current['start'][rows,None]+np.arange(96)[None,:]]
                # Only the history half is passed to the encoder.
                encoded=predict(state,history_features(qep,current['start'][rows],means,stds,period,weekly=True,block_width=history_width))
                f=feature(store,current,rows,s,period,plan,probe[sp][rows],False)
                bases[sp][rows]=current['base'][rows]+beta*bound_delta(covariance(base_state,f)*current['std'][rows,None],current['raw_quantiles'][rows])
                for j,i in enumerate(rows):
                    scale=max(float(current['raw_x'][i].std()),.1*stds[0]);level=float(current['raw_x'][i,-16:].mean())
                    path=level+decode(encoded[j],len(group),width)[:,0]*scale
                    delta=(path-bases[sp][i])/current['metric_std'][i]
                    deltas['encoder_only'][sp][i]=bound_delta((delta*current['metric_std'][i])[None],current['raw_quantiles'][i:i+1])[0]/current['metric_std'][i]
                    r,chosen,_=retrieve_forecast(current,int(i),bases[sp][i],encoded[j],state['precision'],None,codes,past,future,library,weight,channel_index=0)
                    deltas['future_state_rag'][sp][i]=bound_delta((r*current['metric_std'][i])[None],current['raw_quantiles'][i:i+1])[0]/current['metric_std'][i]
                    if sp=='test':selected_starts[i]=chosen
            if sid%40==0 or sid==len(store.series)-1:print('GROUP FUTURE',checkpoint.name,sid+1,'/',len(store.series),flush=True)
        experts={k:dict(v) for k,v in deltas.items()}
        for fraction in [.25,.5,.75]:experts[f'mix_{fraction}']={sp:(1-fraction)*deltas['encoder_only'][sp]+fraction*deltas['future_state_rag'][sp] for sp in ['validation','test']}
        val=data['validation'];test=data['test'];target=(val['raw_y']-bases['validation'])/val['metric_std'][:,None]
        policies={k:choose_policy(target,{k:v['validation']},val['sid'],val['start'],require_mae=True) for k,v in deltas.items()}
        policies['routed']=choose_policy(target,{k:v['validation'] for k,v in experts.items()},val['sid'],val['start'],require_mae=True)
        final_base=bases['test'].copy()
        for sid,p in base_policy.items():
            if p['beta']==0 and p.get('proposed_beta',0.)>0:
                final_base[test['sid']==int(sid)]=test['base'][test['sid']==int(sid)]
                for policy in policies.values():policy[sid].update(beta=0.,family='direct',reason='base expert rejected on final validation block')
        write_json(out/'policy.json',policies);write_json(out/'selection.json',dict(groups=selections))
        forecasts=dict(moirai_direct=test['base'],no_rag=final_base)
        for kind,values in deltas.items():forecasts[kind]=final_base+apply_policy(policies[kind],{kind:values['test']},test['sid'])*test['metric_std'][:,None]
        forecasts['routed']=final_base+apply_policy(policies['routed'],{k:v['test'] for k,v in experts.items()},test['sid'])*test['metric_std'][:,None]
        truth={k:test[k] for k in ['raw_y','sid','start']};truth['std']=test['metric_std']
        np.savez_compressed(out/'predictions.npz',**forecasts,**truth,selected_starts=selected_starts)
        metrics={k:metric(v,truth) for k,v in forecasts.items()}
        result=dict(complete=True,optimize_resolution=optimize_resolution,grouped=True,dataset=checkpoint.name,variables=len(store.series),queries=len(test['sid']),metrics=metrics,
            source_checkpoint=str(checkpoint),source_run_sha256=sha256(checkpoint/'run.json'),groups_sha256=sha256(out/'groups.json'),
            group_model_hashes={p.name:sha256(p) for p in out.glob('encoder_*.npz')},selection_sha256=sha256(out/'selection.json'),policy_sha256=sha256(out/'policy.json'),
            predictions_sha256=sha256(out/'predictions.npz'),gain_vs_direct_percent=100*(1-metrics['routed']['nmse']/metrics['moirai_direct']['nmse']),
            routed_vs_encoder_only_percent=100*(1-metrics['routed']['nmse']/metrics['encoder_only']['nmse']),
            gain_interval=interval(test['base'],forecasts['routed'],truth),rag_interval=interval(forecasts['encoder_only'],forecasts['routed'],truth),
            mechanism='Per-target8-channel future-state encoder; group correlation from memory prefix only; future prototypes and validation-only routing',
            caveat='Reused development test periods; all variables retained; these large datasets have no remaining disjoint full192-point test windows')
        write_json(out/'run.json',result);print('GROUP FUTURE RESULT',checkpoint.name,metrics,flush=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args()
    for name in a.datasets:experiment(Path(a.checkpoint_root)/name,Path(a.source)/name,Path(a.output)/name)
