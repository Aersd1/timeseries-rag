"""Two-stage complementary RAG, using nested forward teacher predictions."""
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
from .trend_rag import ResidualPool
from .complementary_rag import forward_base,build_table,train_incremental
from .benchmark_state_recall import corrections,metric,interval


def experiment(checkpoint,source,out):
    out.mkdir(parents=True,exist_ok=False);run=json.loads((checkpoint/'run.json').read_text(encoding='utf-8'))
    if run.get('online'):raise ValueError('Incremental experiment requires a frozen base')
    cache=Path(run['source_cache']);probes=Path(run['source_probes'])
    assert sha256(cache/'manifest.json')==run['source_manifest_sha256']
    assert sha256(probes/'manifest.json')==run['source_probe_manifest_sha256']
    for name,key in [('policy.json','policy_sha256'),('selection.json','selection_sha256'),('context_plan.json','context_plan_sha256')]:assert sha256(checkpoint/name)==run[key]
    plan=json.loads((checkpoint/'context_plan.json').read_text(encoding='utf-8'))
    old_policy=json.loads((checkpoint/'policy.json').read_text(encoding='utf-8'))['without_rag']
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    pm=json.loads((probes/'manifest.json').read_text(encoding='utf-8'))
    store=Store(source/'store');assert store.meta['data_id']==run['source_data_id']
    data={sp:load(cache,sp) for sp in ['train','validation','test']};probe={}
    for sp,current in data.items():
        current['metric_std']=current['std'].copy()
        if run.get('annual_calendar'):current['annual_calendar']=True
        if run.get('adaptive_scale'):scale_data(current)
        assert sha256(probes/f'{sp}.npz')==pm['files'][sp]
        with np.load(probes/f'{sp}.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['sid'],current['sid']);np.testing.assert_array_equal(f['start'],current['start']);probe[sp]=f['error'].copy()
    bases={sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']}
    deltas={sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']};selection={};trace={}
    try:
        for s in store.series:
            sid=s['sid'];ids=np.flatnonzero(data['train']['sid']==sid);beta=old_policy[str(sid)].get('proposed_beta',0.)
            name=f'past_{sid}_without_rag.npz';assert sha256(checkpoint/name)==run['past_hashes'][name]
            with np.load(checkpoint/name,allow_pickle=False) as f:base_state={k:f[k].copy() for k in f.files}
            x=feature(store,data['train'],ids,s,run['period'],plan,probe['train'][ids],False)
            rows,oof,folds=forward_base(x,data['train']['target'][ids],data['train']['start'][ids],data['train']['base'][ids],
                data['train']['raw_quantiles'][ids],data['train']['std'][ids],beta)
            pool=ResidualPool(store,sid,meta['window'],stride=meta.get('memory_stride',1))
            past=np.stack([store.window(sid,p,96) for p in pool.starts]);future=np.stack([store.window(sid,p+96,96) for p in pool.starts])
            selected_ids=ids[rows]
            table=build_table(data['train'],selected_ids,oof,s,pool.starts,past,future,pool.past,run['period'],meta['window'])
            target=(data['train']['raw_y'][selected_ids]-oof)/data['train']['metric_std'][selected_ids,None]
            state,decision=train_incremental(table,target,data['train']['raw_quantiles'][selected_ids])
            selection[str(sid)]=dict(base_folds=folds,incremental=decision,proposed_base_beta=beta)
            np.savez(out/f'ranker_{sid}.npz',**state)
            np.savez_compressed(out/f'outer_train_{sid}.npz',rows=selected_ids,base=oof,target=target,start=table['start'])
            for sp in ['validation','test']:
                current=data[sp];rows=np.flatnonzero(current['sid']==sid)
                f=feature(store,current,rows,s,run['period'],plan,probe[sp][rows],False)
                raw=bound_delta(covariance(base_state,f)*current['std'][rows,None],current['raw_quantiles'][rows])
                bases[sp][rows]=current['base'][rows]+beta*raw
                table=build_table(current,rows,bases[sp][rows],s,pool.starts,past,future,pool.past,run['period'],meta['window'])
                r=corrections(state,table)
                deltas[sp][rows]=bound_delta(r['correction']*current['metric_std'][rows,None],current['raw_quantiles'][rows])/current['metric_std'][rows,None]
                trace.setdefault(sp,{}).setdefault('selected_starts',{})[sid]=table['starts'][np.arange(len(rows))[:,None],r['selected']]
            if sid%40==0 or sid==len(store.series)-1:print('COMPLEMENTARY',checkpoint.name,sid+1,'/',len(store.series),flush=True)
        val=data['validation'];test=data['test']
        residual=(val['raw_y']-bases['validation'])/val['metric_std'][:,None]
        policy=choose_policy(residual,{'incremental':deltas['validation']},val['sid'],val['start'],require_mae=True)
        final_base=np.zeros_like(test['base'])
        for sid in np.unique(test['sid']):
            rows=test['sid']==sid;p=old_policy[str(int(sid))]
            base_confirmed=p['beta']>0 or p.get('proposed_beta',0.)==0
            final_base[rows]=bases['test'][rows] if base_confirmed else test['base'][rows]
            if not base_confirmed:
                policy[str(int(sid))].update(family='direct',beta=0.,reason='base expert rejected on final validation block; no retry')
        write_json(out/'selection.json',selection);write_json(out/'policy.json',policy)
        final=final_base+apply_policy(policy,{'incremental':deltas['test']},test['sid'])*test['metric_std'][:,None]
        truth={k:test[k] for k in ['raw_y','sid','start']};truth['std']=test['metric_std']
        forecasts=dict(moirai_direct=test['base'],no_rag=final_base,complementary_rag=final)
        np.savez_compressed(out/'predictions.npz',**forecasts,**truth)
        np.savez_compressed(out/'validation_trace.npz',base=bases['validation'],delta=deltas['validation'],target=residual,sid=val['sid'],start=val['start'])
        metrics={k:metric(v,truth) for k,v in forecasts.items()}
        result=dict(complete=True,dataset=checkpoint.name,metrics=metrics,queries=len(test['sid']),variables=len(store.series),
            gain_vs_direct_percent=100*(1-metrics['complementary_rag']['nmse']/metrics['moirai_direct']['nmse']),
            rag_incremental_gain_percent=100*(1-metrics['complementary_rag']['nmse']/metrics['no_rag']['nmse']),
            gain_interval=interval(final_base,final,truth),adopted_variables=[int(sid) for sid,p in policy.items() if p['beta']>0],
            source_checkpoint=str(checkpoint),source_run_sha256=sha256(checkpoint/'run.json'),policy_sha256=sha256(out/'policy.json'),
            selection_sha256=sha256(out/'selection.json'),predictions_sha256=sha256(out/'predictions.npz'),
            ranker_hashes={p.name:sha256(p) for p in out.glob('ranker_*.npz')},
            mechanism='Nested forward no-RAG teacher; learned incremental utility of actual analogue continuations; first2 validation calibrate, third only rejects',
            caveat='Exploratory reused test periods; no universally improving RAG guarantee')
        write_json(out/'run.json',result);print('COMPLEMENTARY RESULT',checkpoint.name,metrics,'RAG gain',result['rag_incremental_gain_percent'],flush=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args()
    for name in a.datasets:experiment(Path(a.checkpoint_root)/name,Path(a.source)/name,Path(a.output)/name)
