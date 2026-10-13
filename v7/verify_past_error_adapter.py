"""Rebuild selected TRAIN models, causal updates, policies and paired metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .contextual_adapter import context_plan
from .past_error_adapter import feature,rerank,fit_completed,WEIGHTS
from .query_past_error_adapter import scale_data
from .residual_adapter import train
from .benefit_rerank import covariance
from .safe_error_policy import choose_policy,apply_policy,bound_delta
from .benchmark_state_recall import metric


def verify(root,source,names=None):
    records=[];root=Path(root)
    for path in sorted(root.glob('*/run.json')):
        if names and path.parent.name not in names:continue
        folder=path.parent;run=json.loads(path.read_text(encoding='utf-8'));cache=Path(run['source_cache']);probes=Path(run['source_probes'])
        assert sha256(cache/'manifest.json')==run['source_manifest_sha256']
        assert sha256(probes/'manifest.json')==run['source_probe_manifest_sha256']
        for name,key in [('selection.json','selection_sha256'),('policy.json','policy_sha256'),('context_plan.json','context_plan_sha256'),('predictions.npz','predictions_sha256')]:assert sha256(folder/name)==run[key]
        meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'));pm=json.loads((probes/'manifest.json').read_text(encoding='utf-8'))
        selections=json.loads((folder/'selection.json').read_text(encoding='utf-8'));policies=json.loads((folder/'policy.json').read_text(encoding='utf-8'))
        data={sp:load(cache,sp) for sp in ['train','validation','test']};metric_data={k:v.copy() for k,v in data['test'].items() if k in ['raw_y','std','sid','start']}
        probe={}
        for sp in data:
            if run.get('annual_calendar'):data[sp]['annual_calendar']=True
            assert sha256(probes/f'{sp}.npz')==pm['files'][sp]
            with np.load(probes/f'{sp}.npz',allow_pickle=False) as f:
                np.testing.assert_array_equal(f['sid'],data[sp]['sid']);np.testing.assert_array_equal(f['start'],data[sp]['start']);probe[sp]=f['error'].copy()
            if run.get('adaptive_scale'):scale_data(data[sp])
        store=Store(Path(source)/folder.name/'store');plan=context_plan(store)
        assert plan==json.loads((folder/'context_plan.json').read_text(encoding='utf-8'))
        deltas={kind:{sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']} for kind in policies};pairs=0
        try:
            for s in store.series:
                sid=s['sid'];ids=np.flatnonzero(data['train']['sid']==sid)
                for current in data.values():
                    i=np.flatnonzero(current['sid']==sid);starts=current['candidate_starts'][i]
                    assert np.all(starts+192<=current['start'][i,None]);assert np.all(starts+192<=s['memory_end']);pairs+=starts.size
                if 'augmentation' in meta:
                    edge=np.quantile(data['train']['start'][ids],.5)
                    assert data['train']['start'][ids][data['train']['start'][ids]>=edge].min()>=s['memory_end']
                memory=cache/f'memory_errors_{sid}.npz';assert sha256(memory)==meta['memory_files'][str(sid)]
                with np.load(memory,allow_pickle=False) as f:ms=f['starts'].copy();me=f['errors'].copy()
                if run.get('adaptive_scale'):
                    sd=np.array([store.window(sid,int(p),96).std() for p in ms]);me=me*(max(s['memory_std'],1e-6)/np.maximum(sd,.1*max(s['memory_std'],1e-6)))[:,None]
                for kind in policies:
                    trials=[]
                    for weight in (WEIGHTS if kind=='with_rag' else [0.]):
                        r=rerank(data['train'],ids,probe['train'][ids],ms,me,weight) if kind=='with_rag' else None
                        x=feature(store,data['train'],ids,s,run['period'],plan,probe['train'][ids],kind=='with_rag',r)
                        state,selection=train(x,data['train']['target'][ids],data['train']['start'][ids])
                        trials.append((float(np.mean(selection['losses'][str(selection['penalty'])])),weight,state,selection,x))
                    _,weight,state,selection,x=min(trials,key=lambda t:(t[0],t[1]))
                    selection.update(error_state_weight=weight,weight_losses={str(t[1]):t[0] for t in trials})
                    assert selection==selections[str(sid)][kind]
                    name=f'past_{sid}_{kind}.npz';assert sha256(folder/name)==run['past_hashes'][name]
                    with np.load(folder/name,allow_pickle=False) as f:
                        for key in f.files:np.testing.assert_allclose(f[key],state[key],rtol=0,atol=1e-10)
                    ax=x.copy();ay=data['train']['target'][ids].copy();ap=data['train']['start'][ids].copy()
                    for sp in ['validation','test']:
                        current=data[sp];rows=np.flatnonzero(current['sid']==sid)
                        r=rerank(current,rows,probe[sp][rows],ms,me,weight) if kind=='with_rag' else None
                        f=feature(store,current,rows,s,run['period'],plan,probe[sp][rows],kind=='with_rag',r)
                        if run.get('online'):
                            pred=np.zeros_like(current['base'][rows],dtype=float)
                            for j in np.argsort(current['start'][rows]):
                                fitted=fit_completed(ax,ay,ap,current['start'][rows[j]],selection['penalty']);pred[j]=covariance(fitted,f[j])
                                ax=np.concatenate([ax,f[j:j+1]]);ay=np.concatenate([ay,current['target'][rows[j:j+1]]]);ap=np.r_[ap,current['start'][rows[j]]]
                        else:pred=covariance(state,f)
                        deltas[kind][sp][rows]=bound_delta(pred*current['std'][rows,None],current['raw_quantiles'][rows])/current['std'][rows,None]
                if sid%100==0:print('VERIFY',folder.name,sid+1,'/',len(store.series),flush=True)
            val=data['validation'];test=data['test']
            with np.load(folder/'predictions.npz',allow_pickle=False) as f:
                for kind in policies:
                    policy=choose_policy(val['target'],{kind:deltas[kind]['validation']},val['sid'],val['start'],require_mae=True);assert policy==policies[kind]
                    pred=test['base']+apply_policy(policy,{kind:deltas[kind]['test']},test['sid'])*test['std'][:,None]
                    np.testing.assert_array_equal(f['past_'+kind],pred)
                for name,m in run['metrics'].items():assert metric(f[name],metric_data)==m
            records.append(dict(dataset=folder.name,train_models_and_forward_selections_rebuilt=True,policies_and_metrics_rebuilt=True,
                nonoverlapping_candidate_pairs=pairs,online=run.get('online',False),adaptive_scale=run.get('adaptive_scale',False)))
            print('VERIFIED',records[-1],flush=True)
        finally:store.close()
    write_json(root/'verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--datasets',nargs='+')
    a=p.parse_args();verify(a.root,a.source,a.datasets)
