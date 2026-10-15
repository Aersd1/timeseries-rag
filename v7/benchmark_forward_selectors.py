"""Train selectors on purged prefix predictions, then lock every TEST policy."""
import argparse,gc
from pathlib import Path
import numpy as np
import torch
import joblib
from v5.data import Store
from .common import sha256,write_json
from .dense_forward import read,materialize,foundation,make_index,complete_origins
from .forward_experts import ARMS,model_from
from .benchmark_unified_residual import predict as expert_predict
from .forward_selectors import FAMILIES,SETTINGS,SummaryUtility,SequenceUtility,inputs,labels,fit_models,prediction,choose_blocks,calibration,apply_policy,portfolio
from .selective_rag import error_metrics


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def development(source,out):
    phases=['fold0','fold1','fold2'];total=sum(read(out/(p+'_held_index.json'))['queries'] for p in phases)
    merged=None;forecasts={arm:np.empty((total,96),np.float32) for arm in ARMS};offset=0
    for phase in phases:
        metadata=read(out/(phase+'_experts.json'))
        if sha256(out/(phase+'_predictions.npz'))!=metadata['predictions_sha256']:raise ValueError('Changed forward predictions')
        d=materialize(source,out,phase+'_held');n=len(d['sid'])
        if merged is None:merged={k:np.empty((total,)+v.shape[1:],v.dtype) for k,v in d.items()}
        for k,v in d.items():merged[k][offset:offset+n]=v
        with np.load(out/(phase+'_predictions.npz'),allow_pickle=False) as f:
            np.testing.assert_array_equal(f['sid'],d['sid']);np.testing.assert_array_equal(f['start'],d['start'])
            for arm in ARMS:forecasts[arm][offset:offset+n]=f[arm]
        offset+=n;del d
    val=materialize(source,out,'final_held');vp=load_npz(out/'final_predictions.npz')
    for k in ['sid','start']:np.testing.assert_array_equal(vp[k],val[k])
    return merged,forecasts,val,{arm:vp[arm] for arm in ARMS}


def history_data(*datasets):
    # The competence library only needs targets, medians and identities.
    b=np.concatenate([d['raw_quantiles'][:,4] for d in datasets])
    return dict(sid=np.concatenate([d['sid'] for d in datasets]),start=np.concatenate([d['start'] for d in datasets]),
        raw_y=np.concatenate([d['raw_y'] for d in datasets]),raw_quantiles=np.broadcast_to(b[:,None],(len(b),9,96)))


def save_models(folder,family,models,normal,records,provenance):
    folder.parent.mkdir(parents=True,exist_ok=True)
    if family=='hist_gbdt':joblib.dump(dict(models=models,normal=normal),folder.with_suffix('.joblib'),compress=3);path=folder.with_suffix('.joblib')
    else:
        path=folder.with_suffix('.pt');torch.save(dict(family=family,normal=normal,models=[{k:v.cpu() for k,v in m.state_dict().items()} for m in models]),path)
    write_json(folder.with_suffix('.json'),dict(family=family,file=path.name,sha256=sha256(path),records=records,normal=normal,provenance=provenance))


def load_models(folder,device):
    meta=read(folder.with_suffix('.json'));path=folder.parent/meta['file']
    if sha256(path)!=meta['sha256']:raise ValueError('Changed selector parameters')
    if meta['family']=='hist_gbdt':saved=joblib.load(path);return saved['models'],saved['normal']
    saved=torch.load(path,map_location='cpu',weights_only=True);normal=saved['normal'];models=[]
    for state in saved['models']:
        m=(SequenceUtility(normal['width']) if meta['family']=='raw_sequence' else SummaryUtility(normal['width']));m.load_state_dict(state);models.append(m.to(device).eval())
    return models,normal


def fit(source,out,device):
    if (out/'selector_contract.json').exists():raise ValueError('All policies already frozen; use --test or verifier')
    protocol=out/'selector_protocol.json'
    if protocol.exists():
        if read(protocol)!=SETTINGS:raise ValueError('Changed selector protocol')
    else:write_json(protocol,SETTINGS)
    meta=read(out/'preparation.json');cuts=meta['layout']['cuts'];anchor=read(out/f'prefix_{cuts[0]}.json')
    train,tp,val,vp=development(source,out);hist=history_data(train);vhist=history_data(train,val);blocks=choose_blocks(val,meta['layout']['validation_end'])
    cohorts=dict(forward_full=np.arange(len(train['sid'])),sparse_last4=np.flatnonzero(np.isin(train['start'],np.unique(train['start'])[-4:])))
    provenance=dict(foundation_manifest_sha256=sha256(out/'foundation_manifest.json'),anchor_sha256=sha256(out/f'prefix_{cuts[0]}.json'),
        phases={p:sha256(out/(p+'_experts.json')) for p in ['fold0','fold1','fold2','final']},
        cohort_sizes={k:dict(queries=len(v),origins=len(np.unique(train['start'][v])),first_start=int(train['start'][v].min()),last_start=int(train['start'][v].max())) for k,v in cohorts.items()})
    policies={};predictions={};models_manifest={}
    for arm in ARMS:
        gain,harm=labels(train,tp[arm],anchor);history_pred=np.concatenate([tp[arm],vp[arm]])
        for family in FAMILIES:
            xt=inputs(train,tp[arm],anchor,meta['period'],arm,hist,tp[arm],raw=family=='raw_sequence')
            xv=inputs(val,vp[arm],anchor,meta['period'],arm,vhist,history_pred,raw=family=='raw_sequence')
            for cohort,ids in cohorts.items():
                key=cohort+'__'+arm+'__'+family;path=out/'selectors'/key
                if path.with_suffix('.json').exists():models,normal=load_models(path,device)
                else:
                    models,normal,records=fit_models(family,xt,gain,harm,train,anchor,ids,device)
                    save_models(path,family,models,normal,records,provenance)
                pred=prediction(family,models,normal,xv,device);policies[key]=calibration(val,vp[arm],pred,family,anchor,blocks);predictions[key]=pred
                models_manifest[key]=sha256(path.with_suffix('.json'))
                print('FORWARD SELECTOR',out.name,key,policies[key]['enabled'],policies[key]['strength'],flush=True)
                del models,pred;gc.collect()
                if device.startswith('cuda'):torch.cuda.empty_cache()
            del xt,xv
        static=calibration(val,vp[arm],None,'static',anchor,blocks)
        for cohort in cohorts:
            key=cohort+'__'+arm;pool={f:policies[key+'__'+f] for f in FAMILIES};pool['static']=static
            # Rank ONLY first-two-block objectives, not confirmation outcomes.
            policies[key+'__portfolio']=portfolio(pool)
        policies[arm+'__static']=static
    write_json(out/'selector_policy.json',policies);np.savez_compressed(out/'selector_validation.npz',**predictions)
    write_json(out/'selector_contract.json',dict(primary='forward_full__real__portfolio',provenance=provenance,settings=SETTINGS,
        models=models_manifest,policy_sha256=sha256(out/'selector_policy.json'),validation_sha256=sha256(out/'selector_validation.npz'),
        calibration_origins=[np.unique(val['start'][b]).tolist() for b in blocks],
        code_sha256={k:sha256('v7/'+k) for k in ['dense_forward.py','forward_experts.py','forward_selectors.py','benchmark_forward_selectors.py']},
        caveat='Whole new learned pipeline is prefix fitted for OOF labels; old TEST periods already explored. Expert seed13 is fixed, selectors use three seeds.'))


def test(source,out,device):
    if (out/'run.json').exists():raise ValueError('Completed TEST is immutable')
    contract=read(out/'selector_contract.json');meta=read(out/'preparation.json');policies=read(out/'selector_policy.json')
    if sha256(out/'selector_policy.json')!=contract['policy_sha256']:raise ValueError('Changed policies')
    for name,expected in contract['code_sha256'].items():
        if sha256('v7/'+name)!=expected:raise ValueError('Changed frozen code '+name)
    foundation(source,out,device,test=True);store=Store(meta['store']);legacy={};dense={}
    try:
        with np.load(source/'test.npz',allow_pickle=False) as f:
            sids,starts=f['sid'],f['start']
            for s in store.series:
                sid=s['sid'];legacy[sid]=starts[sids==sid];dense[sid]=complete_origins(s,range(s['validation_end'],s['n']-191,96))
        all_starts={sid:np.unique(np.r_[dense[sid],legacy[sid]]).tolist() for sid in legacy}
        cutoff=meta['layout']['cuts'][-1];make_index(source,out,cutoff,all_starts,'test')
    finally:store.close()
    d=materialize(source,out,'test');prefix=read(out/f'prefix_{cutoff}.json');anchor=read(out/f'prefix_{meta["layout"]["cuts"][0]}.json')
    candidates={}
    for arm in ARMS:
        checkpoint=out/f'final_{arm}.pt';m=model_from(torch.load(checkpoint,map_location='cpu',weights_only=True),device)
        candidates[arm]=expert_predict(m,d,prefix,meta['period'],arm,device);del m
    train,tp,val,vp=development(source,out);h=history_data(train,val);del train,val;gc.collect()
    outputs={'moirai_direct':d['raw_quantiles'][:,4]};route_data={};score_data={}
    for arm in ARMS:
        hp=np.concatenate([tp[arm],vp[arm]])
        for family in FAMILIES:
            x=inputs(d,candidates[arm],anchor,meta['period'],arm,h,hp,raw=family=='raw_sequence')
            for cohort in SETTINGS['cohorts']:
                key=cohort+'__'+arm+'__'+family;models,normal=load_models(out/'selectors'/key,device)
                pred=prediction(family,models,normal,x,device);score_data[key]=pred
                outputs[key],mask=apply_policy(d,candidates[arm],pred,policies[key]);route_data[key]=mask
                combined=policies[cohort+'__'+arm+'__portfolio']
                if combined['selected_family']==family:
                    name=cohort+'__'+arm+'__portfolio';outputs[name],route_data[name]=apply_policy(d,candidates[arm],pred,combined)
                del models,pred
            del x;gc.collect()
        for cohort in SETTINGS['cohorts']:
            key=cohort+'__'+arm+'__portfolio';policy=policies[key]
            if policy['selected_family']=='static':outputs[key],route_data[key]=apply_policy(d,candidates[arm],None,policy)
        outputs[arm+'__static'],route_data[arm+'__static']=apply_policy(d,candidates[arm],None,policies[arm+'__static'])
        outputs[arm+'__raw']=candidates[arm]
        print('FORWARD TEST',out.name,arm,'done',flush=True)
    dense_mask=np.array([int(t) in set(dense[int(s)]) for s,t in zip(d['sid'],d['start'])]);legacy_mask=np.array([int(t) in set(legacy[int(s)]) for s,t in zip(d['sid'],d['start'])])
    metrics={}
    for scope,mask in [('dense',dense_mask),('legacy',legacy_mask)]:
        truth={k:v[mask] for k,v in d.items() if k in ['raw_y','sid','start','std']}
        metrics[scope]={k:error_metrics(truth,v[mask],anchor) for k,v in outputs.items()}
    np.savez_compressed(out/'predictions.npz',**outputs,**{k:d[k] for k in ['sid','start','raw_y','std']},dense_mask=dense_mask,legacy_mask=legacy_mask)
    np.savez_compressed(out/'routing.npz',**route_data);np.savez_compressed(out/'utility_predictions.npz',**score_data)
    write_json(out/'run.json',dict(complete=True,dataset=out.name,primary=contract['primary'],queries=len(d['sid']),
        dense_queries=int(dense_mask.sum()),legacy_queries=int(legacy_mask.sum()),metrics=metrics,
        contract_sha256=sha256(out/'selector_contract.json'),predictions_sha256=sha256(out/'predictions.npz'),routing_sha256=sha256(out/'routing.npz'),utility_sha256=sha256(out/'utility_predictions.npz')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');p.add_argument('--test',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:(test if a.test else fit)(Path(a.source)/name,Path(a.output)/name,a.device)
