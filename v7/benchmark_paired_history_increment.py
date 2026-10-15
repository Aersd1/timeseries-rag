"""Freeze a query-only base, learn paired-history increments, then evaluate.

Reuses the exact audited 5+5 indices/foundation forecasts of dense_forward.
All requested datasets finish fitting and VAL locking before this CLI loads TEST.
"""
import argparse,gc
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .dense_forward import read,materialize
from .forward_experts import model_from
from .benchmark_unified_residual import predict as old_predict
from .benchmark_forward_selectors import load_npz,save_models,load_models
from .forward_selectors import fit_models,prediction,choose_blocks,calibration,apply_policy
from .selective_rag import error_metrics,nonworse,rows
from .paired_history_increment import ARMS,PLAN,PairedHistoryIncrement,pack,with_base,fit_increment,predict,summary,mask_modalities

PHASES=['fold0','fold1','fold2','final']
DEPENDENCIES=['paired_history_increment.py','benchmark_paired_history_increment.py','dense_forward.py',
    'unified_residual.py','aligned_case_fusion.py','forward_selectors.py','forward_experts.py','benchmark_unified_residual.py']


def implementation():return {p:sha256('v7/'+p) for p in DEPENDENCIES}


def context(forward,out):
    forward=Path(forward);out=Path(out);meta=read(forward/'preparation.json');out.mkdir(parents=True,exist_ok=True)
    contract=read(forward/'selector_contract.json')
    for name,expected in contract['code_sha256'].items():
        if sha256('v7/'+name)!=expected:raise ValueError('Changed audited upstream '+name)
    protocol=dict(forward=str(forward.resolve()),source=meta['source'],plan=PLAN,code=implementation(),
        upstream_contract_sha256=sha256(forward/'selector_contract.json'))
    target=out/'protocol.json'
    if target.exists():
        if read(target)!=protocol:raise ValueError('Changed paired experiment protocol')
    else:write_json(target,protocol)
    return meta


def baseline_model(forward,phase,device):
    r=read(forward/(phase+'_experts.json'));entry=next(m for m in r['models'] if m['mode']=='query_only')
    path=forward/entry['file']
    if sha256(path)!=entry['sha256']:raise ValueError('Changed frozen query-only base')
    return model_from(torch.load(path,map_location='cpu',weights_only=True),device),entry


def load_increment(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True);model=PairedHistoryIncrement()
    model.load_state_dict(saved['state_dict']);return model.to(device).eval()


def fit_phase(forward,out,phase,device):
    meta=read(forward/'preparation.json');source=Path(meta['source']);done=out/(phase+'_experts.json')
    if done.exists():
        record=read(done)
        if sha256(out/(phase+'_predictions.npz'))!=record['prediction_sha256']:raise ValueError('Changed held predictions')
        for arm in ARMS:
            if sha256(out/record['models'][arm]['file'])!=record['models'][arm]['sha256']:raise ValueError('Changed increment')
        return
    index=read(forward/(phase+'_train_index.json'));cut=index['cutoff'];stats=read(forward/f'prefix_{cut}.json')
    provenance=dict(index_sha256=sha256(forward/(phase+'_train_index.npz')),stats_sha256=sha256(forward/f'prefix_{cut}.json'),
        baseline_manifest_sha256=sha256(forward/(phase+'_experts.json')),protocol_sha256=sha256(out/'protocol.json'))
    data=materialize(source,forward,phase+'_train')
    if np.any(data['start']+192>cut):raise ValueError('Increment-fit labels cross prefix cutoff')
    bpath=out/(phase+'_base_train.npz')
    if bpath.exists():
        base=load_npz(bpath);assert all(np.array_equal(base[k],data[k]) for k in ['sid','start'])
        if read(bpath.with_suffix('.json'))!=provenance:raise ValueError('Changed base fit inputs')
        adapted=base['adapted']
    else:
        model,entry=baseline_model(forward,phase,device)
        adapted=old_predict(model,data,stats,meta['period'],'query_only',device)
        np.savez_compressed(bpath,adapted=adapted,sid=data['sid'],start=data['start']);write_json(bpath.with_suffix('.json'),provenance)
        del model;gc.collect();torch.cuda.empty_cache() if device.startswith('cuda') else None
    bases=[data['raw_quantiles'][:,4],adapted];records={}
    for arm in ARMS:
        path=out/(phase+'__'+arm+'.pt');manifest=path.with_suffix('.json')
        if manifest.exists():
            r=read(manifest)
            if r['provenance']!=provenance or sha256(path)!=r['sha256']:raise ValueError('Changed fit resume')
        else:
            model,losses=fit_increment(data,bases,stats,meta['period'],arm,device)
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()}),path)
            r=dict(file=path.name,sha256=sha256(path),losses=losses,provenance=provenance,
                training_queries=len(data['sid']),last_fit_label_end=int(data['start'].max()+192),cutoff=cut,
                parameters=sum(p.numel() for p in model.parameters()))
            write_json(manifest,r);del model;gc.collect()
        records[arm]=r
        print('PAIRED FIT',out.name,phase,arm,flush=True)
        if device.startswith('cuda'):torch.cuda.empty_cache()
    del data,bases,adapted;gc.collect()
    held=materialize(source,forward,phase+'_held')
    if np.any(held['start']<cut):raise ValueError('Held history before prefix cutoff')
    upstream=read(forward/(phase+'_experts.json'))
    if sha256(forward/(phase+'_predictions.npz'))!=upstream['predictions_sha256']:raise ValueError('Changed frozen held base')
    old=load_npz(forward/(phase+'_predictions.npz'));assert all(np.array_equal(old[k],held[k]) for k in ['sid','start'])
    outputs=dict(sid=held['sid'],start=held['start'],adapted=old['query_only'])
    for arm in ARMS:
        model=load_increment(out/records[arm]['file'],device);p,scale=pack(held,stats,meta['period'],arm)
        for mode,base in enumerate([held['raw_quantiles'][:,4],old['query_only']]):
            outputs[f'{arm}__{mode}']=predict(model,with_base(p,base,held,scale,mode),base,scale,device)
        del p,model;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    np.savez_compressed(out/(phase+'_predictions.npz'),**outputs)
    write_json(done,dict(complete=True,models=records,provenance=provenance,held_queries=len(held['sid']),
        first_held_start=int(held['start'].min()),prediction_sha256=sha256(out/(phase+'_predictions.npz'))))
    print('PAIRED PREFIX COMPLETE',out.name,phase,flush=True)


def experts(forward,out,device):
    context(forward,out)
    for phase in PHASES:fit_phase(forward,out,phase,device)


def development(forward,out):
    meta=read(forward/'preparation.json');data=None;preds={};offset=0
    count=sum(read(forward/(p+'_held_index.json'))['queries'] for p in PHASES[:3])
    for phase in PHASES[:3]:
        d=materialize(meta['source'],forward,phase+'_held');saved=load_npz(out/(phase+'_predictions.npz'))
        if sha256(out/(phase+'_predictions.npz'))!=read(out/(phase+'_experts.json'))['prediction_sha256']:raise ValueError('Changed OOF predictions')
        if data is None:data={k:np.empty((count,)+v.shape[1:],v.dtype) for k,v in d.items()}
        for k,v in d.items():data[k][offset:offset+len(v)]=v
        for k,v in saved.items():
            if k not in preds:preds[k]=np.empty((count,)+v.shape[1:],v.dtype)
            preds[k][offset:offset+len(v)]=v
        offset+=len(d['sid']);del d,saved;gc.collect()
    for k in ['sid','start']:np.testing.assert_array_equal(data[k],preds[k])
    return data,preds


def carrier(data,base):
    q=data['raw_quantiles'].copy();q[:,4]=base
    return dict(data,raw_quantiles=q)


def baseline_policy(val,adapted,anchor,blocks):
    direct=val['raw_quantiles'][:,4]
    b=[error_metrics(rows(val,i),direct[i],anchor) for i in blocks]
    a=[error_metrics(rows(val,i),adapted[i],anchor) for i in blocks]
    first=all(nonworse(x,y) for x,y in zip(a[:2],b[:2])) and sum(x['mse'] for x in a[:2])<sum(x['mse'] for x in b[:2])*(1-1e-5)
    confirm=all(nonworse(x,y) and x['mse']<y['mse']*(1-1e-5) for x,y in zip(a[2:],b[2:]))
    return dict(mode=int(first),enabled=bool(not first or confirm),calibration_adapter=a[:2],calibration_moirai=b[:2],
        confirmation_adapter=a[2:],confirmation_moirai=b[2:],
        rule='Choose base on first two blocks alone. Rejected adapted base falls back to Moirai AND disables RAG; no family reselection.')


def utility_labels(data,base,candidate,stats):
    sd=np.asarray(stats['std'])[data['sid'],None]
    gain=(((base.astype(float)-data['raw_y'])**2).reshape(-1,12,8).mean(2)-
        ((candidate.astype(float)-data['raw_y'])**2).reshape(-1,12,8).mean(2))/sd**2
    return gain,np.maximum(-gain,0)


def utility_inputs(data,base,candidate,anchor,period,arm,mode=0):
    p,scale=pack(data,anchor,period,arm);p=with_base(p,base,data,scale,mode)
    return dict(summary=summary(p,np.clip((candidate-base)/scale[:,None],-8,8)))


def selectors(forward,out,device):
    context(forward,out)
    if (out/'contract.json').exists():return
    meta=read(forward/'preparation.json');anchor=read(forward/f'prefix_{meta["layout"]["cuts"][0]}.json')
    train,tp=development(forward,out);val=materialize(meta['source'],forward,'final_held');vp=load_npz(out/'final_predictions.npz')
    blocks=choose_blocks(val,meta['layout']['validation_end']);base_policy=baseline_policy(val,vp['adapted'],anchor,blocks)
    policies={};utilities={};models={};ids=np.arange(len(train['sid']))
    for arm in ARMS:
        for mode in [0,1]:
            key=f'{arm}__{mode}';b=train['raw_quantiles'][:,4] if mode==0 else tp['adapted']
            vb=val['raw_quantiles'][:,4] if mode==0 else vp['adapted']
            path=out/'utilities'/key
            if path.with_suffix('.json').exists():nets,normal=load_models(path,device)
            else:
                x=utility_inputs(train,b,tp[key],anchor,meta['period'],arm,mode);g,h=utility_labels(train,b,tp[key],anchor)
                nets,normal,records=fit_models('summary_mlp',x,g,h,train,anchor,ids,device)
                save_models(path,'summary_mlp',nets,normal,records,dict(protocol_sha256=sha256(out/'protocol.json'),
                    prefix_prediction_sha256={p:sha256(out/(p+'_predictions.npz')) for p in PHASES[:3]}))
                del x,g,h;gc.collect()
            xv=utility_inputs(val,vb,vp[key],anchor,meta['period'],arm,mode);u=prediction('summary_mlp',nets,normal,xv,device)
            d=carrier(val,vb);choice=[('utility',calibration(d,vp[key],u,'summary_mlp',anchor,blocks)),
                ('static',calibration(d,vp[key],None,'static',anchor,blocks))]
            kind,policy=min(choice,key=lambda pair:pair[1]['calibration_objective'])
            policies[key]=dict(policy,selected_kind=kind)
            utilities[key]=u;models[key]=sha256(path.with_suffix('.json'))
            print('PAIRED UTILITY',out.name,key,policy['enabled'],flush=True)
            del nets,xv,u;gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    write_json(out/'policy.json',dict(baseline=base_policy,increments=policies))
    np.savez_compressed(out/'validation_utility.npz',**utilities)
    write_json(out/'contract.json',dict(complete=True,primary='paired',protocol_sha256=sha256(out/'protocol.json'),
        policy_sha256=sha256(out/'policy.json'),utility_sha256=sha256(out/'validation_utility.npz'),models=models,
        phases={p:sha256(out/(p+'_experts.json')) for p in PHASES},
        blocks=[np.unique(val['start'][ids]).tolist() for ids in blocks],code=implementation(),
        caveat='Current TEST periods previously explored; case alignment retains scalar history in without_history_sequence.'))
    print('PAIRED POLICIES LOCKED',out.name,base_policy['mode'],base_policy['enabled'],flush=True)


def final_policy(out,arm):
    saved=read(out/'policy.json');mode=saved['baseline']['mode'];p=dict(saved['increments'][f'{arm}__{mode}'])
    if not saved['baseline']['enabled']:p.update(enabled=False,reason='Adapted baseline confirmation rejected; Moirai fallback')
    return mode,p,saved['baseline']['enabled']


def test(forward,out,device):
    context(forward,out)
    if (out/'run.json').exists():return
    contract=read(out/'contract.json')
    if contract['code']!=implementation() or contract['policy_sha256']!=sha256(out/'policy.json'):raise ValueError('Changed frozen TEST contract')
    meta=read(forward/'preparation.json');stats=read(forward/f'prefix_{meta["layout"]["cuts"][-1]}.json');anchor=read(forward/f'prefix_{meta["layout"]["cuts"][0]}.json')
    d=materialize(meta['source'],forward,'test');old=load_npz(forward/'predictions.npz')
    for k in ['sid','start','raw_y']:np.testing.assert_array_equal(d[k],old[k])
    np.testing.assert_array_equal(d['raw_quantiles'][:,4],old['moirai_direct'])
    base_model,_=baseline_model(forward,'final',device);adapted=old_predict(base_model,d,stats,meta['period'],'query_only',device);del base_model;gc.collect()
    bases=[d['raw_quantiles'][:,4],adapted];mode,_,base_ok=final_policy(out,'paired');selected_base=bases[mode] if base_ok else bases[0]
    outputs=dict(moirai_direct=bases[0],query_adapter_raw=adapted,selected_base=selected_base,
        previous_primary=old[read(forward/'run.json')['primary']],sid=d['sid'],start=d['start'],raw_y=d['raw_y'],
        dense_mask=old['dense_mask'],legacy_mask=old['legacy_mask'])
    utility={};routing={};raws={};masks={};primary_model=None;primary_p=None;primary_scale=None
    for arm in ARMS:
        r=read(out/'final_experts.json')['models'][arm];model=load_increment(out/r['file'],device)
        p,scale=pack(d,stats,meta['period'],arm)
        for bm,base in enumerate(bases):
            key=f'{arm}__{bm}';raws[key]=predict(model,with_base(p,base,d,scale,bm),base,scale,device);outputs[key+'__raw']=raws[key]
        _,policy,_=final_policy(out,arm);key=f'{arm}__{mode}';nets,normal=load_models(out/'utilities'/key,device)
        x=utility_inputs(d,bases[mode],raws[key],anchor,meta['period'],arm,mode);u=prediction('summary_mlp',nets,normal,x,device)
        output,mask=apply_policy(carrier(d,selected_base),raws[key],u,policy)
        outputs[arm]=output;utility[arm]=u;routing[arm]=mask;masks[arm]=mask
        np.testing.assert_array_equal(output.reshape(-1,12,8)[~mask],selected_base.reshape(-1,12,8)[~mask])
        if arm=='paired':primary_model=model;primary_p=with_base(p,bases[mode],d,scale,mode);primary_scale=scale
        del nets,x;gc.collect()
        print('PAIRED TEST',out.name,arm,flush=True)
    # Same real weights, same adoption mask, same strength. Not TEST-selected outputs.
    _,policy,_=final_policy(out,'paired');mask=masks['paired'];interventions={}
    for name in ['empty','history_zero','history_deranged','future_zero','future_deranged','matched_random']:
        if name=='empty':raw=bases[mode].copy()
        else:
            if name=='history_zero':p=mask_modalities(primary_p,history=True)
            elif name=='future_zero':p=mask_modalities(primary_p,future=True)
            elif name=='matched_random':
                p,ss=pack(d,stats,meta['period'],'matched_random');p=with_base(p,bases[mode],d,ss,mode)
            else:
                p=dict(primary_p);k='past' if name=='history_deranged' else 'continuation'
                p[k]=p[k][np.arange(len(adapted))[:,None],d['shuffle_order']]
            raw=predict(primary_model,p,bases[mode],primary_scale,device)
        selected=selected_base.copy();proposal=selected_base+policy['strength']*(raw-selected_base)
        selected.reshape(-1,12,8)[mask]=proposal.reshape(-1,12,8)[mask]
        interventions[name+'__raw']=raw;interventions[name+'__fixed_route']=selected
    np.savez_compressed(out/'predictions.npz',**outputs);np.savez_compressed(out/'routing.npz',**routing)
    np.savez_compressed(out/'test_utility.npz',**utility);np.savez_compressed(out/'interventions.npz',**interventions)
    metrics={scope:{key:error_metrics(rows(d,np.flatnonzero(outputs[scope+'_mask'])),value[outputs[scope+'_mask']],anchor)
        for key,value in outputs.items() if value.shape==d['raw_y'].shape and key!='raw_y'} for scope in ['dense','legacy']}
    write_json(out/'run.json',dict(complete=True,dataset=out.name,primary='paired',metrics=metrics,
        baseline_mode=mode,baseline_enabled=base_ok,queries=len(d['sid']),adoption={a:float(masks[a][outputs['dense_mask']].mean()) for a in ARMS},
        contract_sha256=sha256(out/'contract.json'),predictions_sha256=sha256(out/'predictions.npz'),routing_sha256=sha256(out/'routing.npz'),
        test_utility_sha256=sha256(out/'test_utility.npz'),interventions_sha256=sha256(out/'interventions.npz')))
    print('PAIRED DATASET COMPLETE',out.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--forward',required=True);p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');p.add_argument('--stage',choices=['fit','experts','selectors','test','all'],default='all')
    a=p.parse_args();torch.set_num_threads(4)
    for stage in (['experts','selectors','test'] if a.stage=='all' else ['experts','selectors'] if a.stage=='fit' else [a.stage]):
        for name in a.datasets:globals()[stage](Path(a.forward)/name,Path(a.output)/name,a.device)
