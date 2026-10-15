"""Recompute paired-history predictions and chronological policies, without refit."""
import argparse,gc
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .dense_forward import read,materialize
from .benchmark_forward_selectors import load_npz,load_models
from .forward_selectors import prediction,choose_blocks,calibration,apply_policy
from .selective_rag import error_metrics,rows
from .paired_history_increment import ARMS,pack,with_base,predict,summary,mask_modalities
from .benchmark_paired_history_increment import (PHASES,implementation,baseline_policy,carrier,
    utility_inputs,final_policy,load_increment,baseline_model)
from .benchmark_unified_residual import predict as base_predict


def close(a,b):np.testing.assert_allclose(a,b,rtol=5e-5,atol=5e-5)


def fixed_direct_baseline(out,data,saved,anchor,meta,device):
    """Secondary fixed B=M for EVERY dataset; never replace the primary.

    Mode0 experts and utility policies were already fitted/locked for both base
    modes before TEST. This evaluates that fixed branch without selecting a
    baseline using TEST or re-calibrating any parameter.
    """
    declaration=out.parent/'SECONDARY_DIRECT_MODE_PROTOCOL.json'
    if not declaration.exists():return None
    declared=read(declaration)
    if declared['mode']!=0 or declared['primary_unchanged'] is not True:raise ValueError('Changed secondary definition')
    base=saved['moirai_direct'];policies=read(out/'policy.json')['increments'];outputs={};routing={};scores={}
    for arm in ARMS:
        key=f'{arm}__0';raw=saved[key+'__raw'];nets,normal=load_models(out/'utilities'/key,device)
        x=utility_inputs(data,base,raw,anchor,meta['period'],arm,0)
        u=prediction('summary_mlp',nets,normal,x,device)
        outputs[arm],routing[arm]=apply_policy(carrier(data,base),raw,u,policies[key]);scores[arm]=u
        np.testing.assert_array_equal(outputs[arm].reshape(-1,12,8)[~routing[arm]],base.reshape(-1,12,8)[~routing[arm]])
        del nets,x;gc.collect()
    np.savez_compressed(out/'direct_mode_predictions.npz',**outputs)
    np.savez_compressed(out/'direct_mode_routing.npz',**routing)
    np.savez_compressed(out/'direct_mode_utility.npz',**scores)
    metrics={scope:{arm:error_metrics(rows(data,np.flatnonzero(saved[scope+'_mask'])),v[saved[scope+'_mask']],anchor)
        for arm,v in outputs.items()} for scope in ['dense','legacy']}
    result=dict(dataset=out.name,complete=True,mode=0,primary_unchanged=True,metrics=metrics,
        declaration_sha256=sha256(declaration),contract_sha256=sha256(out/'contract.json'),
        predictions_sha256=sha256(out/'direct_mode_predictions.npz'),routing_sha256=sha256(out/'direct_mode_routing.npz'),
        utility_sha256=sha256(out/'direct_mode_utility.npz'),
        adoption={a:float(routing[a][saved['dense_mask']].mean()) for a in ARMS})
    write_json(out/'direct_mode.json',result);return result


def provenance(out):
    protocol=read(out/'protocol.json');contract=read(out/'contract.json');forward=Path(protocol['forward'])
    assert protocol['code']==contract['code']==implementation()
    for file,key in [('protocol.json','protocol_sha256'),('policy.json','policy_sha256'),('validation_utility.npz','utility_sha256')]:
        assert sha256(out/file)==contract[key],file
    assert sha256(forward/'selector_contract.json')==protocol['upstream_contract_sha256']
    pairs=0
    for phase in PHASES:
        assert sha256(out/(phase+'_experts.json'))==contract['phases'][phase]
        record=read(out/(phase+'_experts.json'));p=record['provenance']
        assert sha256(forward/(phase+'_train_index.npz'))==p['index_sha256']
        assert sha256(forward/(phase+'_experts.json'))==p['baseline_manifest_sha256']
        assert sha256(out/(phase+'_predictions.npz'))==record['prediction_sha256']
        cut=read(forward/(phase+'_train_index.json'))['cutoff']
        assert sha256(forward/f'prefix_{cut}.json')==p['stats_sha256']
        for arm,r in record['models'].items():
            assert sha256(out/r['file'])==r['sha256'];assert len(r['losses'])==16
            assert r['last_fit_label_end']<=cut<=record['first_held_start']
        for split in ['train','held']:
            with np.load(forward/(phase+'_'+split+'_index.npz'),allow_pickle=False) as f:
                for name in ['selected_starts','random_selected_starts']:
                    assert np.all(f[name]+192<=f['start'][:,None]);pairs+=f[name].size
                if split=='train':assert np.all(f['start']+192<=cut)
                else:assert np.all(f['start']>=cut)
    for key,expected in contract['models'].items():
        path=out/'utilities'/(key+'.json');assert sha256(path)==expected
        record=read(path);assert sha256(path.parent/record['file'])==record['sha256']
        assert len(record['records'])==3 and all(len(r['losses'])==20 for r in record['records'])
    return dict(case_pairs=pairs,all_prefix_boundaries=True,all_fitted_artifacts_hashed=True)


def verify(out,device='cuda'):
    integrity=provenance(out);protocol=read(out/'protocol.json');forward=Path(protocol['forward']);meta=read(forward/'preparation.json')
    stats=read(forward/f'prefix_{meta["layout"]["cuts"][-1]}.json');anchor=read(forward/f'prefix_{meta["layout"]["cuts"][0]}.json')
    contract=read(out/'contract.json');policy=read(out/'policy.json');val=materialize(meta['source'],forward,'final_held')
    vp=load_npz(out/'final_predictions.npz');vu=load_npz(out/'validation_utility.npz');blocks=choose_blocks(val,meta['layout']['validation_end'])
    assert baseline_policy(val,vp['adapted'],anchor,blocks)==policy['baseline']
    assert [np.unique(val['start'][i]).tolist() for i in blocks]==contract['blocks']
    for arm in ARMS:
        for mode in [0,1]:
            key=f'{arm}__{mode}';base=val['raw_quantiles'][:,4] if mode==0 else vp['adapted']
            nets,normal=load_models(out/'utilities'/key,device)
            x=utility_inputs(val,base,vp[key],anchor,meta['period'],arm,mode)
            u=prediction('summary_mlp',nets,normal,x,device);close(u,vu[key])
            choices=[('utility',calibration(carrier(val,base),vp[key],vu[key],'summary_mlp',anchor,blocks)),
                ('static',calibration(carrier(val,base),vp[key],None,'static',anchor,blocks))]
            name,p=min(choices,key=lambda z:z[1]['calibration_objective'])
            assert dict(p,selected_kind=name)==policy['increments'][key]
            del x,nets,u;gc.collect()
    del val,vp,vu;gc.collect()
    run=read(out/'run.json')
    for file,key in [('contract.json','contract_sha256'),('predictions.npz','predictions_sha256'),('routing.npz','routing_sha256'),
        ('test_utility.npz','test_utility_sha256'),('interventions.npz','interventions_sha256')]:assert sha256(out/file)==run[key]
    d=materialize(meta['source'],forward,'test');saved=load_npz(out/'predictions.npz');routing=load_npz(out/'routing.npz');utilities=load_npz(out/'test_utility.npz')
    for k in ['sid','start','raw_y']:np.testing.assert_array_equal(saved[k],d[k])
    np.testing.assert_array_equal(saved['moirai_direct'],d['raw_quantiles'][:,4])
    model,_=baseline_model(forward,'final',device);adapted=base_predict(model,d,stats,meta['period'],'query_only',device);close(adapted,saved['query_adapter_raw']);del model
    bases=[saved['moirai_direct'],saved['query_adapter_raw']];mode,_,base_ok=final_policy(out,'paired');base=bases[mode] if base_ok else bases[0]
    np.testing.assert_array_equal(base,saved['selected_base']);records=[]
    for arm in ARMS:
        path=out/read(out/'final_experts.json')['models'][arm]['file'];model=load_increment(path,device)
        p,scale=pack(d,stats,meta['period'],arm)
        poisoned=dict(d,raw_y=np.full_like(d['raw_y'],np.nan));other,_=pack(poisoned,stats,meta['period'],arm)
        for k in p:np.testing.assert_array_equal(p[k],other[k])
        for bm,b in enumerate(bases):
            actual=predict(model,with_base(p,b,d,scale,bm),b,scale,device);close(actual,saved[f'{arm}__{bm}__raw'])
        raw=saved[f'{arm}__{mode}__raw'];nets,normal=load_models(out/'utilities'/f'{arm}__{mode}',device)
        x=utility_inputs(d,bases[mode],raw,anchor,meta['period'],arm,mode)
        px=utility_inputs(poisoned,bases[mode],raw,anchor,meta['period'],arm,mode)
        np.testing.assert_array_equal(x['summary'],px['summary'])
        utility=prediction('summary_mlp',nets,normal,x,device);close(utility,utilities[arm])
        _,poli,_=final_policy(out,arm);forecast,mask=apply_policy(carrier(d,base),raw,utilities[arm],poli)
        np.testing.assert_array_equal(forecast,saved[arm]);np.testing.assert_array_equal(mask,routing[arm])
        np.testing.assert_array_equal(forecast.reshape(-1,12,8)[~mask],base.reshape(-1,12,8)[~mask])
        native,nmask=apply_policy(carrier(d,base),raw,utility,poli)
        if device.startswith('cuda'):
            np.testing.assert_array_equal(nmask,mask);np.testing.assert_array_equal(native,forecast)
        records.append(dict(arm=arm,utility_max_difference=float(abs(utility-utilities[arm]).max()),
            native_route_differences=int((nmask!=mask).sum()),native_forecast_max_difference=float(abs(native-forecast).max()),
            poisoned_current_future_invariant=True,exact_base_fallback=True))
        del model,p,other,nets,x,px,utility;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
        print('PAIRED VERIFIED',out.name,arm,device,flush=True)
    for scope in ['dense','legacy']:
        ids=np.flatnonzero(saved[scope+'_mask']);truth=rows(d,ids)
        for key,values in run['metrics'][scope].items():
            actual=error_metrics(truth,saved[key][ids],anchor)
            for k,v in values.items():np.testing.assert_allclose(actual[k],v,rtol=1e-12,atol=1e-12)
    with np.load(out/'interventions.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['empty__raw'],bases[mode]);np.testing.assert_array_equal(f['empty__fixed_route'],base)
        history_change=float(abs(f['history_deranged__raw']-saved[f'paired__{mode}__raw']).max())
    secondary=fixed_direct_baseline(out,d,saved,anchor,meta,device)
    return dict(dataset=out.name,complete=True,device=device,records=records,integrity=integrity,secondary_direct=secondary,
        supplied_case_history_shape=[10,96],supplied_continuation_shape=[10,96],history_sequence_derangement_max_change=history_change,
        caveat='Changed history proves dependence, not necessarily improved accuracy. Report independent and fixed-route controls.')


def deploy(out,device='cuda',live=False):
    from .query_paired_history_increment import PairedHistoryPredictor
    from .query_forward_selectors import ObservedStore
    protocol=read(out/'protocol.json');forward=Path(protocol['forward']);saved=load_npz(out/'predictions.npz')
    index=load_npz(forward/'test_index.npz');serving=PairedHistoryPredictor(out,device,adapter=None if live else False);records=[]
    sids=sorted({0,int(saved['sid'].max())//2,int(saved['sid'].max())}|({313} if out.name=='electricity' else set())|({8} if out.name=='weather' else set()))
    did_live=False
    try:
        for sid in sids:
            ids=np.flatnonzero(saved['sid']==sid)
            with np.load(forward/'foundation_test'/f'{sid}.npz',allow_pickle=False) as f:cache={int(t):(q,p) for t,q,p in zip(f['starts'],f['quantiles'],f['probe'])}
            for row in ids[np.unique([0,len(ids)-1])]:
                start=int(saved['start'][row]);x=serving.store.window(sid,start,96);q,probe=cache[start]
                result=serving.predict_cached(x,sid,start,q[None],probe[None])
                np.testing.assert_array_equal(result['selected_starts'],index['selected_starts'][row]);close(result['forecast'],saved['paired'][row])
                original=serving.store;serving.store=ObservedStore(original,start+96)
                try:limited=serving.predict_cached(x,sid,start,q[None],probe[None])
                finally:serving.store=original
                np.testing.assert_array_equal(result['forecast'],limited['forecast'])
                records.append(dict(sid=sid,start=start,actual_retrieval=True,actual_history_values_checked=True,
                    physically_future_limited=True,max_forecast_difference=float(abs(np.asarray(result['forecast'])-saved['paired'][row]).max())))
                if live:
                    actual=serving.predict(x,sid,start);close(actual['moirai'],q[4]);close(actual['forecast'],saved['paired'][row]);live=False;did_live=True
                print('PAIRED DEPLOYED',out.name,sid,start,flush=True)
    finally:serving.close()
    return dict(dataset=out.name,complete=True,records=records,live_foundation_checked=did_live)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--deploy',action='store_true');p.add_argument('--live',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    root=Path(a.root);file=root/('deployment_verification.json' if a.deploy else ('gpu_verification.json' if a.device.startswith('cuda') else 'cpu_verification.json'))
    records={r['dataset']:r for r in read(file)['records']} if file.exists() else {}
    for name in a.datasets:
        records[name]=deploy(root/name,a.device,a.live) if a.deploy else verify(root/name,a.device)
        write_json(file,dict(complete=True,records=list(records.values())))
