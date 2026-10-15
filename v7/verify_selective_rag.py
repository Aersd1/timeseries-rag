"""Replay the selector, its temporal inputs, policies and every TEST output."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .prepare_unified_residual import read
from .benchmark_unified_residual import trace
from .benchmark_selective_rag import load_npz,verified_source,frozen_predictions,arm_data,fit_static
from .selective_rag import (ARMS,SETTINGS,UtilityNet,rows,chronological_split,options,
    selector_features,score_selector,route,error_metrics,calibrate_router)


def verify(folder,device):
    run=read(folder/'run.json');contract=read(folder/'contract.json');selection=read(folder/'selection.json');policy=read(folder/'policy.json')
    checks=[('contract.json',run['contract_sha256']),('selection.json',contract['selection_sha256']),
        ('policy.json',contract['policy_sha256']),('validation_scores.npz',contract['validation_scores_sha256']),
        ('predictions.npz',run['predictions_sha256']),('routing.npz',run['routing_sha256'])]
    for file,expected in checks:
        if sha256(folder/file)!=expected:raise ValueError('Changed '+file)
    for file,expected in contract['code_sha256'].items():
        if sha256('v7/'+file)!=expected:raise ValueError('Changed experiment code '+file)
    source=Path(selection['source']);assert verified_source(source)==selection['source_hashes']
    meta=read(source/'preparation.json');stats=read(source/'stats.json');val=trace(source,'validation');test=trace(source,'test')
    fit,cal,confirm=chronological_split(val,selection['train_end'],selection['test_start'])
    for key,ids in [('fit',fit),('calibration',cal),('confirmation',confirm)]:
        assert np.unique(val['start'][ids]).tolist()==selection['split'][key]['starts']
    assert val['start'][confirm].max()+192<=test['start'].min()
    casepairs=0
    for data in [val,test]:
        for key in ['selected_starts','random_selected_starts']:
            assert np.all(data[key]+192<=data['start'][:,None]);casepairs+=data[key].size
    vp=frozen_predictions(source,'validation');tp=frozen_predictions(source,'test');vs=load_npz(folder/'validation_scores.npz');routing=load_npz(folder/'routing.npz');saved=load_npz(folder/'predictions.npz')
    for key in ['sid','start','raw_y','std']:np.testing.assert_array_equal(saved[key],test[key])
    records=[]
    for modelrec in selection['models']:
        arm=modelrec['arm'];assert arm in ARMS
        assert sha256(folder/modelrec['file'])==modelrec['sha256']
        assert len(modelrec['training'])==3 and all(len(r['losses'])==SETTINGS['epochs'] for r in modelrec['training'])
        checkpoint=torch.load(folder/modelrec['file'],map_location='cpu',weights_only=True);normal=checkpoint['normal'];models=[]
        for state in checkpoint['models']:
            model=UtilityNet(normal['input_width'],SETTINGS['width']);model.load_state_dict(state);models.append(model.to(device).eval())
        d=arm_data(val,arm);c=options(d,stats,meta['period'],arm,vp.get(arm));x=selector_features(d,c,stats,meta['period'],arm,d,c)
        # Normalization was fitted on selector TRAIN only, not later VAL/TEST.
        flat=x[fit].reshape(-1,x.shape[-1]).astype(float)
        np.testing.assert_allclose(normal['mean'],flat.mean(0),rtol=0,atol=1e-10)
        np.testing.assert_allclose(normal['scale'],np.maximum(flat.std(0),.05),rtol=0,atol=1e-10)
        score=score_selector(models,normal,x,device);np.testing.assert_allclose(score,vs[arm+'_scores'],rtol=3e-5,atol=3e-5)
        # Reconstruct policies from the SAVED pre-TEST scores to avoid a device
        # rounding tie changing the validation hyperparameter search.
        proposed=calibrate_router(d,c,vs[arm+'_scores'],stats,cal,confirm);proposed['static']=fit_static(d,c,stats,cal,confirm)
        assert proposed==policy[arm]
        td=arm_data(test,arm);tc=options(td,stats,meta['period'],arm,tp.get(arm));tx=selector_features(td,tc,stats,meta['period'],arm,d,c)
        ts=score_selector(models,normal,tx,device);np.testing.assert_allclose(ts,routing[arm+'_scores'],rtol=3e-5,atol=3e-5)
        # Poison every current TEST target, holding already observed historical
        # records fixed. Both candidates and selector inputs must be unchanged.
        poisoned=dict(td,raw_y=np.full_like(td['raw_y'],np.nan))
        np.testing.assert_array_equal(tc,options(poisoned,stats,meta['period'],arm,tp.get(arm)))
        np.testing.assert_array_equal(tx,selector_features(poisoned,tc,stats,meta['period'],arm,d,c))
        p=policy[arm];pred,mask,choice=route(test['raw_quantiles'][:,4],tc,ts,p['threshold'],p['strength'],p['enabled'])
        np.testing.assert_array_equal(mask,routing[arm+'_mask'])
        # Near-tied unused alternatives can exchange order across CPU/GPU.
        # Compare every ADOPTED action exactly, plus all scores numerically and
        # every final forecast exactly. A rejected choice has no forecast effect.
        np.testing.assert_array_equal(choice[mask],routing[arm+'_choice'][mask])
        unused_choice_differences=int(np.sum((choice!=routing[arm+'_choice'])&~mask))
        np.testing.assert_array_equal(pred,saved[arm])
        # Rejected patches remain bit-identical to the original foundation.
        np.testing.assert_array_equal(pred.reshape(-1,12,8)[~mask],test['raw_quantiles'][:,4].reshape(-1,12,8)[~mask])
        metric=error_metrics(test,pred,stats)
        for k,v in metric.items():np.testing.assert_allclose(v,run['results'][arm]['metrics'][k],rtol=1e-12,atol=1e-12)
        st=p['static'];sp=test['raw_quantiles'][:,4]+st['strength']*(tc.mean(2).reshape(-1,96)-test['raw_quantiles'][:,4]) if st['enabled'] else test['raw_quantiles'][:,4]
        np.testing.assert_array_equal(sp,saved[arm+'_static'])
        records.append(dict(arm=arm,queries=len(pred),poison_future_invariant=True,exact_fallback=True,
            unused_choice_differences=unused_choice_differences,score_max_difference=float(np.max(abs(ts-routing[arm+'_scores'])))))
        print('SELECTOR VERIFIED',folder.name,arm,device,flush=True)
        del models,x,tx,c,tc,flat
    return dict(dataset=folder.name,complete=True,device=device,case_pairs=casepairs,arms=records,contract_sha256=run['contract_sha256'])


def deployment_check(folder,device):
    from .query_selective_rag import SelectiveRagPredictor
    import gc
    selection=read(folder/'selection.json');source=Path(selection['source']);test=trace(source,'test');saved=load_npz(folder/'predictions.npz');routing=load_npz(folder/'routing.npz')
    sid=313 if folder.name=='electricity' else 0
    row=int(np.flatnonzero(test['sid']==sid)[0]);model=SelectiveRagPredictor(folder,device,adapter=False)
    try:
        actual=model.predict_cached(test['raw_x'][row],sid,int(test['start'][row]),test['raw_quantiles'][row:row+1],test['probe_quantiles'][row:row+1])
        np.testing.assert_array_equal(actual['selected_starts'],test['selected_starts'][row])
        np.testing.assert_array_equal(actual['adopted_patches'],routing['real_mask'][row])
        np.testing.assert_allclose(actual['forecast'],saved['real'][row],rtol=3e-5,atol=3e-5)
        return dict(dataset=folder.name,sid=sid,start=int(test['start'][row]),actual_retrieval_replayed=True,cached_foundation_forecasts=True,
            max_forecast_difference=float(np.max(abs(np.asarray(actual['forecast'])-saved['real'][row]))),adopted_patches=actual['adopted_patches'])
    finally:model.close();del model;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cpu');p.add_argument('--deploy',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    root=Path(a.root);path=root/('deployment_verification.json' if a.deploy else ('gpu_verification.json' if a.device.startswith('cuda') else 'verification.json'))
    existing=read(path)['records'] if path.exists() else [];records={v['dataset']:v for v in existing}
    for name in a.datasets:
        records[name]=deployment_check(root/name,a.device) if a.deploy else verify(root/name,a.device);write_json(path,dict(complete=True,records=list(records.values())))
