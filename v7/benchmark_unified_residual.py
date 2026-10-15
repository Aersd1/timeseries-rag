"""Independent, identically budgeted 5+5 residual ablations; VAL-only gates."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .unified_residual import MODES,features,UnifiedResidualFusion,aggregate_guard
from .aligned_case_fusion import tensor_inputs,forecast
from .benchmark_case_fusion import calibrate,apply
from .benchmark_state_recall import metric,interval
from .prepare_unified_residual import prepare_split,read

SETTINGS=dict(width=32,seeds=[13,41,73],epochs=16,batch_size=512,learning_rate=.001,
    weight_decay=.001,raw_mse_weight=.5,normalized_mse_weight=.5,
    loss='Equal TRAIN-scaled raw MSE and per-variable standardized MSE + 0.02 weighted MAE + 0.001 correction energy',
    selection='Exactly 16 epochs for every arm/seed; no TEST-selected epoch, seed or architecture',
    reference='Frozen Moirai96-to96; no dataset-fitted reference predictions',
    input_variants=MODES,case_count=10)


def trace(folder,split):
    info=read(folder/f'{split}_info.json')
    if sha256(folder/f'{split}.npz')!=info['sha256']:raise ValueError('Changed paired features')
    with np.load(folder/f'{split}.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def train_one(data,stats,period,mode,seed,device,rows=None):
    torch.manual_seed(seed);np.random.seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    model=UnifiedResidualFusion(len(stats['std']),SETTINGS['width']).to(device)
    f,_,scale=features(data,stats,period,mode)
    if rows is None:rows=np.arange(len(scale))
    f=tensor_inputs({k:v[rows] for k,v in f.items()},device);scale=scale[rows]
    sid=data['sid'][rows];std=np.asarray(stats['std'])[sid];global_variance=float(np.mean(np.asarray(stats['std'])**2))
    weights=.5*(scale/std)**2+.5*scale**2/max(global_variance,1e-12)
    weight=torch.as_tensor(weights,dtype=torch.float32,device=device)
    target=torch.as_tensor((data['raw_y'][rows]-data['raw_quantiles'][rows,4])/scale[:,None],dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay'])
    rng=np.random.default_rng(seed);losses=[]
    for epoch in range(1,SETTINGS['epochs']+1):
        order=rng.permutation(len(rows));total=0.;model.train()
        for begin in range(0,len(order),SETTINGS['batch_size']):
            ids=torch.as_tensor(order[begin:begin+SETTINGS['batch_size']],device=device)
            delta=model(**{k:v[ids] for k,v in f.items()});e=delta-target[ids]
            loss=((e.square()+.02*e.abs()).mean(1)*weight[ids]).mean()+.001*delta.square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(ids)
        losses.append(total/len(rows))
    model.eval();return model,losses


def predict(model,data,stats,period,mode,device):
    f,_,scale=features(data,stats,period,mode)
    return forecast(model,tensor_inputs(f,device),data,scale)


def confirmed(data,raw,stats,base):
    policy=calibrate(data,raw,base);proposal=apply(policy,data,raw,base)
    guard=aggregate_guard(data,proposal,base,stats['std'])
    return (proposal if guard['accepted'] else base.copy()),dict(policy=policy,aggregate_guard=guard)


def apply_confirmed(data,raw,base,policy):
    return apply(policy['policy'],data,raw,base) if policy['aggregate_guard']['accepted'] else base.copy()


def fixed_transfer(data,stats,period):
    f,_,scale=features(data,stats,period,'real')
    return data['raw_quantiles'][:,4]+.25*np.clip(f['future'].mean(1),-f['radius'],f['radius'])*scale[:,None]


def experiment(folder,device,forward_check=True):
    if (folder/'run.json').exists():raise ValueError('Completed run is immutable')
    meta=read(folder/'preparation.json');stats=read(folder/'stats.json');period=meta['period']
    if sha256(folder/'stats.json')!=meta['stats_sha256']:raise ValueError('Changed stats')
    train=trace(folder,'train');val=trace(folder,'validation');models={};val_predictions={};selection=[];forward={}
    # A purged forward TRAIN holdout diagnoses the NEW adapter only. The inherited
    # TRAIN-fitted retriever/case selections are explicitly not claimed to be OOF.
    cutoff=int(np.quantile(np.unique(train['start']),.75));fit_rows=np.flatnonzero(train['start']+192<=cutoff)
    held=np.flatnonzero(train['start']>=cutoff);held_data={k:v[held] for k,v in train.items()}
    for mode in MODES:
        models[mode]=[];vp=[]
        for seed in SETTINGS['seeds']:
            if forward_check and seed==SETTINGS['seeds'][0]:
                # Historical-only normalization for this auxiliary forward check.
                fs=dict(stats);fs['mean']=[];fs['std']=[]
                from v5.data import Store
                store=Store(meta['store'])
                try:
                    for s in store.series:
                        values=np.asarray(store.values(s['sid'])[:cutoff]);values=values[np.isfinite(values)]
                        fs['mean'].append(float(values.mean()));fs['std'].append(max(float(values.std()),1e-6))
                finally:store.close()
                fm,_=train_one(train,fs,period,mode,seed,device,fit_rows)
                forward[mode]=metric(predict(fm,held_data,fs,period,mode,device),held_data);del fm
            model,losses=train_one(train,stats,period,mode,seed,device)
            path=folder/f'{mode}_{seed}.pt';torch.save(dict(state_dict=model.cpu().state_dict(),variables=len(stats['std']),width=SETTINGS['width']),path);model.to(device)
            models[mode].append(model);vp.append(predict(model,val,stats,period,mode,device))
            selection.append(dict(mode=mode,seed=seed,losses=losses,file=path.name,sha256=sha256(path),parameters=sum(p.numel() for p in model.parameters())))
            print('UNIFIED TRAIN',folder.name,mode,seed,'epochs',SETTINGS['epochs'],flush=True)
        val_predictions[mode]=np.mean(vp,axis=0).astype(np.float32)
    if len(set(r['parameters'] for r in selection))!=1:raise ValueError('Unmatched network capacity')
    base=val['raw_quantiles'][:,4];q,qp=confirmed(val,val_predictions['query_only'],stats,base)
    policies={'query_only':qp};own={};extra={}
    for mode in MODES[1:]:
        own[mode],policies['own_'+mode]=confirmed(val,val_predictions[mode],stats,base)
        extra[mode],policies['extra_'+mode]=confirmed(val,val_predictions[mode],stats,q)
    transfer=fixed_transfer(val,stats,period);_,policies['transfer_only']=confirmed(val,transfer,stats,base)
    write_json(folder/'policy.json',policies);write_json(folder/'selection.json',dict(settings=SETTINGS,models=selection,
        forward_check=dict(cutoff=cutoff,fitted=len(fit_rows),held=len(held),last_fitted_end=int((train['start'][fit_rows]+192).max()),metrics=forward,
            caveat='New adapter forward holdout only; inherited upstream TRAIN retrieval is not forward OOF')))
    np.savez_compressed(folder/'validation_predictions.npz',**val_predictions,query_confirmed=q,transfer_only=transfer)
    code=['unified_residual.py','prepare_unified_residual.py','benchmark_unified_residual.py']
    write_json(folder/'contract.json',dict(preparation_sha256=sha256(folder/'preparation.json'),stats_sha256=sha256(folder/'stats.json'),
        policy_sha256=sha256(folder/'policy.json'),selection_sha256=sha256(folder/'selection.json'),validation_sha256=sha256(folder/'validation_predictions.npz'),
        code_sha256={k:sha256('v7/'+k) for k in code},primary='query_confirmed + VAL-confirmed incremental real-case contribution',
        caveat='Previously explored TEST periods; no TEST-based per-dataset winner selection; Moirai frozen'))
    del train
    prepare_split(folder,'test');test=trace(folder,'test');forecasts=dict(moirai_direct=test['raw_quantiles'][:,4]);seed_outputs={}
    for mode in MODES:
        raw=[]
        for seed,model in zip(SETTINGS['seeds'],models[mode]):
            pred=predict(model,test,stats,period,mode,device);raw.append(pred);seed_outputs[f'{mode}_{seed}']=pred
        forecasts['raw_'+mode]=np.mean(raw,axis=0).astype(np.float32)
    base=forecasts['moirai_direct'];q=apply_confirmed(test,forecasts['raw_query_only'],base,policies['query_only']);forecasts['query_confirmed']=q
    for mode in MODES[1:]:
        forecasts['own_'+mode]=apply_confirmed(test,forecasts['raw_'+mode],base,policies['own_'+mode])
        forecasts['extra_'+mode]=apply_confirmed(test,forecasts['raw_'+mode],q,policies['extra_'+mode])
    forecasts['fixed_transfer']=fixed_transfer(test,stats,period)
    forecasts['confirmed_transfer']=apply_confirmed(test,forecasts['fixed_transfer'],base,policies['transfer_only'])
    with np.load(Path(meta['source'])/'predictions.npz',allow_pickle=False) as old:
        for key in ['sid','start','raw_y']:np.testing.assert_array_equal(test[key],old[key])
        for key in ['previous_previous_five','previous_fixed_increment_transport','previous_guarded_dual']:forecasts[key]=old[key].copy()
    truth={k:test[k] for k in ['sid','start','raw_y','std']}
    np.savez_compressed(folder/'predictions.npz',**forecasts,**truth);np.savez_compressed(folder/'seed_predictions.npz',**seed_outputs)
    metrics={k:metric(v,truth) for k,v in forecasts.items()}
    write_json(folder/'run.json',dict(complete=True,dataset=folder.name,queries=len(test['sid']),variables=len(stats['std']),
        metrics=metrics,individual_seed_metrics={k:metric(v,truth) for k,v in seed_outputs.items()},
        additional_rag_nmse_interval=interval(q,forecasts['extra_real'],truth),
        active_rag_variables=[int(s) for s,p in policies['extra_real']['policy'].items() if p['beta']>0] if policies['extra_real']['aggregate_guard']['accepted'] else [],
        contract_sha256=sha256(folder/'contract.json'),predictions_sha256=sha256(folder/'predictions.npz'),seed_predictions_sha256=sha256(folder/'seed_predictions.npz')))
    print('UNIFIED RESULT',folder.name,{k:metrics[k] for k in ['moirai_direct','query_confirmed','own_real','extra_real','extra_shuffled','extra_matched_random']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:experiment(Path(a.root)/name,a.device)
