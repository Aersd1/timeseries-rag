"""Train a small query-dependent reliability gate over frozen Moirai/A forecasts."""
import math
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import sha256,write_json
from v6.common import save_torch

FEATURES=['history_error','future_pinball','continuation_disagreement','moirai_interval_width',
    'model_analog_difference','weight_entropy','episode_overlap','learned_distance',
    'history_trend','history_end_level','history_scale_ratio','future_score_spread']


def quality_features(x,floor,quantiles,levels,mapped,hits,span,history_std):
    """No query future argument. Candidates contain known historical continuations."""
    x=np.asarray(x,dtype=float); scale=max(float(x.std()),float(floor)); weights=np.array([h['weight'] for h in hits])
    levels=np.asarray(levels); median=int(np.flatnonzero(np.isclose(levels,.5))[0])
    direct=np.asarray(quantiles,dtype=float)[median]
    mapped=np.asarray(mapped,dtype=float); analog=weights@mapped
    variance=(weights[:,None]*(mapped-analog)**2).sum(0).mean()/scale**2
    width=(np.max(quantiles,axis=0)-np.min(quantiles,axis=0)).mean()/scale
    entropy=-float((weights*np.log(weights.clip(1e-8))).sum())/math.log(max(len(weights),2))
    overlap=np.mean([max(0.,1-abs(a['start']-b['start'])/span) if a['sid']==b['sid'] else 0.
        for i,a in enumerate(hits) for b in hits[i+1:]]) if len(hits)>1 else 0.
    values=[np.log1p(np.mean([h['history_nmse'] for h in hits])),
        np.log1p(np.mean([h['future_score'] for h in hits])),np.log1p(variance),np.log1p(width),
        np.log1p(np.mean((analog-direct)**2)/scale**2),entropy,overlap,
        np.log1p(np.mean([h['distance'] for h in hits])),(x[-1]-x[0])/scale,(x[-1]-x.mean())/scale,
        np.log1p(scale/max(float(history_std),1e-6)),np.log1p(np.std([h['future_score'] for h in hits]))]
    result=np.asarray(values,dtype=np.float32)
    if not np.isfinite(result).all(): raise FloatingPointError('Invalid reliability features')
    return result


class ReliabilityGate(nn.Module):
    def __init__(self,initial_alpha=.5):
        super().__init__(); self.net=nn.Sequential(nn.Linear(len(FEATURES),8),nn.Tanh(),nn.Linear(8,1))
        nn.init.zeros_(self.net[-1].weight)
        a=min(max(initial_alpha,.01),.99); nn.init.constant_(self.net[-1].bias,math.log(a/(1-a)))

    def forward(self,features): return self.net(features).sigmoid().squeeze(-1)


def examples(folder,source,split,policy):
    from .benchmark_a100 import load_cache,rank_cache
    run,data=load_cache(folder/split); c=run['config']; catalog=json.loads((source/'store/catalog.json').read_text())
    vectors=[]; predictions=[]; stds=[]
    for i,x in enumerate(data['x']):
        result=rank_cache(data,run,i,100,policy['diversity_weight']); hits=result['hits']
        lookup={int(s):j for j,s in enumerate(data['starts'][i])}; ids=[lookup[h['start']] for h in hits]
        std=catalog['series'][int(data['sid'][i])]['memory_std']
        vectors.append(quality_features(x,data['floor'][i],data['quantiles'][i],run['quantile_levels'],
            data['mapped'][i,ids],hits,c['length']+max(c['horizons']),std))
        predictions.append(result['prediction']); stds.append(max(std,1e-6))
    return dict(features=torch.as_tensor(np.stack(vectors)),analog=torch.as_tensor(np.stack(predictions),dtype=torch.float32),
        direct=torch.as_tensor(data['direct'],dtype=torch.float32),target=torch.as_tensor(data['y']),
        std=torch.as_tensor(stds,dtype=torch.float32),run=run)


def forecast_loss(alpha,items,horizons):
    prediction=(1-alpha[:,None])*items['direct']+alpha[:,None]*items['analog']
    error=((prediction-items['target'])/items['std'][:,None])**2
    return sum(error[:,:h].mean() for h in horizons)/len(horizons)


def train_gate(folder,source,policy,epochs=200):
    train=examples(folder,source,'train',policy); val=examples(folder,source,'validation',policy)
    c=train['run']['config']; initial=policy['analogue_alpha'][folder.name]
    mean=train['features'].mean(0); std=train['features'].std(0,correction=0).clamp_min(.1)
    tx=(train['features']-mean)/std; vx=(val['features']-mean)/std
    torch.manual_seed(c['seed']); model=ReliabilityGate(initial); optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    with torch.no_grad(): best=float(forecast_loss(model(vx),val,c['horizons']))
    best_state={n:p.detach().clone() for n,p in model.state_dict().items()}; best_epoch=0; history=[]
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); alpha=model(tx)
        loss=forecast_loss(alpha,train,c['horizons'])+.01*((alpha-initial)**2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        model.eval()
        with torch.no_grad(): score=float(forecast_loss(model(vx),val,c['horizons']))
        if score<best:
            best,best_epoch=score,epoch; best_state={n:p.detach().clone() for n,p in model.state_dict().items()}
        history.append(dict(epoch=epoch,train_loss=float(loss.detach()),validation_nmse=score))
    constant=float(forecast_loss(torch.full((len(vx),),initial),val,c['horizons']))
    adopted='gate' if best<constant else 'constant'
    state=dict(family='a100_reliability_gate',model=best_state,feature_names=FEATURES,mean=mean,std=std,
        diversity_weight=policy['diversity_weight'],constant_alpha=initial,adopted=adopted,config=c,
        data_id=train['run']['data_id'],checkpoint_sha256=train['run']['checkpoint_sha256'],
        index_sha256=train['run']['index_sha256'],policy_sha256=sha256(folder.parent/'policy.json'),
        training_cache_sha256=sha256(folder/'train/cache.npz'),validation_cache_sha256=sha256(folder/'validation/cache.npz'),
        epochs=epochs,best_epoch=best_epoch,validation_gate_nmse=best,validation_constant_nmse=constant)
    save_torch(folder/'quality_gate.pt',state)
    write_json(folder/'gate_history.json',history)
    write_json(folder/'gate_run.json',dict(complete=True,adopted=adopted,best_epoch=best_epoch,epochs=epochs,
        validation_gate_nmse=best,validation_constant_nmse=constant,parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_sha256=sha256(folder/'quality_gate.pt')))
    print('GATE',folder.name,adopted,best_epoch,best,constant,flush=True)


def gate_predict(state,features):
    if state['feature_names']!=FEATURES: raise ValueError('Feature architecture changed')
    model=ReliabilityGate(); model.load_state_dict(state['model']); model.eval()
    features=torch.as_tensor(features,dtype=torch.float32)
    with torch.no_grad(): alpha=model((features-state['mean'])/state['std']).numpy()
    deployed=alpha if state['adopted']=='gate' else np.full(alpha.shape,state['constant_alpha'])
    return alpha,deployed
