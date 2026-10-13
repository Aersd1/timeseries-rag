"""Compact supervised forecast-error adapter with explicit future calendar phase."""
import numpy as np
from .benefit_rerank import fit_ridge,covariance

PENALTIES=[.1,1.,10.,100.]


def features(history,forecast,quantiles,rag_error,starts,std,memory_mean,period,use_rag=True):
    x=np.asarray(history,dtype=float);b=np.asarray(forecast,dtype=float);s=np.maximum(np.asarray(std,dtype=float),1e-6)[:,None]
    n=len(x);lead=np.arange(1,97,dtype=float)[None,:]/96
    if x.shape!=(n,96) or b.shape!=(n,96) or np.asarray(quantiles).shape!=(n,9,96):raise ValueError('Expected paired96 inputs')
    t=np.arange(96)-47.5;slope=(x@t)/(t@t)*95
    phase=2*np.pi*(np.asarray(starts)[:,None]+96+np.arange(96)[None,:])/(period or 1)
    def broadcast(v):return np.broadcast_to(v,(n,96))
    columns=[(b-x.mean(1,keepdims=True))/s,(b-x[:,-1,None])/s,
        broadcast(slope[:,None]/s*lead),broadcast(x.std(1,keepdims=True)/s),
        broadcast((x.mean(1)-np.asarray(memory_mean))[:,None]/s),
        (np.asarray(quantiles)[:,-1]-np.asarray(quantiles)[:,0])/s,
        broadcast(lead),np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)]
    if use_rag:
        e=np.asarray(rag_error,dtype=float)
        if e.shape!=(n,96):raise ValueError('Expected known historical error aggregate')
        smooth=np.repeat(e.reshape(n,6,16).mean(2),16,axis=1)
        columns.extend([e,smooth,broadcast(e.mean(1,keepdims=True))])
    return np.stack(columns,axis=-1).astype(np.float32)


def train(feature,target,starts):
    starts=np.asarray(starts);edges=np.quantile(starts,[.5,.75,1.]);losses={};folds=[]
    for j in range(2):
        held=np.flatnonzero((starts>=edges[j])&(starts<=edges[j+1] if j==1 else starts<edges[j+1]))
        first=int(starts[held].min());fit=np.flatnonzero(starts+192<=first)
        if len(fit)<4 or len(held)<4:raise ValueError('Insufficient purged per-variable training folds')
        folds.append(dict(train_queries=len(fit),held_queries=len(held),train_last_episode_end=int((starts[fit]+192).max()),held_first_start=first))
        for penalty in PENALTIES:
            state=fit_ridge(feature[fit].reshape(-1,feature.shape[-1]),target[fit].reshape(-1),penalty)
            predicted=covariance(state,feature[held])
            losses.setdefault(str(penalty),[]).append(float(np.mean((predicted-target[held])**2)))
    penalty=min(PENALTIES,key=lambda p:(np.mean(losses[str(p)]),-p))
    return fit_ridge(feature.reshape(-1,feature.shape[-1]),target.reshape(-1),penalty),dict(
        penalty=penalty,losses=losses,folds=folds,selection='two purged forward folds inside training only')
