"""A self-trained multivariate history-to-future-state encoder, without Moirai latents."""
import numpy as np

PENALTIES=[.1,1.,10.,100.]


def history_features(history,starts,memory_mean,memory_std,period,weekly=False):
    """History[N,96,C] only; retain physical state and relative dynamics."""
    x=np.asarray(history,dtype=float);mean=np.asarray(memory_mean);sd=np.maximum(memory_std,1e-6)
    if x.ndim!=3 or x.shape[1]!=96:raise ValueError('Expected multivariate96 history')
    valid=np.isfinite(x);x=np.where(valid,x,mean)
    level=x[:,-16:].mean(1);scale=np.maximum(x.std(1),.1*sd)
    shape=((x-level[:,None,:])/scale[:,None,:]).reshape(len(x),6,16,x.shape[-1]).mean(2).reshape(len(x),-1)
    phase=2*np.pi*(np.asarray(starts)+96)/period
    annual=phase/365.2425
    state=np.c_[(level-mean)/sd,np.log(scale/sd),valid.mean(1)]
    features=np.c_[shape,state,np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),
        np.sin(annual),np.cos(annual),np.sin(annual)*np.sin(phase),np.cos(annual)*np.sin(phase)].astype(np.float32)
    if weekly:features=np.c_[features,np.sin(phase/7),np.cos(phase/7),np.sin(2*phase/7),np.cos(2*phase/7)].astype(np.float32)
    return features


def future_state(history,future,memory_std):
    """A supervision/prototype encoder for already known future blocks."""
    x=np.asarray(history,dtype=float);y=np.asarray(future,dtype=float)
    if x.shape!=y.shape or x.shape[1]!=96 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite paired96 multivariate episodes required')
    scale=np.maximum(x.std(1),.1*np.maximum(memory_std,1e-6))
    changes=(y-x[:,-16:].mean(1)[:,None,:])/scale[:,None,:]
    return changes.reshape(len(x),6,16,x.shape[-1]).mean(2).reshape(len(x),-1).astype(np.float32)


def prototype_state(history,future,memory_mean,memory_std):
    """Missing sibling channels are masked, never fabricated as known futures."""
    x=np.asarray(history,dtype=float);y=np.asarray(future,dtype=float)
    past_valid=np.isfinite(x).all(1);future_valid=np.isfinite(y).reshape(len(y),6,16,y.shape[-1]).all(2)
    mask=(future_valid&past_valid[:,None,:]).reshape(len(y),-1)
    safe_x=np.where(np.isfinite(x),x,np.asarray(memory_mean));safe_y=np.where(np.isfinite(y),y,np.asarray(memory_mean))
    code=future_state(safe_x,safe_y,memory_std);code[~mask]=0
    return dict(values=code,mask=mask)


def fit(features,labels,penalty):
    x=np.asarray(features,dtype=float);y=np.asarray(labels,dtype=float)
    mean=x.mean(0);scale=np.maximum(x.std(0),.1);z=np.c_[np.ones(len(x)),(x-mean)/scale]
    bounds=np.quantile(y,[.01,.99],axis=0);y=np.clip(y,*bounds)
    reg=np.eye(z.shape[1])*penalty;reg[0,0]=0
    coef=np.linalg.solve(z.T@z/len(z)+reg,z.T@y/len(z))
    return dict(mean=mean,scale=scale,coef=coef,penalty=penalty,label_bounds=bounds)


def predict(state,features):
    return state['coef'][0]+((features-state['mean'])/state['scale'])@state['coef'][1:]


def train(features,labels,starts):
    edges=np.quantile(starts,[.5,.75,1.]);losses={};traces={};folds=[]
    for j in range(2):
        held=np.flatnonzero((starts>=edges[j])&(starts<=edges[j+1] if j==1 else starts<edges[j+1]))
        cutoff=starts[held].min();ids=np.flatnonzero(starts+192<=cutoff)
        if len(ids)<16 or len(held)<8:raise ValueError('Insufficient independent encoder folds')
        folds.append(dict(fitted=len(ids),held=len(held),last_fitted_episode_end=int((starts[ids]+192).max()),held_first_start=int(cutoff)))
        for penalty in PENALTIES:
            state=fit(features[ids],labels[ids],penalty);p=predict(state,features[held]);residual=(p-labels[held])**2
            losses.setdefault(str(penalty),[]).append(float(residual.mean()))
            traces.setdefault(penalty,[]).append((held,p,residual))
    penalty=min(PENALTIES,key=lambda p:(np.mean(losses[str(p)]),-p))
    chosen=traces[penalty];rows=np.concatenate([t[0] for t in chosen]);oof=np.concatenate([t[1] for t in chosen])
    error=np.concatenate([t[2] for t in chosen]).mean(0)
    precision=1/np.maximum(error,.1);precision=np.clip(precision/max(float(np.median(precision)),1e-6),.05,20.)
    state=fit(features,labels,penalty);state['precision']=precision
    return state,dict(penalty=penalty,losses=losses,folds=folds,queries=len(features)),rows,oof
