"""Learn the incremental value of examples after a purged base prediction."""
import numpy as np
from .residual_adapter import train
from .benefit_rerank import covariance,fit_ridge,pair_features
from .safe_error_policy import bound_delta
from .benchmark_state_recall import corrections

PENALTIES=[.1,1.,10.,100.]


def forward_base(features,target,starts,raw_base,quantiles,scale,beta):
    """Outer held queries never choose the model or enter its fit."""
    edges=np.quantile(starts,[.5,.75,1.]);rows=[];forecasts=[];folds=[]
    for j in range(2):
        held=np.flatnonzero((starts>=edges[j])&(starts<=edges[j+1] if j==1 else starts<edges[j+1]))
        cutoff=int(starts[held].min());fit=np.flatnonzero(starts+192<=cutoff)
        if len(fit)<4:raise ValueError('Insufficient completed base teacher queries')
        try:state,selection=train(features[fit],target[fit],starts[fit])
        except ValueError as error:
            if 'Insufficient purged' not in str(error):raise
            # Small earlier prefixes cannot support an inner search; use a fixed,
            # strongly regularized prior rather than looking at outer outcomes.
            state=fit_ridge(features[fit].reshape(-1,features.shape[-1]),target[fit].reshape(-1),100.)
            selection=dict(penalty=100.,fallback='too few inner-fold queries; predeclared fixed penalty',folds=[])
        delta=bound_delta(covariance(state,features[held])*scale[held,None],quantiles[held])
        forecasts.extend(raw_base[held]+beta*delta);rows.extend(held)
        folds.append(dict(fitted=len(fit),held=len(held),last_fitted_episode_end=int((starts[fit]+192).max()),
            held_first_start=cutoff,nested_selection=selection))
    return np.asarray(rows,dtype=int),np.asarray(forecasts,dtype=np.float32),folds


def candidate_innovations(history,base,past,future,std):
    """Map known old continuations to query state; query truth is absent."""
    past=np.asarray(past,dtype=float);future=np.asarray(future,dtype=float)
    ratio=np.clip(np.std(history)/np.maximum(past.std(1),.1*max(float(std),1e-6)),.25,4.)
    mapped=np.mean(history[-16:])+(future-past[:,-16:].mean(1,keepdims=True))*ratio[:,None]
    return ((mapped-base)/max(float(std),1e-6)).astype(np.float32)


def build_table(data,ids,base,s,library_starts,past,future,residual_past,period,window):
    arrays={}
    for j,i in enumerate(ids):
        starts=data['candidate_starts'][i]
        if np.any(starts+192>min(int(data['start'][i]),s['memory_end'])):raise ValueError('Self/overlapping or unobserved retrieval')
        p=np.searchsorted(library_starts,starts);np.testing.assert_array_equal(library_starts[p],starts)
        std=data['metric_std'][i]
        values=candidate_innovations(data['raw_x'][i],base[j],past[p],future[p],std)
        fs=data['future_scores'][i];hs=data['history_scores'][i]
        f=pair_features(data['raw_x'][i],base[j],int(data['start'][i]),starts,past[p],residual_past[p],values,
            fs,hs,s['memory_mean'],std,period or 1,window)
        item=dict(features=f,values=values,starts=starts,prior=.8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6),
            start=data['start'][i],base=base[j],std=std)
        for key,value in item.items():arrays.setdefault(key,[]).append(value)
    return {key:np.asarray(value) for key,value in arrays.items()}


def train_incremental(table,target,quantiles):
    """Train covariances against the base model's OUTER held residual."""
    starts=table['start'];edges=np.quantile(starts,[.5,.75,1.]);losses={};folds=[]
    labels=np.mean(table['values']*target[:,None,:],axis=-1)
    for j in range(2):
        held=np.flatnonzero((starts>=edges[j])&(starts<=edges[j+1] if j==1 else starts<edges[j+1]))
        cutoff=int(starts[held].min());fit=np.flatnonzero(starts+192<=cutoff)
        if len(fit)<4 or len(held)<4:
            state=fit_ridge(table['features'].reshape(-1,table['features'].shape[-1]),labels.reshape(-1),100.)
            state['coef']*=0
            return state,dict(penalty=100.,losses={},folds=folds,
                fallback='insufficient independent utility folds; zero incremental expert',target='Outer held base residual')
        sub={key:value[held] for key,value in table.items()}
        for penalty in PENALTIES:
            state=fit_ridge(table['features'][fit].reshape(-1,table['features'].shape[-1]),labels[fit].reshape(-1),penalty)
            r=corrections(state,sub)['correction']
            bounded=bound_delta(r*sub['std'][:,None],quantiles[held])/sub['std'][:,None]
            losses.setdefault(str(penalty),[]).append(float(np.mean((target[held]-.25*bounded)**2)))
        folds.append(dict(fitted=len(fit),held=len(held),last_fitted_episode_end=int((starts[fit]+192).max()),held_first_start=cutoff))
    penalty=min(PENALTIES,key=lambda p:(np.mean(losses[str(p)]),-p))
    state=fit_ridge(table['features'].reshape(-1,table['features'].shape[-1]),labels.reshape(-1),penalty)
    return state,dict(penalty=penalty,losses=losses,folds=folds,target='Remaining error of nested forward base forecasts, never in-sample base errors')
