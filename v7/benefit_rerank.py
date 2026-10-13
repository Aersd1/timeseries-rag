"""History-only candidate utility and independently guarded calibration."""
import numpy as np
from .trend_rag import decompose,normalize

BETAS=[0.,.1,.25,.5]
RIDGES=[.01,.1,1.,10.]


def blocks(x):
    return np.asarray(x,dtype=np.float64).reshape(*np.asarray(x).shape[:-1],6,16).mean(-1)


def pair_features(history,base,query_start,candidate_starts,candidate_histories,
        candidate_residual_histories,values,future_scores,history_scores,memory_mean,memory_std,period,window):
    """Query truth is deliberately absent from this signature."""
    x=np.asarray(history,dtype=float);past=np.asarray(candidate_histories,dtype=float)
    e=np.asarray(values,dtype=float);count=len(past);std=max(float(memory_std),1e-6)
    if x.shape!=(96,) or past.shape!=(count,96) or e.shape!=(count,96) or not np.isfinite(e).all():
        raise ValueError('Invalid utility inputs')
    if np.any(np.asarray(candidate_starts)+192>query_start):raise ValueError('Candidate overlaps query')
    _,r=decompose(x,window);qn,_,_=normalize(r,max(std*.01,1e-6))
    q=blocks(qn);c=blocks(candidate_residual_histories);v=blocks(e)
    b=blocks((np.asarray(base)-x[-1])/std)
    def expand(z):return np.full(count,z,dtype=float)
    fs=np.asarray(future_scores);hs=np.asarray(history_scores)
    qphase=2*np.pi*query_start/period;cphase=2*np.pi*np.asarray(candidate_starts)/period
    scalars=np.stack([fs/max(float(np.median(fs)),1e-6),hs/max(float(np.median(hs)),1e-6),
        expand((x.mean()-memory_mean)/std),(past.mean(1)-memory_mean)/std,
        expand(x.std()/std),past.std(1)/std,(past.mean(1)-x.mean())/std,
        ((past[:,-1]-past[:,0])-(x[-1]-x[0]))/std,
        np.log1p((query_start-np.asarray(candidate_starts)-192)/192),
        np.mean(e,axis=1),np.sqrt(np.mean(e*e,axis=1)),
        expand(np.sin(qphase)),expand(np.cos(qphase)),np.sin(cphase),np.cos(cphase),
        np.sin(qphase-cphase),np.cos(qphase-cphase)],axis=1)
    return np.concatenate([scalars,c,np.repeat(q[None],count,axis=0),v,
        np.repeat(b[None],count,axis=0),v*q,v*b,c*q],axis=1).astype(np.float32)


def fit_ridge(features,labels,penalty):
    x=np.asarray(features,dtype=float);y=np.asarray(labels,dtype=float)
    mean=x.mean(0);scale=np.maximum(x.std(0),.1)
    z=np.c_[np.ones(len(x)),(x-mean)/scale]
    bounds=np.quantile(y,[.01,.99]);y=np.clip(y,*bounds)
    regularizer=np.eye(z.shape[1])*penalty;regularizer[0,0]=0.
    coef=np.linalg.solve(z.T@z/len(z)+regularizer,z.T@y/len(z))
    return dict(mean=mean,scale=scale,coef=coef,penalty=float(penalty),label_bounds=bounds)


def covariance(state,features):
    x=(np.asarray(features,dtype=float)-state['mean'])/state['scale']
    return state['coef'][0]+x@state['coef'][1:]


def select_candidates(predicted_covariance,values,prior_scores,starts,sid):
    """Rank estimated loss reduction, preserving compatibility as a small prior."""
    cov=np.asarray(predicted_covariance,dtype=float);e=np.asarray(values,dtype=float)
    gain=.5*cov-.0625*np.mean(e*e,axis=1)  # Expected MSE reduction at beta=.25.
    scale=max(float(np.median(abs(gain))),1e-4)
    score=gain-.02*scale*np.asarray(prior_scores)
    chosen=[];remaining=set(range(len(e)))
    for _ in range(5):
        def key(i):
            overlap=max((max(0.,1-abs(float(starts[i]-starts[j]))/192) for j in chosen),default=0.)
            return (-(score[i]-.1*scale*overlap),prior_scores[i],starts[i])
        i=min(remaining,key=key);chosen.append(i);remaining.remove(i)
    chosen=np.asarray(chosen,dtype=int)
    logits=score[chosen]/scale;w=np.exp(np.clip(logits-logits.max(),-50,0));w/=w.sum()
    correction=w@e[chosen]
    expected_cov=float(w@cov[chosen]);energy=float(np.mean(correction**2))
    # Expected utility is a regression estimate, not a calibrated correctness probability.
    permitted=.5*expected_cov-.0625*energy>0
    return dict(correction=correction.astype(np.float32) if permitted else np.zeros(96,dtype=np.float32),
        selected=chosen,weights=w.astype(np.float32),gain=gain.astype(np.float32),permitted=bool(permitted))


def guarded_policy(target,correction,sids,starts):
    """Choose beta on first two validation blocks, confirm once on untouched third.

    The third block can only reject a coefficient, never choose another one.
    Every variable uses its own chronological blocks. No test argument exists.
    """
    target=np.asarray(target,dtype=float);correction=np.asarray(correction,dtype=float)
    decisions={}
    for sid in np.unique(sids):
        ids=np.flatnonzero(np.asarray(sids)==sid);ids=ids[np.argsort(np.asarray(starts)[ids])]
        parts=np.array_split(ids,3)
        if min(map(len,parts))<3:
            decisions[str(int(sid))]=dict(beta=0.,reason='insufficient validation blocks');continue
        base=np.array([np.mean(target[p]**2) for p in parts])
        table={str(b):[float(np.mean((target[p]-b*correction[p])**2)) for p in parts] for b in BETAS}
        eligible=[b for b in BETAS if b>0 and all(table[str(b)][k]<base[k] for k in [0,1])
            and np.mean(table[str(b)][:2])<=.99*np.mean(base[:2])]
        proposed=min(eligible,key=lambda b:(np.mean(table[str(b)][:2]),b)) if eligible else 0.
        confirmed=proposed>0 and table[str(proposed)][2]<base[2]
        adopted=proposed if confirmed else 0.
        decisions[str(int(sid))]=dict(beta=float(adopted),proposed_beta=float(proposed),
            reason='confirmed on final validation block' if confirmed else 'validation benefit not stable',
            blocks=[dict(first_start=int(starts[p[0]]),last_start=int(starts[p[-1]]),queries=len(p)) for p in parts],
            baseline_losses=base.tolist(),candidate_losses=table)
    return decisions


def apply_policy(policy,correction,sids):
    beta=np.array([policy[str(int(s))]['beta'] for s in sids],dtype=np.float32)
    return beta[:,None]*np.asarray(correction,dtype=np.float32)
