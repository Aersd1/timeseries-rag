"""Fixed, history-conditioned 5+5 interventions and observable benefit proxies.

No current query target is accepted by the feature/transfer API. Historical
continuations are complete observations. No new forecast network is trained.
"""
import numpy as np

ALPHA=.25
FEATURES=('history_distance','continuation_dispersion','state_gap','log_age',
    'old_case_loo_gain','old_case_loo_valid','external_case_fraction',
    'redundancy_alignment','correction_size','recent_drift','base_disagreement',
    'observed_probe_error','short_quantile_width')
QUERY_FEATURES=(9,10,11,12)
ARMS=('real','matched_random','deranged_1','deranged_2','deranged_3','deranged_4')


def moments(x,sd):
    x=np.asarray(x,dtype=float)
    return x.mean(-1),np.maximum(x.std(-1),.1*np.asarray(sd))


def normalized_keys(x,past,sd):
    qm,qs=moments(x,sd);pm,ps=moments(past,np.asarray(sd)[:,None])
    q=(x-qm[:,None])/qs[:,None];p=(past-pm[:,:,None])/ps[:,:,None]
    shape=((p-q[:,None])**2).mean(2)
    level=(pm-qm[:,None])/np.asarray(sd)[:,None]
    amp=np.log(ps/qs[:,None])
    trend=(p[:,:,-16:].mean(2)-p[:,:,:16].mean(2))-(q[:,-16:].mean(1)-q[:,:16].mean(1))[:,None]
    distance=shape+.1*level**2+.1*amp**2
    weights=np.empty_like(distance)
    for j in (0,5):
        z=-np.clip(distance[:,j:j+5],0,80);z-=z.max(1,keepdims=True)
        w=np.exp(z);weights[:,j:j+5]=.5*w/w.sum(1,keepdims=True)
    raw_ps=np.asarray(past,dtype=float).std(2)
    den=raw_ps**2+(np.asarray(x,dtype=float).std(1)[:,None]/5.)**2
    slope=np.divide(np.asarray(x,dtype=float).std(1)[:,None]*raw_ps,den,
        out=np.zeros_like(den),where=den>1e-12)
    return qm,qs,pm,ps,p,shape,level,amp,trend,weights,slope*ps


def old_case_check(past,future,prior,starts,sd):
    """Leave every temporally overlapping episode out of a support self-check.

    All labels are known NOW. This is neither chronological forecasting at the
    held episode nor OOF validation of the inherited TRAIN-fitted retriever.
    """
    _,ps=moments(past,np.asarray(sd)[:,None]);pm=np.asarray(past,dtype=float).mean(2)
    keys=(past-pm[:,:,None])/ps[:,:,None]
    residual=(np.asarray(future,dtype=float)-prior)/ps[:,:,None]
    gains=[];valid=[]
    for j in range(10):
        mask=abs(starts-starts[:,j,None])>=192
        distance=((keys-keys[:,j,None])**2).mean(2)
        w=np.exp(-np.clip(distance,0,80))*mask
        w/=np.maximum(w.sum(1,keepdims=True),1e-30)
        change=ALPHA*np.einsum('nk,nkh->nh',w,residual)
        target=residual[:,j]
        gains.append((target**2-(target-change)**2).mean(1))
        valid.append(mask.sum(1)>=3)
    gains=np.stack(gains,1);valid=np.stack(valid,1)
    return (gains*valid).sum(1)/np.maximum(valid.sum(1),1),valid.mean(1)


def transfer(x,quantiles,probe,sid,start,past,future,prior,starts,train_std,
             base,length,arm='real',with_features=True):
    """Two fixed fusion instruments, both explicitly condition on case HISTORY."""
    x=np.asarray(x,dtype=float);past=np.asarray(past,dtype=float)
    starts=np.asarray(starts,dtype=np.int64);start=np.asarray(start,dtype=np.int64)
    if past.shape!=(len(x),10,96) or np.shape(future)!=past.shape or np.shape(prior)!=past.shape:
        raise ValueError('Need ten complete96/96 cases')
    if np.any(starts+192>start[:,None]):raise ValueError('Case overlaps query96/target96')
    sd=np.maximum(np.asarray(train_std,dtype=float),1e-6)
    qm,qs,pm,ps,keys,shape,level,amp,trend,w,amplitude=normalized_keys(x,past,sd)
    normalized_residual=(np.asarray(future,dtype=float)-prior)/ps[:,:,None]
    normalized_future=(np.asarray(future,dtype=float)-pm[:,:,None])/ps[:,:,None]
    if arm.startswith('deranged_'):
        shift=int(arm.rsplit('_',1)[1])
        if shift not in (1,2,3,4):raise ValueError('Require a derangement of each five-case group')
        normalized_residual=normalized_residual.copy();normalized_future=normalized_future.copy()
        for j in (0,5):
            normalized_residual[:,j:j+5]=np.roll(normalized_residual[:,j:j+5],shift,axis=1)
            normalized_future[:,j:j+5]=np.roll(normalized_future[:,j:j+5],shift,axis=1)
    elif arm not in ('real','matched_random'):raise ValueError('Unknown intervention')
    aligned_residual=amplitude[:,:,None]*normalized_residual
    aligned_future=qm[:,None,None]+amplitude[:,:,None]*normalized_future
    raw=np.einsum('nk,nkh->nh',w,aligned_residual)
    candidate=np.einsum('nk,nkh->nh',w,aligned_future)
    radius=2*qs[:,None]
    delta=np.clip(raw,-radius,radius)
    target_delta=np.clip(candidate-np.asarray(base,dtype=float),-radius,radius)
    output=dict(residual=np.asarray(base,dtype=float)+ALPHA*delta,
                continuation=np.asarray(base,dtype=float)+ALPHA*target_delta,
                correction=ALPHA*delta)
    if not with_features:return output
    dispersion=np.sqrt(np.einsum('nk,nkh->n',w,(aligned_future-candidate[:,None])**2)/96)/qs
    loo,valid=old_case_check(past,future,prior,starts,sd)
    shift=np.asarray(base,dtype=float)-np.asarray(quantiles)[:,4]
    dot=(delta*shift).sum(1);den=np.linalg.norm(delta,axis=1)*np.linalg.norm(shift,axis=1)
    alignment=np.divide(dot,den,out=np.zeros(len(x)),where=den>1e-12)
    # Fraction of each complete192 episode NOT already visible in baseline context.
    left=start+96-int(length);right=start+96
    overlap=np.maximum(0,np.minimum(starts+192,right[:,None])-np.maximum(starts,left[:,None]))
    external=(w*(1-overlap/192)).sum(1)
    features=np.c_[(w*shape).sum(1),dispersion,
        (w*(abs(level)+abs(amp)+abs(trend))).sum(1),
        (w*np.log1p(np.maximum(start[:,None]-starts-192,0)/192)).sum(1),
        loo,valid,external,alignment,np.sqrt((delta**2).mean(1))/qs,
        abs(x[:,-32:].mean(1)-x[:,:32].mean(1))/qs,
        np.sqrt((shift**2).mean(1))/qs,
        np.sqrt(((x[:,64:]-np.asarray(probe)[:,4])**2).mean(1))/qs,
        np.maximum(np.asarray(quantiles)[:,8]-np.asarray(quantiles)[:,0],0).mean(1)/qs]
    if not np.isfinite(features).all():raise FloatingPointError('Nonfinite observable feature')
    output['features']=features
    return output


def improvement_identity(y,base,pred):
    """Ex post loss identity, NOT a feature or deployable oracle gate."""
    e=np.asarray(y,dtype=float)-base;d=np.asarray(pred,dtype=float)-base
    alignment=2*(e*d).mean(1);energy=(d*d).mean(1)
    actual=(e*e).mean(1)-((e-d)**2).mean(1)
    np.testing.assert_allclose(actual,alignment-energy,rtol=1e-8,atol=1e-7)
    return alignment,energy,actual


def fit_sign_model(x,y,weights,columns):
    """A fixed small diagnostic classifier; never changes the transfer network."""
    from sklearn.linear_model import LogisticRegression
    cols=np.asarray(columns,dtype=int);x=np.asarray(x)[:,cols]
    mean=np.average(x,axis=0,weights=weights)
    std=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),.05)
    z=np.clip((x-mean)/std,-8,8)
    if len(np.unique(y))<2:
        coef=np.zeros(len(cols));intercept=float(20 if y[0] else -20)
    else:
        model=LogisticRegression(C=.1,max_iter=500,solver='lbfgs',random_state=20261019)
        model.fit(z,y,sample_weight=np.asarray(weights)*len(y)/np.sum(weights))
        coef=model.coef_[0];intercept=float(model.intercept_[0])
    return dict(columns=cols.tolist(),mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),intercept=intercept)


def sign_probability(model,x):
    z=np.clip((np.asarray(x)[:,model['columns']]-model['mean'])/model['std'],-8,8)
    logits=np.clip(z@np.asarray(model['coef'])+model['intercept'],-40,40)
    return 1/(1+np.exp(-logits))
