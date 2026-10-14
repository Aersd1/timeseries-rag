"""Confidence-weighted case detail and validation-only horizon routing."""
import numpy as np
from .safe_error_policy import BETAS,bound_delta


def block_mean(path):
    x=np.asarray(path,dtype=float)
    if x.shape[-1]!=96:raise ValueError('Expected horizon96')
    return np.repeat(x.reshape(*x.shape[:-1],6,16).mean(-1),16,axis=-1)


def detail_paths(encoded,cases,weights,starts):
    """No query outcome: preserve encoder block means; discount duplicate events."""
    encoded=np.asarray(encoded,dtype=float);cases=np.asarray(cases,dtype=float)
    w=np.asarray(weights,dtype=float);starts=np.asarray(starts)
    if cases.shape!=(5,96) or w.shape!=(5,) or starts.shape!=(5,) or np.any(w<0) or w.sum()<=0:
        raise ValueError('Expected five finite weighted cases')
    if not np.isfinite(cases).all() or not np.isfinite(encoded).all():raise ValueError('Nonfinite paths')
    w=w/w.sum();detail=cases-block_mean(cases);mean=w@detail
    variance=w@((detail-mean)**2)
    energy=(mean**2).reshape(6,16).mean(1);noise=variance.reshape(6,16).mean(1)
    overlap=np.maximum(0,1-abs(starts[:,None]-starts[None,:])/192.)
    effective=1/float(w@overlap@w)
    confidence=energy/(energy+noise+1e-12)*np.clip((effective-1)/4,0,1)
    adjustment=np.repeat(confidence,16)*(mean-(encoded-block_mean(encoded)))
    return {f'detail_{g}':encoded+g*adjustment for g in [.25,.5,1.]},confidence,effective


def component_experts(components,weights,starts):
    paths,confidence,effective=detail_paths(components['encoded_path'],components['cases'],weights,starts)
    experts=dict(components['deltas']);std=components['std'];base=components['base'];q=components['quantiles'][None]
    for kind,path in paths.items():experts[kind]=bound_delta((path-base)[None],q)[0]/std
    return experts,confidence,effective


def correction(decision,experts):
    family=decision.get('family','direct');beta=decision.get('beta',0.)
    return np.zeros_like(next(iter(experts.values()))) if family=='direct' else beta*experts[family]


def choose_segments(target,experts,sids,starts,legacy,disabled):
    """First two time blocks select six corrections; last block only rejects."""
    target=np.asarray(target,dtype=float);policies={}
    for sid in np.unique(sids):
        key=str(int(sid));rows=np.flatnonzero(sids==sid);rows=rows[np.argsort(starts[rows])];parts=np.array_split(np.arange(len(rows)),3)
        local_target=target[rows];local_experts={k:v[rows] for k,v in experts.items()}
        old=legacy[key];anchor=dict(family=old.get('proposed_family',old['family']),beta=old.get('proposed_beta',old['beta']))
        policy=dict(accepted=False,anchor=anchor,fallback=old,segments=[],reason='no stable component improvement')
        if min(map(len,parts))<3 or np.any(disabled[rows]):policies[key]=policy;continue
        anchor_delta=correction(anchor,local_experts);fallback_delta=correction(old,local_experts)
        assembled=anchor_delta.copy();changed=0
        for block in range(6):
            sl=slice(block*16,(block+1)*16);options=[]
            for family in ['direct']+list(experts):
                for beta in ([0.] if family=='direct' else BETAS):
                    cand=correction(dict(family=family,beta=beta),local_experts)
                    # Fixed shrinkage: small validation cohorts cannot justify full switches.
                    trial=.5*anchor_delta+.5*cand;gains=[];valid=True
                    for ids in parts[:2]:
                        original=local_target[ids,sl]-anchor_delta[ids,sl];error=local_target[ids,sl]-trial[ids,sl]
                        gain=np.mean(original**2,axis=1)-np.mean(error**2,axis=1)
                        if gain.mean()<=.01*np.mean(original**2) or np.mean(gain>0)<.75 or np.mean(abs(error))>=np.mean(abs(original)):
                            valid=False;break
                        gains.extend(gain.tolist())
                    if valid:options.append((-float(np.mean(gains)),family,beta))
            if options:
                _,family,beta=min(options);decision=dict(family=family,beta=beta);changed+=1
                assembled[:,sl]=(.5*anchor_delta+.5*correction(decision,local_experts))[:,sl]
            else:decision=dict(family=anchor['family'],beta=anchor['beta'])
            policy['segments'].append(decision)
        held=parts[2];err=local_target[held]-assembled[held]
        def confirms(reference):
            olderr=local_target[held]-reference[held];gain=np.mean(olderr**2,axis=1)-np.mean(err**2,axis=1)
            return bool(gain.mean()>0 and np.mean(gain>0)>=.75 and np.mean(abs(err))<np.mean(abs(olderr)))
        policy['accepted']=bool(changed and confirms(anchor_delta) and confirms(fallback_delta))
        policy['changed_blocks']=changed;policy['reason']='composite confirmed once' if policy['accepted'] else 'composite rejected; keep prior router'
        policies[key]=policy
    return policies


def apply_segments(policies,experts,sids):
    result=np.zeros_like(next(iter(experts.values())))
    for sid in np.unique(sids):
        policy=policies[str(int(sid))];rows=sids==sid
        local={k:v[rows] for k,v in experts.items()}
        if not policy['accepted']:result[rows]=correction(policy['fallback'],local);continue
        anchor=correction(policy['anchor'],local)
        output=result[rows].copy()
        for block,decision in enumerate(policy['segments']):
            sl=slice(block*16,(block+1)*16)
            output[:,sl]=(.5*anchor+.5*correction(decision,local))[:,sl]
        result[rows]=output
    return result
