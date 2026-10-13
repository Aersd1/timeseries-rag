"""Bound retrieval nudges and choose experts on calibration, then confirm once."""
import numpy as np

BETAS=[.1,.25,.5]


def bound_delta(delta,quantiles):
    q=np.asarray(quantiles,dtype=float)
    radius=.25*np.maximum(q[:,-1]-q[:,0],0)
    return np.clip(np.asarray(delta,dtype=float),-radius,radius).astype(np.float32)


def choose_policy(target,deltas,sids,starts,require_mae=False):
    """Only validation targets; final block rejects the single chosen expert."""
    target=np.asarray(target,dtype=float);policies={}
    for sid in np.unique(sids):
        ids=np.flatnonzero(sids==sid);ids=ids[np.argsort(starts[ids])];parts=np.array_split(ids,3)
        default=dict(family='direct',beta=0.,reason='insufficient stable calibration evidence')
        if min(map(len,parts))<3:policies[str(int(sid))]=default;continue
        fitted=np.r_[parts[0],parts[1]];base=np.mean(target[fitted]**2,axis=1)
        eligible=[]
        for family,values in deltas.items():
            for beta in BETAS:
                loss=np.mean((target[fitted]-beta*values[fitted])**2,axis=1);gain=base-loss
                rng=np.random.default_rng(20261013)
                samples=gain[rng.integers(0,len(gain),size=(2000,len(gain)))].mean(1)
                lower=float(np.quantile(samples,.05));wins=float(np.mean(gain>0))
                mae_ok=(not require_mae or np.mean(abs(target[fitted]-beta*values[fitted]))<np.mean(abs(target[fitted])))
                if lower>.01*base.mean() and wins>=.75 and mae_ok:
                    eligible.append((float(loss.mean()),family,beta,lower,wins))
        if not eligible:policies[str(int(sid))]=default;continue
        _,family,beta,lower,wins=min(eligible)
        held=parts[2];baseline=np.mean(target[held]**2,axis=1)
        gain=baseline-np.mean((target[held]-beta*deltas[family][held])**2,axis=1)
        mae_ok=(not require_mae or np.mean(abs(target[held]-beta*deltas[family][held]))<np.mean(abs(target[held])))
        confirmed=bool(gain.mean()>0 and np.mean(gain>0)>=.75 and mae_ok)
        policies[str(int(sid))]=dict(family=family if confirmed else 'direct',beta=beta if confirmed else 0.,
            proposed_family=family,proposed_beta=beta,calibration_lower_gain=lower,calibration_win_fraction=wins,
            final_gain=float(gain.mean()),final_win_fraction=float(np.mean(gain>0)),
            reason='single expert confirmed' if confirmed else 'final validation block rejected the selected expert')
    return policies


def apply_policy(policy,deltas,sids):
    shape=next(iter(deltas.values())).shape;result=np.zeros(shape,dtype=np.float32)
    for sid in np.unique(sids):
        decision=policy[str(int(sid))];ids=sids==sid
        if decision['family']!='direct':result[ids]=decision['beta']*deltas[decision['family']][ids]
    return result
