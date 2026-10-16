"""Fast pilot: adapt a small residual predictor to actual history/future pairs.

The feature map is TRAIN-only PCA, not the proposed fully meta-trained encoder.
Every query gets its own regularized 9x12 head; Moirai stays frozen.
"""
import numpy as np


def describe(x, starts, sid, stats, period):
    x = np.asarray(x, dtype=float); sid = np.asarray(sid, dtype=int)
    sd = np.maximum(np.asarray(stats['memory_std'])[sid], 1e-6)
    center = x[:, -16:].mean(1); scale = np.maximum(x.std(1), .1*sd)
    h = (x-center[:, None])/scale[:, None]
    phase = 2*np.pi*(np.asarray(starts)+96)/(period or 1)
    state = np.stack([(center-np.asarray(stats['memory_mean'])[sid])/sd,
                      np.log(scale/sd), np.sin(phase), np.cos(phase),
                      np.sin(phase/7), np.cos(phase/7)], 1)
    return np.clip(np.c_[h, state], -20, 20), scale


def encode(model, x):
    z = np.clip((x-model['mean'])/model['std'], -8, 8)
    return np.c_[np.ones(len(x)), np.clip(z@model['components'], -8, 8)]


def compress(y): return np.asarray(y).reshape(-1, 12, 8).mean(2)


def expand(y):
    centers = np.arange(12)*8+3.5
    return np.stack([np.interp(np.arange(96), centers, row) for row in y])


def train_global(data, stats, period):
    x, scale = describe(data['raw_x'], data['start'], data['sid'], stats, period)
    mean=x.mean(0); std=np.maximum(x.std(0), .1); z=np.clip((x-mean)/std, -8, 8)
    eig, vec = np.linalg.eigh(z.T@z/len(z))
    order=np.argsort(eig)[-8:][::-1]
    model=dict(mean=mean, std=std, components=vec[:, order]/np.sqrt(np.maximum(eig[order], .1)))
    phi=encode(model, x)
    target=compress((data['raw_y']-data['raw_quantiles'][:, 4])/scale[:, None])
    reg=np.eye(phi.shape[1])*10.; reg[0,0]=1e-6
    model['prior']=np.linalg.solve(phi.T@phi+reg, phi.T@target)
    return model


def bounded(correction, q, scale):
    radius=np.minimum(.25*np.maximum(q[:,8]-q[:,0], 0), 2*scale[:,None])
    return np.clip(correction, -radius, radius)


def pair_features(model, data, stats, period, arm):
    """No current raw_y is read. Old targets are legal, completed observations."""
    random=arm=='random'; prefix='random_' if random else ''
    past=data[prefix+'case_past']; future=data[prefix+'case_future']
    oldbase=data[prefix+'case_baseline']; starts=data[prefix+'selected_starts']
    n,k,_=past.shape
    if k!=10 or np.any(starts+192>data['start'][:,None]):
        raise ValueError('Need ten complete, nonoverlapping-with-query cases')
    raw, scale=describe(past.reshape(-1,96), starts.ravel(), np.repeat(data['sid'],10), stats, period)
    phi=encode(model,raw).reshape(n,10,-1)
    target=compress((future-oldbase).reshape(-1,96)/scale[:,None]).reshape(n,10,12)
    if arm=='deranged':
        # Shuffle normalized residual VALUES after alignment, retain all history KEYS.
        target=target.copy()
        for row in range(n):
            shift=1+(int(data['start'][row])+int(data['sid'][row]))%4
            for j in (0,5):target[row,j:j+5]=np.roll(target[row,j:j+5],shift,axis=0)
    return phi,target,starts


def local_update(phi, target, prior, penalty, mask=None):
    if mask is None:mask=np.ones(phi.shape[:2])
    a=np.einsum('nki,nk,nkj->nij',phi,mask,phi)+penalty*np.eye(phi.shape[-1])[None]
    residual=target-phi@prior
    rhs=np.einsum('nki,nk,nkj->nij',phi,mask,residual)
    return np.linalg.solve(a,rhs)


def transfer_check(phi,target,starts,prior,penalty):
    """Leave an entire overlapping episode out; diagnostic uses ONLY old cases.

    All support labels are known at the current query. This is not a forward
    re-training of the global prior and must not be called full-pipeline OOF.
    """
    gains=[];valid=[]
    for j in range(10):
        mask=(abs(starts-starts[:,j,None])>=192).astype(float)
        update=local_update(phi,target,prior,penalty,mask)
        change=np.einsum('ni,nij->nj',phi[:,j],update)
        error=target[:,j]-phi[:,j]@prior
        gains.append(np.mean(error**2-(error-change)**2,1))
        valid.append(mask.sum(1)>=3)
    g=np.stack(gains,1);v=np.stack(valid,1)
    count=v.sum(1);mean=(g*v).sum(1)/np.maximum(count,1)
    wins=((g>0)&v).sum(1)/np.maximum(count,1)
    return (count>=3)&(mean>0)&(wins>=.5)


def forecasts(model,data,stats,period,penalties=(1.,10.,100.)):
    x,scale=describe(data['raw_x'],data['start'],data['sid'],stats,period)
    query=encode(model,x);q=data['raw_quantiles'];base=q[:,4]
    global_raw=base+bounded(expand(query@model['prior'])*scale[:,None],q,scale)
    outputs={'global_raw':global_raw};gates={}
    for arm in ('real','deranged','random'):
        phi,target,starts=pair_features(model,data,stats,period,arm)
        for penalty in penalties:
            update=local_update(phi,target,model['prior'],penalty)
            correction=query@model['prior']+np.einsum('ni,nij->nj',query,update)
            outputs[f'{arm}_{penalty}']=base+bounded(expand(correction)*scale[:,None],q,scale)
            if arm=='real':gates[str(penalty)]=transfer_check(phi,target,starts,model['prior'],penalty)
        if arm=='real':
            outputs['mean_transfer']=base+bounded(expand(target.mean(1))*scale[:,None],q,scale)
    return outputs,gates
