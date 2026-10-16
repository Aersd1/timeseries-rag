"""Untrained memory-access controls with observed same-channel case pairs."""
import numpy as np

CONTEXT=96+10*192


def diverse_indices(starts,order,excluded=(),k=10):
    chosen=[];excluded=set(map(int,excluded))
    for i in order:
        t=int(starts[i])
        if t not in excluded and all(abs(t-int(starts[j]))>=192 for j in chosen):
            chosen.append(int(i))
            if len(chosen)==k:return np.asarray(chosen,dtype=int)
    raise ValueError('Insufficient completed diverse historical cases')


def select_clock(starts,query_start,limit,period,mode):
    starts=np.asarray(starts,dtype=np.int64)
    eligible=np.flatnonzero(starts+192<=min(query_start,limit))
    if mode=='recent':order=eligible[np.argsort(-starts[eligible],kind='stable')]
    elif mode=='periodic':
        p=max(1,int(round(period or 1)));r=(query_start-starts[eligible])%p
        phase=np.minimum(r,p-r)
        order=eligible[np.lexsort((-starts[eligible],phase))]
    else:raise ValueError('Unknown clock selection')
    return starts[diverse_indices(starts,order)]


def align_pack(history,past,future,starts):
    """Fit affine alignment from histories ONLY; preserve complete pair order.

    The backbone sees a single concatenated series, not special demonstration
    tokens. No phase shifting, learned fusion, or post-forecast blend is used.
    """
    x=np.asarray(history,dtype=float);p=np.asarray(past,dtype=float);f=np.asarray(future,dtype=float)
    if x.shape!=(96,) or p.shape!=(10,96) or f.shape!=p.shape:
        raise ValueError('Expected current96 plus ten complete96/96 pairs')
    if not np.isfinite(x).all() or not np.isfinite(p).all() or not np.isfinite(f).all():
        raise ValueError('Nonfinite memory values')
    qs=x.std();ps=p.std(1);den=ps**2+(qs/5.)**2
    slope=np.divide(qs*ps,den,out=np.zeros(10),where=den>1e-12)
    intercept=x.mean()-slope*p.mean(1)
    episodes=np.concatenate([p,f],axis=1)*slope[:,None]+intercept[:,None]
    order=np.argsort(starts,kind='stable')
    return np.r_[episodes[order].ravel(),x].astype(np.float32)


def history_controls(values,starts,history,query_start,limit):
    """Ten same-channel controls: five raw matches plus five shape matches."""
    starts=np.asarray(starts,dtype=np.int64)
    eligible=np.flatnonzero(starts+192<=min(query_start,limit));lib=starts[eligible]
    past=np.asarray(values[lib[:,None]+np.arange(96)],dtype=float)
    x=np.asarray(history,dtype=float)
    raw=np.mean((past-x)**2,1)
    first=diverse_indices(lib,np.lexsort((lib,raw)),k=5)
    q=(x-x.mean())/max(x.std(),1e-6)
    z=(past-past.mean(1,keepdims=True))/np.maximum(past.std(1,keepdims=True),1e-6)
    shape=np.mean((z-q)**2,1)
    second=diverse_indices(lib,np.lexsort((lib,shape)),excluded=lib[first],k=5)
    return np.r_[lib[first],lib[second]]
