"""History-only multivariate state and completed analogue continuations."""
import numpy as np
from .residual_adapter import features


def context_plan(store, count=4):
    """Choose related channels using the common initial memory prefix only."""
    plan={}
    for source in sorted({s['source'] for s in store.series}):
        group=[s for s in store.series if s['source']==source]
        end=min(s['memory_end'] for s in group)
        rows=[]
        for s in group:
            x=np.asarray(store.values(s['sid'])[:end],dtype=float)
            z=(np.nan_to_num(x,nan=s['memory_mean'])-s['memory_mean'])/max(s['memory_std'],1e-6)
            rows.append(z-z.mean())
        z=np.stack(rows);norm=np.sqrt(np.sum(z*z,axis=1));corr=(z@z.T)/np.maximum(norm[:,None]*norm[None,:],1e-12)
        for i,s in enumerate(group):
            ordered=sorted((j for j in range(len(group)) if j!=i),key=lambda j:(-abs(corr[i,j]),group[j]['sid']))[:count]
            plan[str(s['sid'])]=[group[j]['sid'] for j in ordered]
    return plan


def feature(store,data,ids,s,period,plan,use_rag=True):
    """No query targets read here; other channels stop at the history boundary."""
    x=np.asarray(data['raw_x'][ids],dtype=float);base=data['base'][ids]
    starts=data['start'][ids];std=data['std'][ids];n=len(ids)
    rag=(data['weights'][ids,:,None]*data['errors'][ids]).sum(1)
    f=features(x,base,data['raw_quantiles'][ids],rag,starts,std,np.full(n,s['memory_mean']),period,False)
    lead=np.arange(1,97)[None,:]/96
    phase=2*np.pi*(starts[:,None]+96+np.arange(96)[None,:])/(7*(period or 1))
    extra=[np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)]
    t=np.arange(96)-47.5
    for sid in plan[str(s['sid'])]:
        other=store.series[sid];values=store.values(sid)
        histories=np.stack([np.array(values[int(p):int(p)+96],dtype=float) for p in starts])
        valid=np.isfinite(histories);coverage=valid.mean(1)
        histories=np.where(valid,histories,other['memory_mean'])
        z=np.clip((histories-other['memory_mean'])/max(other['memory_std'],1e-6),-20,20)
        state=[z.mean(1),(z[:,-1]-z.mean(1)),z.std(1),coverage]
        extra.extend(np.broadcast_to(v[:,None],(n,96)) for v in state)
        extra.append((z@t)[:,None]/(t@t)*95*lead)
    if use_rag:
        analogues=[]
        for row,i in enumerate(ids):
            old=data['selected_starts'][i].astype(int)
            if np.any(old+192>starts[row]) or np.any(old+192>s['memory_end']):
                raise ValueError('Analogue must be complete in initial memory, before query history')
            past=np.stack([store.window(s['sid'],p,96) for p in old]).astype(float)
            future=np.stack([store.window(s['sid'],p+96,96) for p in old]).astype(float)
            # Transfer change relative to the last16 past points; retain query scale.
            ratio=np.clip(x[row].std()/np.maximum(past.std(1),max(s['memory_std']*.1,1e-6)),.25,4)
            path=x[row,-16:].mean()+(future-past[:,-16:].mean(1,keepdims=True))*ratio[:,None]
            weights=data['weights'][i];mean=weights@path
            disagreement=np.sqrt(weights@((path-mean)**2))
            analogues.append(np.stack([(mean-base[row])/std[row],disagreement/std[row]],axis=-1))
        a=np.asarray(analogues);innovation=a[:,:,0]
        smooth=np.repeat(innovation.reshape(n,6,16).mean(2),16,axis=1)
        error_smooth=np.repeat(rag.reshape(n,6,16).mean(2),16,axis=1)
        extra.extend([rag,error_smooth,np.broadcast_to(rag.mean(1,keepdims=True),(n,96)),
            innovation,smooth,np.broadcast_to(innovation.mean(1,keepdims=True),(n,96)),a[:,:,1],innovation/(1+a[:,:,1])])
    return np.concatenate([f,np.stack(extra,axis=-1).astype(np.float32)],axis=-1)
