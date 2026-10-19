"""Same512/96, disjoint memory, and effective-df-matched residual probes."""
import math
from pathlib import Path
import numpy as np
from scipy.fft import dct
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from v5.data import Store
from .common import sha256

L,H=512,96
PHASES={'A':(0.,.32,32,16384),'B':(.32,.40,96,4096),
        'C':(.40,.55,96,None),'D':(.55,.70,96,16384),
        'E':(.70,.80,96,None),'F':(.80,1.,96,None)}
MODELS=('moirai2','patchtst','dlinear','ridge')
ARMS=('query_only','real','matched_random','deranged_1','deranged_2','deranged_3','deranged_4','memory_mean')


def inside(series,start,length=L+H):
    return any(a<=start and start+length<=b for a,b in series['runs'])


def refs_for(series,phase):
    lo,hi,stride,_=PHASES[phase];left=int(series['n']*lo);right=int(series['n']*hi)
    anchor=((left+stride-1)//stride)*stride
    return np.array([t for t in range(anchor,right-L-H+1,stride) if inside(series,t)],dtype=np.int64)


def plan(store):
    keep=[];excluded=[];stats={}
    for series in store.series:
        sid=series['sid'];raw=np.asarray(store.values(sid)[:int(.32*series['n'])]);finite=raw[np.isfinite(raw)]
        reason=None
        if len(refs_for(series,'C'))<20:reason='Fewer than20 complete independent-memory candidates'
        if any(not len(refs_for(series,k)) for k in ('A','B','D','E','F')):reason='Empty complete temporal phase'
        if len(finite)<L+H:reason='Insufficient finite base TRAIN'
        if reason:excluded.append(dict(sid=sid,reason=reason));continue
        stats[str(sid)]=dict(mean=float(finite.mean()),std=max(float(finite.std()),1e-6))
        keep.append(sid)
    if not keep:raise ValueError('No eligible variables')
    refs={}
    for phase,(_,_,_,cap) in PHASES.items():
        # Use a complete panel of temporal origins for D/E/F. This makes the
        # independent resampling unit time, without overweighting large channels.
        starts={sid:refs_for(store.series[sid],phase) for sid in keep}
        if phase in ('B','D','E','F'):
            common=starts[keep[0]]
            for sid in keep[1:]:common=np.intersect1d(common,starts[sid])
            if not len(common):raise ValueError('No common valid temporal panel '+phase)
            starts={sid:common for sid in keep}
        quota=max(1,cap//len(keep)) if cap else None
        pairs=[]
        for sid in keep:
            ts=starts[sid]
            if quota and len(ts)>quota:ts=ts[np.unique(np.rint(np.linspace(0,len(ts)-1,quota)).astype(int))]
            pairs.extend((sid,int(t)) for t in ts)
        refs[phase]=np.asarray(pairs,dtype=np.int64)
    hashes={str(sid):sha256(store.path/store.series[sid]['file']) for sid in keep}
    for sid,h in hashes.items():
        if h!=store.series[int(sid)]['values_sha256']:raise ValueError('Changed physical data')
    return dict(sids=keep,excluded=excluded,stats=stats,refs=refs,values_sha256=hashes,
                catalog_sha256=sha256(store.path/'catalog.json'))


def materialize(store,refs,stats,targets=True):
    x=np.stack([store.window(s,t,L) for s,t in refs]);sd=np.array([stats[str(s)]['std'] for s,t in refs],dtype=np.float32)
    out=dict(x=x,sid=refs[:,0],start=refs[:,1],std=sd)
    if targets:out['y']=np.stack([store.window(s,t+L,H) for s,t in refs])
    return out


def norm(x,sd):
    x=np.asarray(x,dtype=float);center=x.mean(1);scale=np.maximum(x.std(1),.1*np.asarray(sd))
    return (x-center[:,None])/scale[:,None],center,scale


def query_metadata(d,packet):
    """Preserve observable level, amplitude and clock in EVERY predictor arm."""
    out=dict(d)
    out['training_mean']=np.array([packet['stats'][str(int(s))]['mean'] for s in d['sid']])
    out['daily_period']=np.full(len(d['sid']),packet['daily_period'])
    # All panel variables in a dataset have the same original length. Infer it
    # from the immutable original catalog, never from TEST outcomes.
    import json
    catalog=json.loads((Path(packet['store'])/'catalog.json').read_text(encoding='utf-8'))
    out['series_length']=np.array([catalog['series'][int(s)]['n'] for s in d['sid']])
    return out


class DLinear(nn.Module):
    """Shared-channel DLinear,25-point moving mean, with explicit instance scaling."""
    def __init__(self):
        super().__init__();self.season=nn.Linear(L,H);self.trend=nn.Linear(L,H)
        nn.init.constant_(self.season.weight,1/L);nn.init.constant_(self.trend.weight,1/L)
        nn.init.zeros_(self.season.bias);nn.init.zeros_(self.trend.bias)
    def forward(self,x,floor):
        center=x.mean(1);scale=x.std(1,correction=0).clamp_min(floor);z=(x-center[:,None])/scale[:,None]
        trend=F.avg_pool1d(F.pad(z[:,None],(12,12),mode='replicate'),25,stride=1)[:,0]
        out=self.season(z-trend)+self.trend(trend)
        return dict(raw=out*scale[:,None]+center[:,None],normalized=out,center=center,scale=scale)


def neural_model(kind):
    if kind=='dlinear':return DLinear()
    if kind!='patchtst':raise ValueError(kind)
    from v6.patchtst_direct import PatchTSTForecast
    return PatchTSTForecast(dict(length=L,horizons=[H],model=dict(hidden=64,patch_len=16,
        patch_stride=8,n_heads=4,e_layers=2,dropout=.1)))


def train_neural(kind,a,b,seed,device,max_epochs=32):
    torch.manual_seed(seed);np.random.seed(seed);model=neural_model(kind).to(device)
    x=torch.as_tensor(a['x'],device=device);y=torch.as_tensor(a['y'],device=device)
    floor=torch.as_tensor(.1*a['std'],device=device)
    bx=torch.as_tensor(b['x'],device=device);by=torch.as_tensor(b['y'],device=device);bf=torch.as_tensor(.1*b['std'],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=float('inf');state=None;stale=0;history=[];rng=np.random.default_rng(seed)
    for epoch in range(max_epochs):
        model.train();loss_sum=0.
        for ids in np.array_split(rng.permutation(len(x)),math.ceil(len(x)/256)):
            out=model(x[ids],floor[ids]);target=(y[ids]-out['center'][:,None])/out['scale'][:,None]
            loss=(out['normalized']-target).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite backbone fit')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
        model.eval();total=0.
        with torch.no_grad():
            for begin in range(0,len(bx),256):
                sl=slice(begin,begin+256);out=model(bx[sl],bf[sl]);target=(by[sl]-out['center'][:,None])/out['scale'][:,None]
                total+=float((out['normalized']-target).square().mean())*len(bx[sl])
        score=total/len(bx);history.append(dict(epoch=epoch+1,train=loss_sum/len(x),validation=score))
        if score<best-1e-5:
            best=score;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=6:break
    model.load_state_dict(state);model.eval()
    return model,dict(best_validation=best,epochs=len(history),history=history,parameters=sum(p.numel() for p in model.parameters()))


def train_ridge(a,b):
    x,_,_=norm(a['x'],a['std']);_,center,scale=norm(a['x'],a['std']);y=(a['y']-center[:,None])/scale[:,None]
    x=np.c_[np.ones(len(x)),x];gram=x.T@x;rhs=x.T@y;best=None
    for penalty in (1.,10.,100.):
        reg=np.eye(L+1)*penalty;reg[0,0]=1e-8;coef=np.linalg.solve(gram+reg,rhs)
        pred=predict_ridge(coef,b);_,bc,bs=norm(b['x'],b['std'])
        score=float(np.mean(((pred-b['y'])/bs[:,None])**2))
        if best is None or score<best[0]:best=(score,penalty,coef)
    return best[2],dict(best_validation=best[0],penalty=best[1],parameters=(L+1)*H)


def predict_ridge(coef,d):
    x,center,scale=norm(d['x'],d['std'])
    return (np.c_[np.ones(len(x)),x]@coef)*scale[:,None]+center[:,None]


def predict(model,kind,d,device):
    if kind=='ridge':return predict_ridge(model,d).astype(np.float32)
    parts=[]
    with torch.no_grad():
        for begin in range(0,len(d['x']),128):
            sl=slice(begin,begin+128)
            if kind=='moirai2':out=model.forecast(d['x'][sl],H)[:,4]
            else:out=model(torch.as_tensor(d['x'][sl],device=device),torch.as_tensor(.1*d['std'][sl],device=device))['raw']
            parts.append(out.cpu().numpy())
    return np.concatenate(parts)


def choose_cases(query,bank,seed):
    """Shared history-only5+5, plus state/age-matched complete random controls."""
    real=np.empty((len(query['x']),10),dtype=np.int64);random=np.empty_like(real)
    for sid in np.unique(query['sid']):
        qi=np.flatnonzero(query['sid']==sid);bi=np.flatnonzero(bank['sid']==sid)
        p=bank['x'][bi].astype(float);pz,pm,ps=norm(p,bank['std'][bi]);starts=bank['start'][bi]
        for row in qi:
            if np.any(starts+L+H>query['start'][row]):raise ValueError('Memory overlaps FULL query history')
            x=query['x'][row].astype(float);q=(x-x.mean())/max(x.std(),.1*query['std'][row])
            shape=((pz-q)**2).mean(1);raw=((p-x)**2).mean(1)
            first=np.lexsort((starts,shape))[:5];last=[]
            for j in np.lexsort((starts,raw)):
                if j not in first and all(abs(starts[j]-starts[k])>=192 for k in last):last.append(int(j))
                if len(last)==5:break
            if len(last)!=5:raise ValueError('Insufficient diverse raw-history5')
            chosen=np.r_[first,last];real[row]=bi[chosen]
            pool=np.flatnonzero(~np.isin(np.arange(len(bi)),chosen))
            # Conditional random sampling uses only known historical level,
            # amplitude and age. Keep exact10 distinct entries and no outcomes.
            sf=np.c_[pm/query['std'][row],np.log(ps/query['std'][row]),
                np.log1p((query['start'][row]-starts-L-H)/96)]
            rng=np.random.default_rng(seed+int(sid)*1000003+int(query['start'][row]))
            distance=((sf[chosen,None]-sf[pool][None])**2).mean(2)
            _,assignment=linear_sum_assignment(distance+.1*rng.gumbel(size=distance.shape))
            random[row]=bi[pool[assignment]]
    return real,random


def feature_blocks(query,base,bank,case_base,real,random,arm,seed):
    q,center,scale=norm(query['x'],query['std']);b=(base-center[:,None])/scale[:,None]
    n=len(q);training_mean=query.get('training_mean',np.zeros(n))
    period=query.get('daily_period',np.full(n,24.));length=query.get('series_length',np.full(n,1.))
    phase=2*np.pi*(query['start']+L)/period
    state=np.c_[(center-training_mean)/query['std'],np.log(scale/query['std']),
        np.sin(phase),np.cos(phase),np.sin(phase/7),np.cos(phase/7),query['start']/length]
    common=np.c_[q,b,state]
    if arm=='query_only':
        # Same dimensional pathway, using query information only, not zeroed
        # weights. Includes nonlinear summaries absent from the raw linear block.
        c=dct(q,type=2,norm='ortho',axis=1)[:,:16]
        return common,np.c_[c,np.sign(c)*np.sqrt(abs(c))],scale
    out=np.empty((len(q),32),dtype=float)
    fixed_mean={sid:np.flatnonzero(bank['sid']==sid)[np.rint(np.linspace(0,np.sum(bank['sid']==sid)-1,10)).astype(int)] for sid in np.unique(query['sid'])} if arm=='memory_mean' else None
    for begin in range(0,len(q),256):
        rows=np.arange(begin,min(begin+256,len(q)))
        if arm=='memory_mean':
            # Same10-per-query budget: fixed uniformly spaced historical cases,
            # independent of query shape, gives a global-memory calibration arm.
            ids=np.stack([fixed_mean[query['sid'][r]] for r in rows])
        else:ids=random[rows] if arm=='matched_random' else real[rows]
        past=bank['x'][ids].astype(float);pm=past.mean(2);ps=np.maximum(past.std(2),.1*query['std'][rows,None])
        keys=(past-pm[:,:,None])/ps[:,:,None]
        distance=((keys-q[rows,None])**2).mean(2)+.1*((pm-center[rows,None])/query['std'][rows,None])**2
        if arm=='memory_mean':weights=np.full(distance.shape,.1)
        else:
            weights=np.zeros_like(distance)
            for j in (0,5):
                z=-np.clip(distance[:,j:j+5],0,80);z-=z.max(1,keepdims=True);w=np.exp(z)
                weights[:,j:j+5]=.5*w/w.sum(1,keepdims=True)
        errors=(bank['y'][ids].astype(float)-case_base[ids])/ps[:,:,None]
        if arm.startswith('deranged_'):
            shift=int(arm.rsplit('_',1)[1]);errors=errors.copy()
            for j in (0,5):errors[:,j:j+5]=np.roll(errors[:,j:j+5],shift,axis=1)
        old=np.einsum('nk,nkl->nl',weights,keys).reshape(-1,16,32).mean(2)
        resid=np.einsum('nk,nkh->nh',weights,errors)
        patches=resid.reshape(-1,12,8).mean(2)
        dispersion=np.einsum('nk,nkh->n',weights,(errors-resid[:,None])**2)/H
        age=np.log1p((query['start'][rows,None]-bank['start'][ids]-L-H)/96)
        meta=np.c_[(weights*distance).sum(1),np.sqrt(np.maximum(dispersion,0)),
            (weights*age).sum(1),np.sqrt((resid**2).mean(1))]
        out[rows]=np.c_[old,patches,meta]
    return common,np.clip(out,-40,40),scale


def fit_probe(common,extra,target,seed):
    x=np.c_[common,extra];mean=x.mean(0);std=np.maximum(x.std(0),.05)
    rng=np.random.default_rng(seed);projection=rng.normal(size=(x.shape[1],128))/math.sqrt(x.shape[1])
    offset=rng.uniform(0,2*np.pi,128)
    phi=np.cos(np.clip((x-mean)/std,-8,8)@projection+offset)*math.sqrt(2/128)
    fm=phi.mean(0);tm=target.mean(0);z=phi-fm;gram=z.T@z
    eig,vec=np.linalg.eigh(gram);eig=np.maximum(eig,0)
    df_target=min(64.,len(x)/2);lo=1e-12;hi=max(float(eig.max()),1.)*1e8
    for _ in range(80):
        middle=np.sqrt(lo*hi);df=float(np.sum(eig/(eig+middle)))
        if df>df_target:lo=middle
        else:hi=middle
    penalty=np.sqrt(lo*hi);coef=vec@((vec.T@(z.T@(target-tm)))/(eig[:,None]+penalty))
    return dict(mean=mean,std=std,projection=projection,offset=offset,feature_mean=fm,
        target_mean=tm,coef=coef,penalty=np.array(penalty),effective_df=np.array(np.sum(eig/(eig+penalty))))


def apply_probe(model,common,extra,base,scale):
    x=np.c_[common,extra];phi=np.cos(np.clip((x-model['mean'])/model['std'],-8,8)@model['projection']+model['offset'])*math.sqrt(2/128)
    correction=(phi-model['feature_mean'])@model['coef']+model['target_mean']
    return np.asarray(base,dtype=float)+np.clip(correction,-2,2)*scale[:,None]
