"""Learn historical incremental utility; an explicit null action returns Moirai.

No query target is an input. Older completed prediction errors are legitimate
features, but must mature before the CURRENT HISTORY begins. The selector is
fitted on post-expert historical predictions, never expert TRAIN residuals.
"""
import numpy as np
import torch
from torch import nn
from .unified_residual import features

ARMS=['query_only','history_only','real','shuffled','matched_random','deranged',
      'value_only','case_utility','case_random_utility','case_deranged_utility']
SETTINGS=dict(seeds=[13,41,73],epochs=40,width=32,batch_size=8192,learning_rate=.002,
    weight_decay=.01,risk_penalty=.5,ensemble_uncertainty_penalty=1.,
    thresholds=[0.,.05,.1,.25,.5,1.,2.],strengths=[.1,.25,.5,1.],
    primary='real',case_strength=.25,patch_size=8)


def rows(data,ids):
    return {k:v[ids] for k,v in data.items()}


def chronological_split(data,train_end,test_start):
    starts=np.unique(data['start']);chunks=np.array_split(starts,3)
    if min(map(len,chunks))<3:raise ValueError('Need at least nine historical origins')
    fit=np.flatnonzero((data['start']>=train_end)&np.isin(data['start'],chunks[0])&(data['start']+192<=chunks[1][0]))
    cal=np.flatnonzero(np.isin(data['start'],chunks[1])&(data['start']+192<=chunks[2][0]))
    confirm=np.flatnonzero(np.isin(data['start'],chunks[2])&(data['start']+192<=test_start))
    if min(map(len,[fit,cal,confirm]))==0:raise ValueError('Empty purged split')
    if data['start'][fit].max()+192>data['start'][cal].min() or data['start'][cal].max()+192>data['start'][confirm].min():raise ValueError('Overlapping selection blocks')
    return fit,cal,confirm


def arm_mode(arm):
    if arm in ['case_random_utility','matched_random']:return 'matched_random'
    if arm in ['deranged','case_deranged_utility','shuffled']:return 'shuffled'
    if arm in ['value_only','case_utility']:return 'real'
    return arm


def options(data,stats,period,arm,frozen=None):
    """N x 12 x actions x 8: complete forecasts, not target-derived labels."""
    base=np.asarray(data['raw_quantiles'][:,4],dtype=float).reshape(-1,12,8)
    if arm.startswith('case_') or arm=='value_only':
        f,_,scale=features(data,stats,period,arm_mode(arm))
        r=f['future'].astype(float).reshape(-1,10,12,8).transpose(0,2,1,3)*scale[:,None,None,None]
        if arm=='value_only':r=r.mean(2,keepdims=True)
        radius=f['radius'].reshape(-1,12,8)*scale[:,None,None]
        correction=np.clip(SETTINGS['case_strength']*r,-radius[:,:,None],radius[:,:,None])
        return (base[:,:,None]+correction).astype(np.float32)
    if frozen is None:raise ValueError('Frozen forecast required')
    return np.asarray(frozen,dtype=np.float32).reshape(-1,12,1,8)


def losses(data,candidates):
    y=data['raw_y'].astype(float).reshape(-1,12,8)
    b=data['raw_quantiles'][:,4].astype(float).reshape(-1,12,8)
    return ((b-y)**2).mean(2),((candidates.astype(float)-y[:,:,None])**2).mean(3)


def utility_targets(data,candidates,stats):
    base,other=losses(data,candidates)
    sd=np.asarray(stats['std'])[data['sid']]
    gain=(base[:,:,None]-other)/sd[:,None,None]**2
    return gain,np.maximum(-gain,0.)


def selector_features(data,candidates,stats,period,arm,history,history_options):
    """Only past query outcomes in history may be read; raw_y(data) is not read."""
    mode=arm_mode(arm);f,_,scale=features(data,stats,period,mode)
    x=data['raw_x'].astype(float);n=len(x);j=candidates.shape[2]
    q=data['raw_quantiles'].astype(float);mean=x.mean(1);sd=np.asarray(stats['std'])[data['sid']]
    def patch(v):return np.asarray(v).reshape(n,12,8)
    base=patch(q[:,4]);norm=(base-mean[:,None,None])/scale[:,None,None]
    cq=(candidates.astype(float)-base[:,:,None])/scale[:,None,None,None]
    common=np.stack([norm.mean(2),norm.std(2),(norm[:,:,-1]-norm[:,:,0]),
        patch((q[:,8]-q[:,0])/scale[:,None]).mean(2),
        patch((q[:,6]-q[:,2])/scale[:,None]).mean(2),
        patch((q[:,8]+q[:,0]-2*q[:,4])/scale[:,None]).mean(2),
        np.broadcast_to(np.arange(12)[None]/11,(n,12))],2)
    h=(x-mean[:,None])/scale[:,None]
    probe=f['probe_error']
    globalf=np.concatenate([f['state'],h.reshape(n,12,8).mean(2),
        np.stack([h.std(1),np.mean(abs(np.diff(h)),1),np.mean(h[:,:-1]*h[:,1:],1),
            probe.mean(1),probe.std(1),np.mean(probe**2,1),probe[:,-8:].mean(1)],1)],1)
    # Prior full-96-horizon competence: only completed, same-series forecasts.
    hb,ho=losses(history,history_options);hg=(hb-ho.mean(2))/np.asarray(stats['std'])[history['sid'],None]**2
    hb=hb/np.asarray(stats['std'])[history['sid'],None]**2
    prior=np.zeros((n,12,5),dtype=float)
    for sid in np.unique(data['sid']):
        past=np.flatnonzero(history['sid']==sid);past=past[np.argsort(history['start'][past])]
        for row in np.flatnonzero(data['sid']==sid):
            ready=past[history['start'][past]+192<=data['start'][row]][-8:]
            if len(ready):prior[row]=np.stack([hg[ready].mean(0),hg[ready[-1]],(hg[ready]>0).mean(0),hb[ready].mean(0),np.full(12,len(ready)/8)],1)
    # The history-only control never receives an observed case future.
    past=f['past'].astype(float);dist=np.mean((past-h[:,None])**2,2)
    residual=f['future'].astype(float).reshape(n,10,12,8).transpose(0,2,1,3)
    pool=np.stack([residual.mean((2,3)),residual.mean(3).std(2),residual.std(3).mean(2),
        np.mean(np.sign(residual.mean(3)),2),np.broadcast_to(dist.mean(1)[:,None],(n,12)),
        np.broadcast_to(dist.min(1)[:,None],(n,12))],2)
    if mode=='query_only':pool[:]=0.
    context=np.concatenate([common,np.broadcast_to(globalf[:,None],(n,12,globalf.shape[1])),prior,pool],2)
    own=np.zeros((n,12,j,7),dtype=float)
    own[:,:,:,:4]=np.stack([cq.mean(3),cq.std(3),cq[:,:,:,-1]-cq[:,:,:,0],np.mean(cq**2,3)],3)
    if arm.startswith('case_'):
        own[:,:,:,4]=dist[:,None]
        own[:,:,:,5]=f['metadata'][:,None,:,3]
        own[:,:,:,6]=np.broadcast_to((np.arange(10)//5)[None,None],(n,12,10))
    out=np.concatenate([np.broadcast_to(context[:,:,None],(n,12,j,context.shape[2])),own],3)
    if not np.isfinite(out).all():raise ValueError('Nonfinite selector features')
    return np.clip(out,-10000,10000).astype(np.float32)


class UtilityNet(nn.Module):
    def __init__(self,inputs,width=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(inputs,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,3))
    def forward(self,x):
        v=self.net(x);return torch.stack([v[:,0],torch.nn.functional.softplus(v[:,1]),v[:,2]],1)


def fit_selector(x,gain,harm,sid,stats,device='cpu'):
    """Post-expert historical labels; all epochs/seeds fixed before TEST."""
    shape=x.shape;n,p,j,d=shape;a=x.reshape(-1,d).astype(float)
    mean=a.mean(0);scale=np.maximum(a.std(0),.05);a=np.clip((a-mean)/scale,-8,8).astype(np.float32)
    target_scale=max(float(np.sqrt(np.mean(gain**2))),.01)
    targets=np.stack([gain/target_scale,harm/target_scale,(gain>0).astype(float)],-1).reshape(-1,3).astype(np.float32)
    if not np.isfinite(targets).all():raise ValueError('Nonfinite utility label')
    std=np.asarray(stats['std']);weights=.5+.5*std[np.asarray(sid)]**2/np.mean(std**2)
    weights=np.broadcast_to(weights[:,None,None],(n,p,j)).reshape(-1).copy();weights/=weights.mean()
    tx=torch.as_tensor(a,device=device);ty=torch.as_tensor(targets,device=device);tw=torch.as_tensor(weights,dtype=torch.float32,device=device)
    models=[];records=[]
    for seed in SETTINGS['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=UtilityNet(d,SETTINGS['width']).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay']);losses_out=[]
        for _ in range(SETTINGS['epochs']):
            order=rng.permutation(len(a));total=0.
            for begin in range(0,len(order),SETTINGS['batch_size']):
                ids=torch.as_tensor(order[begin:begin+SETTINGS['batch_size']],device=device);out=model(tx[ids]);target=ty[ids]
                loss=((out[:,0]-target[:,0])**2+.5*(out[:,1]-target[:,1])**2+.1*torch.nn.functional.binary_cross_entropy_with_logits(out[:,2],target[:,2],reduction='none'))
                loss=(loss*tw[ids]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Invalid selector loss')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(ids)
            losses_out.append(total/len(a))
        model.eval();models.append(model);records.append(dict(seed=seed,losses=losses_out,parameters=sum(p.numel() for p in model.parameters())))
    normal=dict(mean=mean.tolist(),scale=scale.tolist(),target_scale=target_scale,input_width=d)
    return models,normal,records


@torch.no_grad()
def score_selector(models,normal,x,device='cpu'):
    d=x.shape[-1];flat=x.reshape(-1,d);a=np.clip((flat-np.asarray(normal['mean']))/np.asarray(normal['scale']),-8,8).astype(np.float32);pred=[]
    for model in models:
        out=[]
        for begin in range(0,len(a),SETTINGS['batch_size']):out.append(model(torch.as_tensor(a[begin:begin+SETTINGS['batch_size']],device=device)).cpu().numpy())
        pred.append(np.concatenate(out))
    pred=np.asarray(pred);gain=pred[:,:,0].mean(0);harm=pred[:,:,1].mean(0)
    score=gain-SETTINGS['risk_penalty']*harm-SETTINGS['ensemble_uncertainty_penalty']*pred[:,:,0].std(0)
    return score.reshape(x.shape[:-1]).astype(np.float32)


def route(base,candidates,scores,threshold,strength,enabled=True):
    """An inactive patch is bit-identical to the foundation forecast."""
    n=len(base);b=np.asarray(base).reshape(n,12,8);choice=np.argmax(scores,2)
    best=np.take_along_axis(scores,choice[:,:,None],2)[:,:,0]
    mask=(best>threshold)&bool(enabled)&(strength>0)
    picked=np.take_along_axis(candidates,choice[:,:,None,None],2)[:,:,0]
    out=b.copy();out[mask]=(b+strength*(picked-b))[mask]
    return out.reshape(n,96),mask,choice


def error_metrics(data,pred,stats):
    e=np.asarray(pred,dtype=float)-data['raw_y'].astype(float)
    return dict(mse=float(np.mean(e**2)),mae=float(np.mean(abs(e))),train_nmse=float(np.mean((e/np.asarray(stats['std'])[data['sid'],None])**2)),
        source_nmse=float(np.mean((e/np.asarray(data['std'])[:,None])**2)))


def nonworse(proposal,base):
    return all(proposal[k]<=base[k]*(1+1e-9)+1e-12 for k in ['mse','mae','train_nmse'])


def calibrate_router(data,candidates,scores,stats,cal,confirm):
    b=data['raw_quantiles'][:,4];bc=error_metrics(rows(data,cal),b[cal],stats);best=None
    for threshold in SETTINGS['thresholds']:
        for strength in SETTINGS['strengths']:
            pred,mask,_=route(b,candidates,scores,threshold,strength)
            m=error_metrics(rows(data,cal),pred[cal],stats)
            if mask[cal].any() and nonworse(m,bc) and m['mse']<bc['mse']*(1-1e-5):
                option=(m['mse']/max(bc['mse'],1e-20)+m['train_nmse']/max(bc['train_nmse'],1e-20),threshold,strength,m)
                if best is None or option[:3]<best[:3]:best=option
    if best is None:return dict(enabled=False,threshold=0.,strength=0.,reason='No calibration improvement')
    _,threshold,strength,calmetric=best;pred,mask,_=route(b,candidates,scores,threshold,strength)
    bm=error_metrics(rows(data,confirm),b[confirm],stats);cm=error_metrics(rows(data,confirm),pred[confirm],stats)
    enabled=bool(mask[confirm].any() and nonworse(cm,bm) and cm['mse']<bm['mse']*(1-1e-5))
    return dict(enabled=enabled,threshold=threshold,strength=strength,calibration=calmetric,confirmation_base=bm,confirmation=cm,
        reason='Confirmed on later history' if enabled else 'Later confirmation rejected; direct Moirai')


def diagnostic(data,candidates,scores,pred,mask,choice,stats,strength=1.,seed=20261015):
    """TEST labels are used only HERE, after the router policy is frozen."""
    b=data['raw_quantiles'][:,4];base,other=losses(data,candidates)
    bp=b.reshape(-1,12,8).astype(float);truth=data['raw_y'].reshape(-1,12,8).astype(float)
    # Same candidate/strength family as calibration; only the diagnostic oracle
    # is allowed to see the test losses. Actual selection never reads them.
    alternatives=np.concatenate([((bp[:,:,None]+s*(candidates-bp[:,:,None])-truth[:,:,None])**2).mean(3) for s in SETTINGS['strengths']],2)
    oracle=np.minimum(base,alternatives.min(2));whole=np.minimum(base.mean(1),alternatives.mean(1).min(1))
    actual=np.mean((pred.astype(float)-data['raw_y'])**2,1).mean()
    selected=np.take_along_axis(candidates,choice[:,:,None,None],2)[:,:,0]
    delta=pred.reshape(-1,12,8).astype(float)-b.reshape(-1,12,8)
    gain=base-((pred.reshape(-1,12,8).astype(float)-data['raw_y'].reshape(-1,12,8))**2).mean(2)
    active=mask;helpful=active&(gain>0)
    full_best_score=scores.max(2);rng=np.random.default_rng(seed);random_mse=[]
    # Permute the ACTUAL proposed correction within each series and lead only via
    # its adoption mask. Candidate and strength stay attached to their own query.
    cd=selected.astype(float)-b.reshape(-1,12,8)
    for _ in range(200):
        rm=active.copy()
        for sid in np.unique(data['sid']):
            ids=np.flatnonzero(data['sid']==sid)
            for patch in range(12):rm[ids,patch]=active[rng.permutation(ids),patch]
        rp=b.reshape(-1,12,8).astype(float)+rm[:,:,None]*strength*cd
        random_mse.append(float(np.mean((rp-data['raw_y'].reshape(-1,12,8))**2)))
    coverage=[]
    for fraction in [.1,.25,.5,.75,1.]:
        count=max(1,int(np.ceil(fraction*full_best_score.size)));flat=np.argsort(-full_best_score.ravel(),kind='stable')[:count]
        cm=np.zeros(full_best_score.size,dtype=bool);cm[flat]=True;cm=cm.reshape(full_best_score.shape)
        cp=b.reshape(-1,12,8).astype(float)+cm[:,:,None]*cd
        coverage.append(dict(coverage=fraction,mse=float(np.mean((cp-data['raw_y'].reshape(-1,12,8))**2)),diagnostic_only=True))
    return dict(base_mse=float(base.mean()),actual_mse=float(actual),patch_oracle_mse=float(oracle.mean()),window_oracle_mse=float(whole.mean()),
        oracle_patch_gain_percent=float(100*(1-oracle.mean()/base.mean())),adoption_fraction=float(active.mean()),
        adopted_help_fraction=float(helpful.sum()/active.sum()) if active.any() else None,
        adopted_gain_contribution=float(np.maximum(gain,0).sum()/gain.size),adopted_harm_contribution=float(np.maximum(-gain,0).sum()/gain.size),
        matched_coverage_random_mse_mean=float(np.mean(random_mse)),matched_coverage_random_mse_95=np.quantile(random_mse,[.025,.975]).tolist(),
        risk_coverage=coverage,oracle_not_deployable=True)
