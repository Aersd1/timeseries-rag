"""Matched historical cohorts for tabular, tree and raw-sequence utility models."""
import numpy as np
import torch
from torch import nn
from .unified_residual import features
from .selective_rag import selector_features

FAMILIES=['summary_mlp','hist_gbdt','raw_sequence']
SETTINGS=dict(seeds=[13,41,73],epochs=20,summary_width=32,sequence_width=32,sequence_layers=2,
    sequence_heads=4,sequence_batch=128,summary_batch=1024,learning_rate=.002,weight_decay=.01,
    tree_iterations=100,tree_leaves=15,tree_min_leaf=32,tree_learning_rate=.05,tree_bins=63,tree_l2=1.,
    cohorts=['sparse_last4','forward_full'],risk_penalties=[0.,.5],uncertainty_penalties=[0.,1.],
    thresholds=[0.,.05,.1,.25,.5,1.],strengths=[.1,.25,.5,1.],
    validation='First two purged blocks choose settings/family; final two only confirm or reject',
    primary='forward_full portfolio chosen by calibration only; direct Moirai if confirmation fails',
    stability='Positive weekly-cluster95% lower bound AND positive MSE reduction in each of four chronological TEST blocks')


def inputs(data,forecast,stats,period,arm,history,history_forecast,raw=False):
    candidates=np.asarray(forecast).reshape(-1,12,1,8);hc=np.asarray(history_forecast).reshape(-1,12,1,8)
    summary=selector_features(data,candidates,stats,period,arm,history,hc)[:,:,0]
    result=dict(summary=summary)
    if raw:
        f,_,scale=features(data,stats,period,arm);n=len(forecast)
        quantiles=f['quantiles'].reshape(n,9,12,8).transpose(0,2,1,3).reshape(n,12,72)
        candidate=(forecast-data['raw_quantiles'][:,4])/scale[:,None]
        result.update(query=np.concatenate([f['history'].reshape(n,12,8),quantiles,np.clip(candidate,-8,8).reshape(n,12,8)],2).astype(np.float32),
            past=f['past'],future=f['future'].reshape(n,10,12,8).transpose(0,2,1,3),metadata=f['metadata'],probe=f['probe_error'])
    return result


class SummaryUtility(nn.Module):
    def __init__(self,width=52):
        super().__init__();self.net=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,summary,**kwargs):
        v=self.net(summary);return torch.stack([v[...,0],torch.nn.functional.softplus(v[...,1])],-1)


class SequenceUtility(nn.Module):
    """All raw96 samples, forecast quantiles and known old residuals are retained."""
    def __init__(self,width=52):
        super().__init__();w=32
        self.query=nn.Linear(88,w);self.probe=nn.Linear(32,w)
        self.position=nn.Parameter(torch.randn(12,w)*.02)
        layer=nn.TransformerEncoderLayer(w,4,dim_feedforward=64,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.past=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,w));self.metadata=nn.Linear(15,w)
        self.value=nn.Linear(8,w);self.route=nn.Embedding(2,w)
        self.null=nn.Parameter(torch.zeros(1,w));self.attention=nn.MultiheadAttention(w,4,batch_first=True)
        self.output=nn.Sequential(nn.LayerNorm(2*w+width),nn.Linear(2*w+width,64),nn.GELU(),nn.Linear(64,2))
    def forward(self,summary,query,past,future,metadata,probe):
        n=len(summary);h=self.query(query)+self.position[None]+self.probe(probe)[:,None]
        h=self.temporal(h)
        k=self.past(past)+self.metadata(metadata)+self.route(torch.arange(10,device=h.device)//5)[None]
        values=k[:,None]+self.value(future)+self.position[None,:,None]
        null=self.null[None,None].expand(n,12,1,-1);values=torch.cat([values,null],2).reshape(n*12,11,32)
        context,_=self.attention(h.reshape(n*12,1,32),values,values,need_weights=False)
        v=self.output(torch.cat([h,context.reshape(n,12,32),summary],2))
        return torch.stack([v[...,0],torch.nn.functional.softplus(v[...,1])],-1)


def labels(data,forecast,stats):
    base=data['raw_quantiles'][:,4].astype(float);y=data['raw_y'].astype(float);sd=np.asarray(stats['std'])[data['sid']]
    gain=(((base-y)**2).reshape(-1,12,8).mean(2)-((forecast.astype(float)-y)**2).reshape(-1,12,8).mean(2))/sd[:,None]**2
    return gain,np.maximum(-gain,0.)


def fit_models(family,x,gain,harm,data,stats,ids,device='cpu'):
    tab=x['summary'][ids].astype(float);d=tab.shape[-1]
    normal=dict(mean=tab.reshape(-1,d).mean(0).tolist(),scale=np.maximum(tab.reshape(-1,d).std(0),.05).tolist(),
        target_scale=max(float(np.sqrt(np.mean(gain[ids]**2))),.01),width=d)
    z=np.clip((tab-normal['mean'])/normal['scale'],-8,8).astype(np.float32)
    target=np.stack([gain[ids],harm[ids]],2)/normal['target_scale'];std=np.asarray(stats['std'])
    weights=.5+.5*std[data['sid'][ids]]**2/np.mean(std**2);weights/=weights.mean()
    models=[];records=[]
    if family=='hist_gbdt':
        from sklearn.ensemble import HistGradientBoostingRegressor
        for seed in SETTINGS['seeds']:
            rng=np.random.default_rng(seed);origins=np.unique(data['start'][ids]);counts=rng.poisson(1.,len(origins))
            if not counts.any():counts[:]=1
            ww=weights*counts[np.searchsorted(origins,data['start'][ids])];ww=np.repeat(ww,12)
            pair=[]
            for head in range(2):
                model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=32,
                    learning_rate=.05,max_bins=63,l2_regularization=1.,early_stopping=False,random_state=seed)
                model.fit(z.reshape(-1,d),target[:,:,head].ravel(),sample_weight=ww);pair.append(model)
            models.append(pair);records.append(dict(seed=seed,iterations=100,origin_bootstrap=counts.tolist()))
    else:
        xx={k:np.asarray(v[ids],dtype=np.float32) for k,v in x.items()};xx['summary']=z
        for seed in SETTINGS['seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=(SequenceUtility(d) if family=='raw_sequence' else SummaryUtility(d)).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);history=[]
            batch=128 if family=='raw_sequence' else 1024
            for _ in range(SETTINGS['epochs']):
                model.train();order=rng.permutation(len(ids));total=0.
                for begin in range(0,len(order),batch):
                    ii=order[begin:begin+batch];f={k:torch.as_tensor(v[ii],device=device) for k,v in xx.items()}
                    y=torch.as_tensor(target[ii],dtype=torch.float32,device=device);w=torch.as_tensor(weights[ii],dtype=torch.float32,device=device)
                    out=model(**f);loss=(((out[:,:,0]-y[:,:,0])**2+.5*(out[:,:,1]-y[:,:,1])**2).mean(1)*w).mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite utility loss')
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(ii)
                history.append(total/len(ids))
            model.eval();models.append(model);records.append(dict(seed=seed,losses=history,parameters=sum(v.numel() for v in model.parameters())))
    return models,normal,records


@torch.no_grad()
def prediction(family,models,normal,x,device='cpu'):
    tab=x['summary'];n=len(tab);d=tab.shape[-1];z=np.clip((tab-normal['mean'])/normal['scale'],-8,8).astype(np.float32);out=[]
    for model in models:
        if family=='hist_gbdt':
            flat=z.reshape(-1,d);v=np.stack([m.predict(flat) for m in model],1).reshape(n,12,2);v[:,:,1]=np.maximum(v[:,:,1],0);out.append(v)
        else:
            result=[];batch=128 if family=='raw_sequence' else 1024
            for begin in range(0,n,batch):
                f={k:torch.as_tensor(v[begin:begin+batch],device=device) for k,v in x.items()};f['summary']=torch.as_tensor(z[begin:begin+batch],device=device)
                result.append(model(**f).cpu().numpy())
            out.append(np.concatenate(result))
    return np.asarray(out,dtype=np.float32)


def scores(pred,risk,uncertainty):return pred[:,:,:,0].mean(0)-risk*pred[:,:,:,1].mean(0)-uncertainty*pred[:,:,:,0].std(0)


def choose_blocks(data,test_start):
    parts=np.array_split(np.unique(data['start']),4);out=[]
    for i,part in enumerate(parts):
        next_start=parts[i+1][0] if i<3 else test_start
        ids=np.flatnonzero(np.isin(data['start'],part)&(data['start']+192<=next_start))
        if not len(ids):raise ValueError('Insufficient purged validation blocks')
        out.append(ids)
    return out


def apply_policy(data,candidate,prediction,policy):
    base=data['raw_quantiles'][:,4];out=base.copy();mask=np.zeros((len(base),12),bool)
    if not policy['enabled']:return out,mask
    if policy['family']=='static':mask[:]=True
    else:mask=scores(prediction,policy['risk'],policy['uncertainty'])>policy['threshold']
    delta=(candidate-base).reshape(-1,12,8);view=out.reshape(-1,12,8);view[mask]=(view+policy['strength']*delta)[mask]
    return out,mask


def portfolio(policies):
    names=[f for f in FAMILIES+['static'] if f in policies]
    chosen=min(names,key=lambda f:(policies[f]['calibration_objective'],names.index(f)))
    return dict(policies[chosen],selected_family=chosen)


def calibration(data,candidate,pred,family,stats,blocks):
    from .selective_rag import error_metrics,nonworse,rows
    base=data['raw_quantiles'][:,4];bases=[error_metrics(rows(data,b),base[b],stats) for b in blocks];best=None
    e=base.astype(float).reshape(-1,12,8)-data['raw_y'].astype(float).reshape(-1,12,8);delta=(candidate.astype(float)-base).reshape(-1,12,8)
    mse0=(e**2).mean(2);mae0=abs(e).mean(2);sd=np.asarray(stats['std'])[data['sid']]
    gains={s:(mse0-((e+s*delta)**2).mean(2),mae0-abs(e+s*delta).mean(2)) for s in SETTINGS['strengths']}
    params=[(0.,0.,0.)] if family=='static' else [(r,u,t) for r in SETTINGS['risk_penalties'] for u in SETTINGS['uncertainty_penalties'] for t in SETTINGS['thresholds']]
    for risk,uncertainty,threshold in params:
        for strength in SETTINGS['strengths']:
            policy=dict(family=family,risk=risk,uncertainty=uncertainty,threshold=threshold,strength=strength,enabled=True)
            mask=np.ones((len(base),12),bool) if family=='static' else scores(pred,risk,uncertainty)>threshold
            gm,ga=gains[strength];metrics=[]
            for ids,b in zip(blocks[:2],bases[:2]):
                selected=gm[ids]*mask[ids]
                metrics.append(dict(mse=b['mse']-float(selected.mean()),mae=b['mae']-float((ga[ids]*mask[ids]).mean()),
                    train_nmse=b['train_nmse']-float((selected/sd[ids,None]**2).mean())))
            if not mask[np.r_[blocks[0],blocks[1]]].any() or not all(nonworse(a,b) for a,b in zip(metrics,bases)):continue
            objective=sum(a['mse']/max(b['mse'],1e-20)+a['train_nmse']/max(b['train_nmse'],1e-20) for a,b in zip(metrics,bases))
            if objective>=4-1e-5:continue
            key=(objective,risk,uncertainty,threshold,strength)
            if best is None or key<best[0]:best=(key,policy,metrics)
    if best is None:return dict(family=family,enabled=False,strength=0.,risk=0.,uncertainty=0.,threshold=0.,calibration_objective=4.,reason='No calibration improvement')
    key,policy,metrics=best;p,mask=apply_policy(data,candidate,pred,policy)
    metrics=[error_metrics(rows(data,b),p[b],stats) for b in blocks[:2]]
    confirmation=[error_metrics(rows(data,b),p[b],stats) for b in blocks[2:]]
    accepted=all(nonworse(a,b) for a,b in zip(metrics,bases[:2])) and all(nonworse(a,b) and a['mse']<b['mse']*(1-1e-5) for a,b in zip(confirmation,bases[2:]))
    policy.update(enabled=bool(accepted),calibration_objective=key[0],calibration=metrics,confirmation=confirmation,
        reason='Both later blocks confirmed' if accepted else 'Later confirmation rejected; direct Moirai')
    return policy
