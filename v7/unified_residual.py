"""Matched 5+5 ablations using errors of one frozen Moirai forecaster."""
import numpy as np
import torch
from torch import nn
from .aligned_case_fusion import CaseFusion,inputs as old_inputs

MODES=['query_only','history_only','real','shuffled','matched_random']


def state_features(past,starts,period,mean,sd):
    x=np.asarray(past,dtype=float);t=np.arange(96)-47.5
    scale=np.maximum(x.std(-1),.1*sd);phase=2*np.pi*(np.asarray(starts)+96)/(period or 1)
    return np.stack([(x.mean(-1)-mean)/sd,np.log(scale/sd),
        (x@t)/(t@t)*95/scale,np.sin(phase),np.cos(phase),
        np.sin(phase/7),np.cos(phase/7)],-1)


def matched_starts(bank_starts,bank_past,selected,start,period,mean,sd,seed,bank_features=None):
    """Match each real case's observed state; never inspect a case future."""
    starts=np.asarray(bank_starts);ids=np.searchsorted(starts,selected)
    if np.any(ids>=len(starts)) or not np.array_equal(starts[ids],selected):raise ValueError('Unknown real case')
    features=state_features(bank_past,starts,period,mean,sd) if bank_features is None else bank_features
    eligible=(starts+192<=start)&~np.isin(starts,selected)
    rng=np.random.default_rng(seed);chosen=[]
    for j,index in enumerate(ids):
        ok=eligible&~np.isin(starts,chosen)
        if j>=5 and chosen[5:]:ok&=np.all(abs(starts[:,None]-np.array(chosen[5:])[None])>=192,axis=1)
        pool=np.flatnonzero(ok)
        if not len(pool):raise ValueError('Insufficient matched cases')
        # Match calendar, level, amplitude, trend and log age to the real case.
        distance=np.mean((features[pool]-features[index])**2,axis=1)
        age=np.log1p(np.maximum(start-starts[pool]-192,0)/192.)
        distance+=.25*(age-np.log1p((start-starts[index]-192)/192.))**2
        nearest=pool[np.lexsort((starts[pool],distance))[:min(32,len(pool))]]
        chosen.append(int(starts[int(rng.choice(nearest))]))
    return np.array(chosen,dtype=np.int64)


def features(data,stats,period,mode):
    """The query target raw_y is deliberately not read here."""
    if mode not in MODES:raise ValueError('Unknown ablation')
    random=mode=='matched_random';d=dict(data)
    if random:
        for key in ['case_past','case_future','selected_starts']:d[key]=data['random_'+key]
    if mode=='history_only':d['case_future']=d['case_past']
    s=dict(stats,case_count=10,metadata_width=13,residual_reference=False)
    values,level,scale=old_inputs(d,s,period,'wiener' if mode!='query_only' else 'without_retrieval')
    x=np.asarray(data['raw_x'],dtype=float);sd=np.asarray(stats['std'])[data['sid']]
    actual_scale=np.maximum(x.std(1),.1*sd)
    prior=data['random_case_baseline'] if random else data['case_baseline']
    past=d['case_past']
    if mode=='query_only':
        past=np.repeat(x[:,None],10,axis=1);prior=np.repeat(data['raw_quantiles'][:,4,None],10,axis=1)
    ps=past.std(2);qs=x.std(1)
    slope=np.divide(qs[:,None]*ps,ps**2+(qs[:,None]/5.)**2,
        out=np.zeros_like(ps,dtype=float),where=ps**2+(qs[:,None]/5.)**2>1e-12)
    known=(x[:,64:]-data['probe_quantiles'][:,4])/actual_scale[:,None]
    if mode in ['query_only','history_only']:
        # Observed probe errors activate the identical value/transfer pathway.
        # This is a history-derived placeholder, never a fabricated observed future.
        residual=np.repeat(np.tile(known,(1,3))[:,None],10,axis=1)
    else:residual=(d['case_future'].astype(float)-prior.astype(float))*slope[:,:,None]/actual_scale[:,None,None]
    if mode=='shuffled':residual=residual[np.arange(len(x))[:,None],data['shuffle_order']]
    prior_level=(prior.mean(2)-past[:,:,-16:].mean(2))/actual_scale[:,None]
    prior_change=(prior[:,:,-1]-prior[:,:,0])/actual_scale[:,None]
    extra=np.stack([prior_level,prior_change],2)
    if mode=='query_only':extra=np.zeros_like(extra)
    values.update(future=np.clip(residual,-8,8).astype(np.float32),
        metadata=np.clip(np.concatenate([values['metadata'],extra],2),-8,8).astype(np.float32),
        reference=values['quantiles'][:,4],probe_error=np.clip(known,-8,8).astype(np.float32))
    return values,level,scale


class UnifiedResidualFusion(CaseFusion):
    def __init__(self,variables,width=32):
        super().__init__(variables,width,conditioned=True,paired=True,metadata_width=15)
        self.residual_gate=nn.Linear(96,12)
        nn.init.zeros_(self.residual_gate.weight);nn.init.zeros_(self.residual_gate.bias)

    def forward(self,history,future,radius,return_attention=False,**kwargs):
        delta,weights=super().forward(history=history,future=future,radius=radius,
            return_attention=True,**kwargs)
        old=future.reshape(len(history),10,12,8)
        transfer=torch.einsum('npk,nkpd->npd',weights[:,:,:10],old)
        gate=torch.tanh(self.residual_gate(history))[:,:,None]
        delta=delta+(gate*transfer).reshape(len(history),96)
        delta=torch.maximum(torch.minimum(delta,radius),-radius)
        return (delta,weights) if return_attention else delta


def time_blocks(data):
    blocks=[[],[],[]]
    for sid in np.unique(data['sid']):
        rows=np.flatnonzero(data['sid']==sid);rows=rows[np.argsort(data['start'][rows])]
        for b,ids in zip(blocks,np.array_split(rows,3)):b.extend(ids.tolist())
    return [np.array(b,dtype=int) for b in blocks]


def aggregate_guard(data,proposal,base,std):
    """Same validation rule for every arm; never use TEST losses to revert."""
    checks=[]
    for ids in time_blocks(data):
        a=base[ids].astype(float)-data['raw_y'][ids];b=proposal[ids].astype(float)-data['raw_y'][ids]
        ea=[float(np.mean(a*a)),float(np.mean((a/np.asarray(std)[data['sid'][ids],None])**2)),float(np.mean(abs(a)))]
        eb=[float(np.mean(b*b)),float(np.mean((b/np.asarray(std)[data['sid'][ids],None])**2)),float(np.mean(abs(b)))]
        checks.append(dict(base=ea,proposal=eb,accepted=all(v<=u*(1+1e-9)+1e-12 for u,v in zip(ea,eb))))
    return dict(accepted=all(v['accepted'] for v in checks),blocks=checks,
        rule='Each of three VAL blocks must not worsen raw MSE, standardized MSE or MAE')
