"""A frozen raw Moirai forecast with a learned retrieval-error correction head.

The foundation model input stays raw. Only retrieval is detrended. This head
is external to pretrained Moirai, and predicts an error vector, not a mixture
of two independently generated final forecasts.
"""
import math
import numpy as np
import torch
from torch import nn
from .trend_rag import decompose,normalize


def retrieve(history,quantiles,pool,start,window):
    x=np.asarray(history,dtype=np.float32)
    quantiles=np.asarray(quantiles,dtype=np.float32)
    if x.shape!=(96,) or quantiles.shape!=(9,96) or not np.isfinite(quantiles).all():
        raise ValueError('Expected96 history and pinned nine96-point quantile forecasts')
    _,r=decompose(x,window);q,mean,scale=normalize(r,pool.floor)
    # Project the RAW model's forecast trajectories into the residual domain.
    # These are predicted trajectories, never the query's actual continuation.
    paths=np.concatenate([np.repeat(x[None],len(quantiles),axis=0),quantiles],axis=1)
    _,forecast_residual=decompose(paths,window)
    residual_quantiles=(forecast_residual[:,96:]-mean)/scale
    memory=pool.retrieve(q,start,residual_quantiles,[.1,.2,.3,.4,.5,.6,.7,.8,.9])
    return q,memory


def features(history,quantiles,q,memory,errors,memory_std):
    """Only observed query history, predictions and known old model errors."""
    x=np.asarray(history,dtype=np.float32);std=max(float(memory_std),1e-6)
    w=memory['weights'];e=np.asarray(errors,dtype=np.float32)
    if e.shape!=(5,96): raise ValueError('Expected five old normalized error profiles')
    center=np.sum(w[:,None]*e,axis=0)
    # Compatibility scores are summarized over the full100-candidate shortlist.
    quality=np.array([np.log1p(np.median(memory['history_scores'])),
        np.log1p(np.median(memory['future_scores'])),np.log1p(np.mean((e-center)**2)),
        np.mean(quantiles[-1]-quantiles[0])/std,(x[-1]-x[0])/std,
        float(x.std())/std,np.log1p(np.mean(e**2)),-np.sum(w*np.log(np.maximum(w,1e-8)))],dtype=np.float32)
    return np.concatenate([q,(quantiles[4]-x[-1])/std,quality]).astype(np.float32)


def innovations(history,quantiles,pool,memory,memory_std,window,start):
    """Known candidate residual futures minus the raw model's predicted residual.

    The model's predicted trend is obtained by projecting its raw trajectory,
    so no separately forecast trend introduces another error source.
    """
    _,r=decompose(history,window);_,mean,scale=normalize(r,pool.floor)
    _,predicted_r=decompose(np.r_[history,quantiles[4]],window)
    std=max(float(memory_std),1e-6)
    selected=(memory['examples'][:,96:]*scale+mean-predicted_r[96:])/std
    eligible=pool.starts+192<=min(start,pool.memory_end)
    if not eligible.any():raise ValueError('No eligible historical futures')
    # Global control also excludes every not-yet-observed episode.
    global_value=(pool.future[eligible].mean(0)*scale+mean-predicted_r[96:])/std
    return selected.astype(np.float32),global_value.astype(np.float32)


class ErrorCorrection(nn.Module):
    def __init__(self,width=32,rank=16):
        super().__init__()
        self.query=nn.Sequential(nn.Linear(200,width),nn.Tanh())
        self.key=nn.Sequential(nn.Linear(96,width),nn.Tanh())
        self.gate=nn.Sequential(nn.Linear(200,16),nn.Tanh(),nn.Linear(16,1),nn.Sigmoid())
        self.temporal=nn.Sequential(nn.Linear(96,rank,bias=False),nn.Linear(rank,96,bias=False))
        nn.init.zeros_(self.temporal[-1].weight)
    def forward(self,query,keys,errors,weights):
        logits=(self.query(query)[:,None]*self.key(keys)).sum(-1)/math.sqrt(self.key[0].out_features)
        attention=torch.softmax(logits+weights.clamp_min(1e-8).log(),-1)
        context=(attention[...,None]*errors).sum(1)
        return (self.gate(query)*self.temporal(context)).clamp(-3.,3.)


def predict_delta(state,data):
    if state['adopted']=='fixed':
        return state['beta']*np.sum(data['weights'][...,None]*data['errors'],axis=1)
    model=ErrorCorrection();model.load_state_dict(state['model']);model.eval()
    with torch.no_grad():
        return model(torch.as_tensor((data['features']-state['mean'].numpy())/state['scale'].numpy()),
            torch.as_tensor(data['keys']),torch.as_tensor(data['errors']),torch.as_tensor(data['weights'])).numpy()
