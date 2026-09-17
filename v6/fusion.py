"""Validation-only convex forecast fusion using aggregate error Gram matrices.

No raw curves or query futures need to be saved. No neural training occurs here.
Raw retrieval predictions remain available as a separately scored baseline.
"""
import numpy as np
from scipy.optimize import minimize
from .common import read_json, write_json


BASES = ('persistence','encoder_forecast','probabilistic_prior')


class FusionStats:
    def __init__(self,horizons):
        self.edges=[0]+sorted(set(horizons)); self.grams={}; self.count=0

    def add(self,forecasts,target,scale):
        for method in forecasts:
            if not method.startswith(('learned_leaves','joint_leaves')): continue
            names=[*BASES,method]
            errors=np.stack([(np.asarray(forecasts[name],dtype=float)-target)/scale for name in names])
            values=[errors[:,a:b] @ errors[:,a:b].T/(b-a) for a,b in zip(self.edges[:-1],self.edges[1:])]
            if method not in self.grams: self.grams[method]=np.zeros_like(values)
            self.grams[method]+=values
        self.count+=1

    def export(self,run):
        return dict(version=6,split=run['split'],complete=run['complete'],data_id=run['data_id'],
            checkpoint_sha256=run['checkpoint_sha256'],index_sha256=run['index_sha256'],
            scope=run['config']['evaluation']['scope'],k=run['config']['evaluation']['k'],
            oversample=run['config']['index']['oversample'],time_budget_ms=run['config']['evaluation']['time_budget_ms'],
            edges=self.edges,queries=self.count,grams={k:(v/self.count).tolist() for k,v in self.grams.items()})


def calibrate(results,output,method='learned_leaves0'):
    from pathlib import Path
    stats=read_json(Path(results)/'fusion_stats.json')
    if stats['split']!='validation' or not stats['complete']:
        raise ValueError('Fusion fitting requires a complete VALIDATION run, never test')
    if method not in stats['grams']: raise ValueError('Method not present in validation statistics')
    weights=[]; scores=[]
    for gram in stats['grams'][method]:
        g=np.array(gram,dtype=float); g=(g+g.T)/2
        if g.shape!=(4,4) or not np.isfinite(g).all(): raise ValueError('Invalid fusion statistics')
        # The simplex includes every individual predictor as a vertex.
        initial=np.eye(4)[int(g.diagonal().argmin())]
        fit=minimize(lambda w:float(w @ g @ w),initial,jac=lambda w:2*g@w,method='SLSQP',
                     bounds=[(0,1)]*4,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(4)}],
                     options={'ftol':1e-12,'maxiter':200})
        if not fit.success: raise RuntimeError(f'Fusion optimizer failed: {fit.message}')
        w=np.clip(fit.x,0,1); w/=w.sum()
        if w@g@w > initial@g@initial+1e-10: w=initial
        weights.append(w.tolist()); scores.append(float(w@g@w))
    artifact={k:v for k,v in stats.items() if k!='grams'}
    artifact.update(method=method,names=[*BASES,method],weights=weights,validation_segment_nmse=scores)
    write_json(output,artifact)
    return artifact


def check(fusion, data_id, checkpoint_sha256, index_sha256, c, method):
    if fusion.get('split')!='validation' or not fusion.get('complete'): raise ValueError('Invalid fusion artifact')
    for key,expected in [('data_id',data_id),('checkpoint_sha256',checkpoint_sha256),('index_sha256',index_sha256),
                         ('method',method),('scope',c['evaluation']['scope']),('k',c['evaluation']['k']),
                         ('oversample',c['index']['oversample']),('time_budget_ms',c['evaluation']['time_budget_ms']),
                         ('edges',[0]+sorted(set(c['horizons'])))]:
        if fusion.get(key)!=expected: raise ValueError(f'Fusion policy/identity mismatch: {key}')
    w=np.asarray(fusion['weights'])
    if w.shape!=(len(c['horizons']),4) or not np.isfinite(w).all() or np.any(w<0) or not np.allclose(w.sum(1),1):
        raise ValueError('Invalid convex fusion weights')
    if fusion['names']!=[*BASES,method]: raise ValueError('Invalid fusion methods')


def apply(fusion,forecasts):
    predictions=np.stack([np.asarray(forecasts[name],dtype=float) for name in fusion['names']])
    result=np.empty(fusion['edges'][-1],dtype=float)
    for a,b,w in zip(fusion['edges'][:-1],fusion['edges'][1:],fusion['weights']):
        result[a:b]=np.asarray(w) @ predictions[:,a:b]
    return result
