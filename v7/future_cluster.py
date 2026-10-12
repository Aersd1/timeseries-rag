"""History-only scratch encoder supervised by future regimes; no pretrained model."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v6.model import PatchTSTBackbone, normalize


def eligible(query_sid, query_start, candidate_sid, candidate_start, span):
    """Same variable, strictly historical complete episodes, no self/overlap.

    Intervals are half open: an old episode ending exactly at query start is
    allowed. Its full history AND continuation must be observed beforehand.
    """
    qs, qt = np.asarray(query_sid)[:, None], np.asarray(query_start)[:, None]
    cs, ct = np.asarray(candidate_sid)[None, :], np.asarray(candidate_start)[None, :]
    return (qs==cs) & (ct+span<=qt) & ~((ct<qt+span)&(qt<ct+span))


def future_features(x, y, floor, horizons):
    """Full ordered continuations at multiple horizons, relative to history mean.

    All scaling statistics come from observed history. No future normalization
    or CDF pooling. Prefixes have equal weight; differences add shape detail.
    """
    past,center,scale=normalize(x,floor)
    future=(y.float()-center[:,None])/scale[:,None]
    parts=[future[:,:h]/math.sqrt(len(horizons)*h) for h in horizons]
    change=torch.diff(torch.cat([past[:,-1,None],future],-1),dim=-1)
    parts.append(.25*change/math.sqrt(y.shape[-1]))
    return torch.cat(parts,-1)


def fit_clusters(features, count=16, seed=0):
    """Call with TRAIN features only; retain prototypes and assignment scale."""
    values=np.asarray(features,dtype=np.float32)
    if not np.isfinite(values).all() or len(values)<count or count<2:
        raise ValueError('Insufficient finite training features for clustering')
    # Dependency-free k-means++ with three seeded restarts, TRAIN CPU data only.
    data=torch.as_tensor(values); rng=np.random.default_rng(seed)
    best_error=float('inf'); best=None
    with torch.no_grad():
        for restart in range(3):
            ids=[int(rng.integers(len(data)))]; nearest=((data-data[ids[0]])**2).sum(-1)
            for _ in range(1,count):
                mass=nearest.double().numpy(); total=mass.sum()
                chosen=int(rng.choice(len(data),p=mass/total)) if total>0 else int(rng.integers(len(data)))
                ids.append(chosen); nearest=torch.minimum(nearest,((data-data[chosen])**2).sum(-1))
            centers=data[ids].clone()
            for _ in range(50):
                distance=torch.cdist(data,centers).square(); labels=distance.argmin(-1)
                sums=torch.zeros_like(centers).index_add_(0,labels,data)
                counts=torch.bincount(labels,minlength=count)
                updated=sums/counts.clamp_min(1)[:,None]
                empty=counts==0
                if empty.any():
                    updated[empty]=data[distance.min(-1).values.argsort(descending=True)[:int(empty.sum())]]
                converged=torch.allclose(updated,centers,rtol=1e-5,atol=1e-5)
                centers=updated
                if converged: break
            minima=torch.cdist(data,centers).square().min(-1).values
            error=float(minima.mean())
            if error<best_error: best_error,best=error,centers.clone()
        distances=torch.cdist(data,best).square().min(-1).values
        temperature=max(float(distances.median())*.5,1e-3)
    return best.numpy(),temperature


def assignments(features, centers, temperature):
    if temperature<=0:
        raise ValueError('Positive cluster temperature required')
    return (-torch.cdist(features.float(),centers.float()).square()/temperature).softmax(-1)


class FutureClusterEncoder(nn.Module):
    def __init__(self,c,use_clusters=True):
        super().__init__()
        self.c,self.use_clusters=c,use_clusters
        self.backbone=PatchTSTBackbone(c['length'],c['width'],16,8,4,2,c.get('dropout',.1))
        self.embedding=nn.Sequential(nn.Linear(c['width'],c['width']),nn.GELU(),nn.Linear(c['width'],c['dim']))
        self.forecast=nn.Linear(c['dim'],max(c['horizons']))
        self.cluster=nn.Linear(c['width'],c['clusters']) if use_clusters else None

    def forward(self,x,floor):
        # The inference API deliberately accepts no query future.
        past,center,scale=normalize(x,floor)
        hidden=self.backbone(past)
        z=F.normalize(self.embedding(hidden),dim=-1)
        future=self.forecast(z)+past[:,-1,None]
        return dict(z=z,probability=None if self.cluster is None else self.cluster(hidden).softmax(-1),
            forecast=future*scale[:,None]+center[:,None])


def row_loss(logits,target,valid):
    if logits.shape!=target.shape or valid.shape!=target.shape:
        raise ValueError('Pair shapes differ')
    usable=valid.sum(-1)>=2
    masked=logits.masked_fill(~valid,-1e9)
    target=target*valid
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
    loss=-(target*masked.log_softmax(-1)).sum(-1)
    return loss[usable].mean() if usable.any() else logits.sum()*0


def encoder_loss(model,batch,bank_z,bank_probability,centers,cluster_temperature):
    out=model(batch['x'],batch['floor'])
    z=out['z']; valid=batch['eligible']
    logits=(z[:,None]*bank_z[batch['candidates']]).sum(-1)/.1
    teacher=(-batch['future_distance']/.5).masked_fill(~valid,-1e9).softmax(-1).detach()
    pair=row_loss(logits,teacher,valid)
    _,center,scale=normalize(batch['x'],batch['floor'])
    truth=(batch['y']-center[:,None])/scale[:,None]
    pred=(out['forecast']-center[:,None])/scale[:,None]
    forecast=sum(F.smooth_l1_loss(pred[:,:h],truth[:,:h]) for h in model.c['horizons'])/len(model.c['horizons'])
    scaled=z*math.sqrt(z.shape[-1]); centered=scaled-scaled.mean(0)
    variance=F.relu(.5-torch.sqrt(centered.square().mean(0)+1e-4)).mean()
    loss=pair+forecast+.1*variance
    cluster_ce,cluster_pair=z.sum()*0,z.sum()*0
    if model.use_clusters:
        feature=future_features(batch['x'],batch['y'],batch['floor'],model.c['horizons'])
        target=assignments(feature,centers,cluster_temperature).detach()
        p=out['probability'].clamp_min(1e-8)
        cluster_ce=-(target*p.log()).sum(-1).mean()
        # Historical continuations are known keys; only the query tendency is predicted.
        distance=((p[:,None].sqrt()-bank_probability[batch['candidates']].sqrt())**2).sum(-1)
        cluster_pair=row_loss(-distance/.1,teacher,valid)
        loss=loss+.5*cluster_ce+.5*cluster_pair
    return loss,dict(pair=pair,forecast=forecast,cluster_ce=cluster_ce,cluster_pair=cluster_pair)


def rank_from_history(out,past,future,bank_z,bank_probability,mode,top_k=5):
    """No query y accepted: future means the OLD candidates' known continuations.

    Recall 50, then keep five ordinary neighbors, matching the A50 policy.
    Candidate-candidate overlap is permitted; query-candidate overlap is banned
    independently by the caller's chronological eligibility mask.
    """
    if mode=='embedding':
        distance=np.sum((bank_z-out['z'])**2,axis=-1)
    elif mode in ['future_cluster','history_future_cluster']:
        distance=np.sum((np.sqrt(bank_probability)-np.sqrt(out['probability']))**2,axis=-1)
    elif mode=='history_future_regression':
        distance=np.mean([np.mean((future[:,:h]-out['normalized_forecast'][:h])**2,axis=-1)
            for h in out['horizons']],axis=0)
    else:
        raise ValueError('Unknown retrieval mode')
    history_all=np.mean((past-out['normalized_history'])**2,axis=-1)
    recall=np.argsort(history_all if mode.startswith('history_future') else distance,kind='stable')[:min(50,len(distance))]
    history=history_all[recall]
    score=distance[recall]
    if mode!='embedding':
        score=.8*score/max(float(np.median(score)),1e-6)+.2*history/max(float(np.median(history)),1e-6)
    selected=np.argsort(score,kind='stable')[:min(top_k,len(recall))]
    indices=recall[selected]; scores=score[selected]
    weights=np.exp(-(scores-scores.min())/max(float(np.median(scores)),1e-6)); weights/=weights.sum()
    prediction=(weights[:,None]*future[indices]).sum(0)*out['scale']+out['center']
    return dict(indices=indices,weights=weights,prediction=prediction,recall=recall,scores=scores)
