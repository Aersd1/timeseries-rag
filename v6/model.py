"""Conditional Bayesian DAG X -> Z, (X,Z) -> Y; belief-guided encoder.

Weights are point estimates, not a Bayesian posterior over neural weights.
Y coordinates are conditionally independent given X and shared regime Z.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def normalize(x, floor):
    x = x.float()
    mean = x.mean(-1)
    scale = x.std(-1, correction=0).clamp_min(floor.float())
    return (x - mean[:, None]) / scale[:, None], mean, scale


def pool(x, bins):
    shape = x.shape[:-1]
    return F.adaptive_avg_pool1d(x.reshape(-1, 1, x.shape[-1]), bins).reshape(*shape, bins)


CDF_GRID = (-8., -4., -2., -1., 0., 1., 2., 4., 8.)


def belief_dim(c):
    m = c['model']
    return m['belief_bins']*(len(CDF_GRID)+2) if m.get('belief_signature','components') == 'cdf' else m['components']*(1+2*m['belief_bins'])


def signature(logp, mu, sigma, bins, kind='components'):
    p = logp.exp()
    if kind == 'components':
        return torch.cat([p.sqrt(), (p.sqrt()[:,:,None]*pool(mu,bins).asinh()).flatten(1),
                          (p.sqrt()[:,:,None]*pool(sigma,bins).log()).flatten(1)],1)
    if kind != 'cdf': raise ValueError('Unknown belief signature')
    # A distribution is unchanged when its mixture components are relabelled.
    grid = torch.tensor(CDF_GRID, device=mu.device, dtype=mu.dtype)
    cdf = 0.5*(1+torch.erf((grid[None,None,None,:]-mu[:,:,:,None])/(sigma[:,:,:,None]*math.sqrt(2))))
    cdf = (p[:,:,None,None]*cdf).sum(1).transpose(1,2)
    mean = (p[:,:,None]*mu).sum(1)
    var = (p[:,:,None]*(sigma.square()+(mu-mean[:,None]).square())).sum(1)
    return torch.cat([pool(cdf,bins).flatten(1),pool(mean,bins).asinh(),pool(var,bins).clamp_min(1e-6).log()/2],1)


class Backbone(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, hidden, 7, padding=3), nn.GELU(),
                                 nn.Conv1d(hidden, hidden, 5, stride=4, padding=2), nn.GELU())
        self.readout = nn.Linear(hidden * 4, hidden)

    def forward(self, x):
        return F.gelu(self.readout(F.adaptive_avg_pool1d(self.net(x[:, None]), 4).flatten(1)))


class FuturePrior(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        h, k, width = max(c['horizons']), c['model']['components'], c['model']['hidden']
        self.h, self.k = h, k
        self.backbone = Backbone(width)
        self.head = nn.Linear(width, k * (1 + 2 * h))

    def forward(self, x, floor):
        z, center, scale = normalize(x, floor)
        raw = self.head(self.backbone(z)).float().reshape(-1, self.k, 1 + 2 * self.h)
        logp = raw[:, :, 0].log_softmax(-1)
        # Start near persistence; bounded scales keep AMP density arithmetic finite.
        mu = z[:, -1, None, None] + raw[:, :, 1:1 + self.h]
        sigma = (F.softplus(raw[:, :, 1 + self.h:]) + 0.05).clamp_max(1000)
        p = logp.exp()
        mean = (p[:, :, None] * mu).sum(1)
        variance = (p[:, :, None] * (sigma.square() + mu.square())).sum(1) - mean.square()
        b = self.c['model']['belief_bins']
        # Component-specific signatures preserve more than just the mixture mean.
        belief_features = signature(logp,mu,sigma,b,self.c['model'].get('belief_signature','components'))
        return dict(logp=logp, mu=mu, sigma=sigma, mean=mean, std=variance.clamp_min(1e-6).sqrt(),
                    signature=belief_features, center=center, scale=scale, normalized=z)


def log_components(prior, target):
    y = (target.float() - prior['center'][:, None]) / prior['scale'][:, None]
    return (-0.5 * ((y[:, None] - prior['mu']) / prior['sigma']).square()
            - prior['sigma'].log() - 0.5 * math.log(2 * math.pi))


def prior_loss(prior, target, horizons=None):
    components=log_components(prior,target)
    horizons=horizons or [target.shape[-1]]
    if min(horizons)<1 or max(horizons)>target.shape[-1]: raise ValueError('Invalid likelihood horizon')
    # Equal weight per forecast horizon, instead of the longest horizon only.
    return sum(-(components[:,:,:h].sum(-1)+prior['logp']).logsumexp(-1).mean()/h for h in horizons)/len(horizons)


def posterior(prior, target):
    """Bayes responsibilities for diagnostics/TRAINING only, never retrieval."""
    return (log_components(prior, target).sum(-1) + prior['logp']).softmax(-1)


class BeliefEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.prior = FuturePrior(c)
        self.prior.requires_grad_(False)
        width, dim = c['model']['hidden'], c['model']['dim']
        signature_dim = belief_dim(c)
        self.backbone = Backbone(width)
        self.use_belief = c['model'].get('use_belief', True)
        self.readout = nn.Sequential(nn.Linear(width + (signature_dim if self.use_belief else 0), width),
                                     nn.GELU(), nn.Linear(width, dim))
        self.decode_belief = nn.Linear(dim, signature_dim)
        self.decode_future = nn.Linear(dim, max(c['horizons']))
        self.decode_history=nn.Linear(dim,c['model']['history_bins']) if c['model'].get('history_reconstruction',False) else None

    def forward(self, x, floor):
        with torch.no_grad():
            p = self.prior(x, floor)
        hidden = self.backbone(p['normalized'])
        features = torch.cat([hidden, p['signature']], -1) if self.use_belief else hidden
        embedding = F.normalize(self.readout(features).float(), dim=-1)
        history = F.normalize(pool(p['normalized'], self.c['model']['history_bins']), dim=-1)
        belief = F.normalize(p['signature'], dim=-1)
        weight=self.c['index'].get('joint_history_weight',.5)
        joint=torch.cat([math.sqrt(1-weight)*embedding,math.sqrt(weight)*history],dim=-1)
        return dict(learned=embedding, history=history, belief=belief, joint=joint, prior=p,
                    history_reconstruction=self.decode_history(embedding).float() if self.decode_history is not None else None,
                    belief_reconstruction=self.decode_belief(embedding).float(),
                    forecast=self.decode_future(embedding).float() + p['normalized'][:, -1, None])


def encoder_loss(model, batch):
    out = model(batch['x'], batch['floor'])
    p = out['prior']
    y = (batch['y'].float() - p['center'][:, None]) / p['scale'][:, None]
    n = len(y)
    # Training futures define pair similarity; mask self and overlapping episodes.
    span = model.c['length'] + max(model.c['horizons'])
    allowed = ~((batch['sid'][:, None] == batch['sid'][None, :]) &
                ((batch['start'][:, None] - batch['start'][None, :]).abs() < span))
    allowed &= ~torch.eye(n, dtype=torch.bool, device=y.device)
    usable = allowed.sum(-1) >= 2
    dist = sum(torch.cdist(y[:, :h].contiguous(), y[:, :h].contiguous()).square() / h
               for h in model.c['horizons']) / len(model.c['horizons'])
    target = (-dist / model.c['training']['target_temperature']).masked_fill(~allowed, -1e9).softmax(-1).detach()
    teacher_weight = model.c['training'].get('distribution_teacher_weight',0.) if model.use_belief else 0.
    if teacher_weight:
        teacher_dist = torch.cdist(out['belief'].float(),out['belief'].float()).square()
        teacher = (-teacher_dist/model.c['training']['target_temperature']).masked_fill(~allowed,-1e9).softmax(-1)
        target = (1-teacher_weight)*target + teacher_weight*teacher.detach()
    history_weight=model.c['training'].get('history_teacher_weight',0.)
    if history_weight:
        history_dist=torch.cdist(p['normalized'].float(),p['normalized'].float()).square()/model.c['length']
        # Product of future and history kernels: a positive must satisfy both.
        joint_log=target.clamp_min(1e-30).log()-history_weight*history_dist/model.c['training']['target_temperature']
        target=joint_log.masked_fill(~allowed,-1e9).softmax(-1).detach()
    logits = (out['learned'] @ out['learned'].T).float() / model.c['training']['temperature']
    logits = logits.masked_fill(~allowed, -1e9)
    per_row = -(target * logits.log_softmax(-1)).sum(-1)
    pair = per_row[usable].mean() if usable.any() else out['learned'].sum() * 0
    future = sum(F.smooth_l1_loss(out['forecast'][:, :h], y[:, :h]) for h in model.c['horizons']) / len(model.c['horizons'])
    belief = F.smooth_l1_loss(out['belief_reconstruction'], p['signature']) if model.use_belief else pair * 0
    history_loss=F.smooth_l1_loss(out['history_reconstruction'],pool(p['normalized'],model.c['model']['history_bins'])) if out['history_reconstruction'] is not None else pair*0
    # Variance/covariance regularization on the actual retrieval vector.
    z = out['learned'].float()*math.sqrt(out['learned'].shape[1])
    centered = z-z.mean(0)
    variance = F.relu(0.5-torch.sqrt(centered.square().mean(0)+1e-4)).mean() if n>1 else z.sum()*0
    covariance = centered.T @ centered / max(n-1,1)
    offdiag = covariance-torch.diag_embed(covariance.diagonal())
    decorrelation = offdiag.square().sum()/z.shape[1] if n>1 else z.sum()*0
    total = (pair + model.c['training']['future_weight']*future + model.c['training']['belief_weight']*belief
             + model.c['training'].get('variance_weight',0.)*variance
             + model.c['training'].get('covariance_weight',0.)*decorrelation
             + model.c['training'].get('history_reconstruction_weight',0.)*history_loss)
    return total, dict(pair=pair, future=future, belief=belief, variance=variance,covariance=decorrelation,
                      history=history_loss,usable_fraction=usable.float().mean())
