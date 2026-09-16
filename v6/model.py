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
        signature = torch.cat([p.sqrt(), (p.sqrt()[:, :, None] * pool(mu, b).asinh()).flatten(1),
                               (p.sqrt()[:, :, None] * pool(sigma, b).log()).flatten(1)], dim=1)
        return dict(logp=logp, mu=mu, sigma=sigma, mean=mean, std=variance.clamp_min(1e-6).sqrt(),
                    signature=signature, center=center, scale=scale, normalized=z)


def log_components(prior, target):
    y = (target.float() - prior['center'][:, None]) / prior['scale'][:, None]
    return (-0.5 * ((y[:, None] - prior['mu']) / prior['sigma']).square()
            - prior['sigma'].log() - 0.5 * math.log(2 * math.pi))


def prior_loss(prior, target):
    lp = log_components(prior, target).sum(-1) + prior['logp']
    return -lp.logsumexp(-1).mean() / target.shape[-1]


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
        belief_dim = c['model']['components'] * (1 + 2 * c['model']['belief_bins'])
        self.backbone = Backbone(width)
        self.use_belief = c['model'].get('use_belief', True)
        self.readout = nn.Sequential(nn.Linear(width + (belief_dim if self.use_belief else 0), width),
                                     nn.GELU(), nn.Linear(width, dim))
        self.decode_belief = nn.Linear(dim, belief_dim)
        self.decode_future = nn.Linear(dim, max(c['horizons']))

    def forward(self, x, floor):
        with torch.no_grad():
            p = self.prior(x, floor)
        hidden = self.backbone(p['normalized'])
        features = torch.cat([hidden, p['signature']], -1) if self.use_belief else hidden
        embedding = F.normalize(self.readout(features).float(), dim=-1)
        history = F.normalize(pool(p['normalized'], self.c['model']['history_bins']), dim=-1)
        belief = F.normalize(p['signature'], dim=-1)
        return dict(learned=embedding, history=history, belief=belief, prior=p,
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
    logits = (out['learned'] @ out['learned'].T).float() / model.c['training']['temperature']
    logits = logits.masked_fill(~allowed, -1e9)
    per_row = -(target * logits.log_softmax(-1)).sum(-1)
    pair = per_row[usable].mean() if usable.any() else out['learned'].sum() * 0
    future = sum(F.smooth_l1_loss(out['forecast'][:, :h], y[:, :h]) for h in model.c['horizons']) / len(model.c['horizons'])
    belief = F.smooth_l1_loss(out['belief_reconstruction'], p['signature']) if model.use_belief else pair * 0
    total = pair + model.c['training']['future_weight'] * future + model.c['training']['belief_weight'] * belief
    return total, dict(pair=pair, future=future, belief=belief, usable_fraction=usable.float().mean())
