"""One VQ vocabulary, two readouts, one small causal token forecaster."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class UnifiedTokens(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        mc = config['model']; self.p = mc['primitive']; d = mc['dim']; hidden = mc['hidden']; v = mc['vocab']
        self.encoder = nn.Sequential(nn.Linear(self.p, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.codebook = nn.Embedding(v, d)
        nn.init.uniform_(self.codebook.weight, -1/math.sqrt(d), 1/math.sqrt(d))
        self.decoder = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, self.p))
        self.prior_logits = nn.Parameter(torch.zeros(v))
        self.project = nn.Linear(d+3, hidden)
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.shape_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.predictive_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.forecaster = nn.Linear(hidden, max(config['horizons']))

    def forward(self, x, scale_floor=None):
        if x.ndim != 2 or x.shape[1] < 1: raise ValueError('Expected batch x raw context length')
        batch, length = x.shape; pad = (-length) % self.p
        padded = F.pad(x, (0, pad)); segments = padded.reshape(batch, -1, self.p)
        mask = (torch.arange(segments.shape[1]*self.p, device=x.device) < length).reshape(1, -1, self.p)
        duration = mask.sum(-1).expand(batch, -1)
        mu = (segments*mask).sum(-1)/duration
        centered = (segments-mu[..., None])*mask
        sigma = ((centered.square().sum(-1)/duration).clamp_min(0)).sqrt()
        normalized = centered/sigma.clamp_min(1e-6)[..., None]
        z = self.encoder(normalized)
        # Quantization/rate arithmetic stays float32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dist = (z.float().square().sum(-1, keepdim=True)+self.codebook.weight.float().square().sum(-1)
                    -2*z.float() @ self.codebook.weight.float().T)
        ids = dist.argmin(-1)
        quantized = self.codebook(ids)
        straight = z+(quantized-z).detach()
        vq_per_item = (quantized-z.detach()).square().mean((-1,-2))+self.config['model']['commitment']*(z-quantized.detach()).square().mean((-1,-2))
        vq = vq_per_item.mean()
        assignments = F.softmax(-dist/self.config['temperatures']['assignment'], dim=-1)
        rate_per_item = -(assignments*F.log_softmax(self.prior_logits.float(), dim=-1)).sum(-1).mean(-1)/math.log(2)
        rate = rate_per_item.mean()
        reconstruction = (self.decoder(straight)*sigma[..., None]+mu[..., None]).reshape(batch, -1)[:, :length]
        # Level/scale are computed from TOKEN metadata; no hidden raw bypass.
        qmu = (mu*duration).sum(-1)/length
        qvar = ((sigma.square()+(mu-qmu[:, None]).square())*duration).sum(-1)/length
        floor = torch.full_like(qmu, 1e-6) if scale_floor is None else scale_floor.to(x.device)
        qscale = torch.maximum(qvar.clamp_min(0).sqrt(), floor)
        side = torch.stack(((mu-qmu[:, None])/qscale[:, None], sigma/qscale[:, None], duration/self.p), dim=-1)
        features = self.project(torch.cat((straight, side), dim=-1))
        sequence, _ = self.temporal(features)
        pooled = (features*duration[..., None]).sum(1)/length
        shape = F.normalize(self.shape_head(pooled), dim=-1)
        predictive = F.normalize(self.predictive_head(sequence[:, -1]), dim=-1)
        prediction = qmu[:, None]+qscale[:, None]*self.forecaster(sequence[:, -1])
        return dict(ids=ids, duration=duration, mu=mu, sigma=sigma, reconstruction=reconstruction,
                    shape=shape, predictive=predictive, prediction=prediction, vq=vq, rate=rate,
                    scale=qscale, mean=qmu, rate_per_item=rate_per_item, vq_per_item=vq_per_item)


def masked_teacher(distance, mask, temperature):
    return F.softmax((-distance.float()/temperature).masked_fill(~mask, -1e9), dim=-1)


def distill(teacher, logits, mask):
    # KL(P_teacher || P_student), reduced over candidates then averaged over Q.
    log_student = F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)
    return F.kl_div(log_student, teacher, reduction='batchmean')


def objective(model, batch):
    c = model.config; q, candidates = batch['query'], batch['candidates']; b, n, length = candidates.shape
    raw = torch.cat((q[:, None], candidates), dim=1).reshape(-1, length)
    floor = batch['scale_floor'][:, None].expand(b, n+1).reshape(-1)
    out = model(raw, floor)
    shape = out['shape'].reshape(b, n+1, -1); pred = out['predictive'].reshape(b, n+1, -1)
    tau = c['temperatures']; valid = batch['valid']
    ps = masked_teacher(batch['past_distance'], valid, tau['shape'])
    retrieval = distill(ps, torch.einsum('bd,bnd->bn', shape[:, 0], shape[:, 1:])/tau['student'], valid)
    # Separate distributions for each H, not a softmax of averaged NMSE.
    future = sum(distill(masked_teacher(batch['future_nmse'][:, :, j], valid, tau['future']),
                        torch.einsum('bd,bnd->bn', pred[:, 0], pred[:, 1:])/tau['student'], valid)
                 for j in range(len(c['horizons'])))/len(c['horizons'])
    scales = out['scale'].reshape(b, n+1)
    reconstruction_error = ((out['reconstruction'].reshape(b, n+1, length)-raw.reshape(b, n+1, length))/scales[..., None]).square().mean(-1)
    all_valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=q.device), valid), dim=1)
    rec = reconstruction_error[all_valid].mean()
    forecast = out['prediction'].reshape(b, n+1, -1)[:, 0]
    forecasting = sum(((forecast[:, :h]-batch['target'][:, :h])/scales[:, 0, None]).square().mean()
                      for h in c['horizons'])/len(c['horizons'])
    losses = dict(rate=out['rate_per_item'].reshape(b,n+1)[all_valid].mean(), reconstruction=rec,
                  retrieval=retrieval, future=future, forecast=forecasting,
                  vq=out['vq_per_item'].reshape(b,n+1)[all_valid].mean())
    total = sum(c['loss_weights'][name]*value for name, value in losses.items())
    return total, losses
