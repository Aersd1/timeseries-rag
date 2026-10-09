"""LoRA Moirai 2 with optional attention to retrieved historical examples.

Query futures are loss targets only. Retrieved examples contain old histories
and their already-observed continuations, normalized using histories only.
"""
import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from .moirai import MoiraiAdapter


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.b.weight)
        self.scale = 1.0

    def forward(self, x):
        return self.base(x)+self.scale*self.b(self.a(x))


class RetrievalAttention(nn.Module):
    def __init__(self, model_dim, history_length, horizon, patch=16, width=128):
        super().__init__()
        self.history_length, self.horizon, self.patch = history_length, horizon, patch
        self.width = width
        self.past_tokens = math.ceil(history_length/patch)
        self.future_tokens = math.ceil(horizon/patch)
        n = self.past_tokens+self.future_tokens
        self.patch_embed = nn.Linear(patch*2, width)
        self.position = nn.Parameter(torch.randn(1, 1, n, width)*.02)
        self.phase = nn.Embedding(2, width)
        self.query_norm = nn.LayerNorm(model_dim)
        self.q = nn.Linear(model_dim, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, model_dim, bias=False)
        nn.init.zeros_(self.output.weight)  # Preserve pretrained forecasts initially.

    def memory(self, examples, weights, valid):
        if examples.ndim != 3 or examples.shape[-1] != self.history_length+self.horizon:
            raise ValueError('Expected [batch, candidates, history+future] examples')
        if weights.shape != examples.shape[:2] or valid.shape != weights.shape:
            raise ValueError('Invalid candidate weights or masks')
        if not torch.isfinite(examples).all() or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid retrieval memory')
        patches, observations = [], []
        for values in [examples[..., :self.history_length], examples[..., self.history_length:]]:
            pad = (-values.shape[-1]) % self.patch
            patches.append(F.pad(values, (0, pad)).unfold(-1, self.patch, self.patch))
            observations.append(F.pad(torch.ones_like(values), (0, pad)).unfold(-1, self.patch, self.patch))
        values, mask = torch.cat(patches, -2), torch.cat(observations, -2)
        tokens = self.patch_embed(torch.cat([values, mask], -1))+self.position
        phase = torch.cat([torch.zeros(self.past_tokens), torch.ones(self.future_tokens)]).long().to(examples.device)
        tokens = tokens+self.phase(phase)[None, None]
        n = tokens.shape[-2]
        bias = weights.clamp_min(1e-8).log().masked_fill(~valid, -1e9)
        bias = bias[..., None].expand(-1, -1, n).flatten(1)
        return tokens.flatten(1, 2), bias, valid.any(-1)

    def forward(self, hidden, memory):
        tokens, bias, any_valid = memory
        # Official recursive forecasts add a quantile-trajectory batch axis.
        if hidden.shape[0] != tokens.shape[0]:
            raise ValueError('Query/memory batch mismatch')
        if hidden.ndim == 4:
            count = hidden.shape[1]
            tokens = tokens[:, None].expand(-1, count, -1, -1)
            bias = bias[:, None].expand(-1, count, -1)
            any_valid = any_valid[:, None].expand(-1, count)
        elif hidden.ndim != 3:
            raise ValueError('Unexpected Moirai representation shape')
        query, key, value = self.q(self.query_norm(hidden)), self.k(tokens), self.v(tokens)
        attention = torch.softmax(query@key.transpose(-1, -2)/math.sqrt(self.width)+bias[..., None, :], dim=-1)
        update = self.output(attention@value)
        return hidden+update*any_valid[..., None, None]


class GatedRetrievalAttention(RetrievalAttention):
    """Legacy token memory with an explicit, query-dependent residual gate."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Sequential(nn.Linear(self.width*2+4, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.constant_(self.gate[-1].bias, -2.)
        self.last_gate = None

    def forward(self, hidden, memory):
        tokens, bias, any_valid, quality = memory
        if hidden.ndim == 4:
            tokens = tokens[:, None].expand(-1, hidden.shape[1], -1, -1)
            bias = bias[:, None].expand(-1, hidden.shape[1], -1)
            any_valid = any_valid[:, None].expand(-1, hidden.shape[1])
            quality = quality[:, None].expand(-1, hidden.shape[1], -1)
        elif hidden.ndim != 3:
            raise ValueError('Unexpected Moirai representation shape')
        query, key, value = self.q(self.query_norm(hidden)), self.k(tokens), self.v(tokens)
        attention = torch.softmax(query@key.transpose(-1, -2)/math.sqrt(self.width)+bias[..., None, :], -1)
        context = attention@value
        features = quality[..., None, :].expand(*query.shape[:-1], 4)
        gate = torch.sigmoid(self.gate(torch.cat([query, context, features], -1)))
        gate = gate*any_valid[..., None, None]
        self.last_gate = gate.detach()
        return hidden+gate*self.output(context)


class PairedRetrievalAttention(GatedRetrievalAttention):
    """One key per old history, with its corresponding continuation as value.

    Separate whole-window projections preserve the ordered history/continuation
    within each pair. A continuation cannot change its own relevance key.
    """
    def __init__(self, model_dim, history_length, horizon, patch=16, width=128):
        super().__init__(model_dim, history_length, horizon, patch, width)
        # These token-only parameters belong exclusively to the legacy branch.
        del self.patch_embed, self.position, self.phase
        self.history_embed = nn.Sequential(nn.Linear(history_length, width), nn.GELU(), nn.Linear(width, width))
        self.future_embed = nn.Sequential(nn.Linear(horizon, width), nn.GELU(), nn.Linear(width, width))

    def memory(self, examples, weights, valid):
        return (self.history_embed(examples[..., :self.history_length]),
            self.future_embed(examples[..., self.history_length:]),
            weights.clamp_min(1e-8).log().masked_fill(~valid, -1e9), valid.any(-1))

    def forward(self, hidden, memory):
        past, future, bias, any_valid, quality = memory
        if hidden.ndim == 4:
            count = hidden.shape[1]
            past, future = [v[:, None].expand(-1, count, -1, -1) for v in [past, future]]
            bias = bias[:, None].expand(-1, count, -1)
            any_valid = any_valid[:, None].expand(-1, count)
            quality = quality[:, None].expand(-1, count, -1)
        elif hidden.ndim != 3:
            raise ValueError('Unexpected Moirai representation shape')
        query, key, value = self.q(self.query_norm(hidden)), self.k(past), self.v(future)
        attention = torch.softmax(query@key.transpose(-1, -2)/math.sqrt(self.width)+bias[..., None, :], -1)
        context = attention@value
        features = quality[..., None, :].expand(*query.shape[:-1], 4)
        gate = torch.sigmoid(self.gate(torch.cat([query, context, features], -1)))
        gate = gate*any_valid[..., None, None]
        self.last_gate, self.last_attention = gate.detach(), attention.detach()
        return hidden+gate*self.output(context)


def memory_quality(x, examples, weights, valid, history_length):
    """Observed-history mismatch, continuation disagreement, entropy and count.

    No current query future is accepted. All features respect the effective
    candidate mask (including training dropout) and are permutation invariant.
    """
    w = weights*valid
    w = w/w.sum(-1, keepdim=True).clamp_min(1e-8)
    past, future = examples[..., :history_length], examples[..., history_length:]
    q = (x-x.mean(-1, keepdim=True))/x.std(-1, correction=0, keepdim=True).clamp_min(1e-6)
    mismatch = ((past-q[:, None])**2).mean(-1)
    disagreement = ((future-(future*w[..., None]).sum(1, keepdim=True))**2).mean(-1)
    return torch.stack([valid.float().mean(-1),
        -(w*w.clamp_min(1e-8).log()).sum(-1)/math.log(max(weights.shape[-1], 2)),
        torch.log1p((disagreement*w).sum(-1)), torch.log1p((mismatch*w).sum(-1))], -1)


class RetrievalFineTune(nn.Module):
    def __init__(self, adapter, history_length, horizon, use_retrieval=True, rank=8,
            fusion='legacy', candidate_dropout=0., memory_dropout=0.):
        super().__init__()
        if fusion not in ['legacy', 'gated', 'paired'] or not (0 <= candidate_dropout <= 1 and 0 <= memory_dropout <= 1):
            raise ValueError('Invalid retrieval fusion/dropout configuration')
        self.module = adapter.module
        self.module.requires_grad_(False)
        self.architecture, self.identity = adapter.architecture, adapter.identity
        self.history_length, self.horizon = history_length, horizon
        self.use_retrieval, self.rank = use_retrieval, rank
        self.fusion, self.candidate_dropout, self.memory_dropout = fusion, candidate_dropout, memory_dropout
        for layer in self.module.encoder.layers[-2:]:
            for name in ('q_proj', 'v_proj'):
                attention = layer.self_attn
                setattr(attention, name, LoRALinear(getattr(attention, name), rank))
        attention = {'legacy':RetrievalAttention, 'gated':GatedRetrievalAttention, 'paired':PairedRetrievalAttention}[fusion]
        self.retrieval = attention(self.module.d_model, history_length, horizon, self.module.patch_size) if use_retrieval else None
        self._memory = None
        if self.retrieval is not None:
            self.module.encoder.register_forward_hook(self._condition)

    def _condition(self, module, args, hidden):
        return hidden if self._memory is None else self.retrieval(hidden, self._memory)

    @contextmanager
    def condition(self, examples=None, weights=None, valid=None, x=None):
        if self._memory is not None:
            raise RuntimeError('Nested retrieval condition')
        if self.retrieval is not None and examples is not None:
            if examples.ndim != 3 or examples.shape[0] != x.shape[0] or examples.shape[-1] != self.history_length+self.horizon:
                raise ValueError('Invalid example shape')
            if weights.shape != examples.shape[:2] or valid.shape != weights.shape or valid.dtype != torch.bool:
                raise ValueError('Invalid candidate weights or masks')
            if not torch.isfinite(examples).all() or not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Invalid retrieval memory')
            if (valid & (weights <= 0)).any():
                raise ValueError('Valid candidates require positive weights')
            if self.training:
                if self.candidate_dropout:
                    valid = valid & (torch.rand_like(weights) >= self.candidate_dropout)
                if self.memory_dropout:
                    valid = valid & (torch.rand_like(weights[:, :1]) >= self.memory_dropout)
            self._memory = self.retrieval.memory(examples, weights, valid)
            if self.fusion != 'legacy':
                self._memory += (memory_quality(x, examples, weights, valid, self.history_length),)
        try:
            yield
        finally:
            self._memory = None

    def forward(self, x, examples=None, weights=None, valid=None):
        from uni2ts.model.moirai2 import Moirai2Forecast
        if x.ndim != 2 or x.shape[1] != self.history_length or not torch.isfinite(x).all():
            raise ValueError('Invalid query history')
        predictor = Moirai2Forecast(module=self.module, prediction_length=self.horizon,
            context_length=self.history_length, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(x.device)
        with self.condition(examples, weights, valid, x):
            result = predictor(past_target=x[..., None], past_observed_target=torch.ones_like(x[..., None], dtype=torch.bool),
                past_is_pad=torch.zeros_like(x, dtype=torch.bool))
        return result.float()

    def adaptation_state(self):
        return {name:parameter.detach().cpu().clone() for name, parameter in self.named_parameters() if parameter.requires_grad}

    def load_adaptation(self, state):
        parameters = {name:p for name,p in self.named_parameters() if p.requires_grad}
        if set(parameters) != set(state):
            raise ValueError('Adaptation architecture mismatch')
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(state[name].to(parameter))


def quantile_loss(prediction, target, x, floor, levels, horizons):
    scale = x.std(-1, correction=0).clamp_min(floor)
    residual = (target[:, None]-prediction)/scale[:, None, None]
    levels = torch.as_tensor(levels, device=prediction.device, dtype=prediction.dtype)[None, :, None]
    loss = torch.maximum(levels*residual, (levels-1)*residual)
    return sum(loss[..., :h].mean() for h in horizons)/len(horizons)
