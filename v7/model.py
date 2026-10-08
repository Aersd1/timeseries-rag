"""Variant A: replace V6 Gaussian-prior/CDF features with Moirai hidden states."""
import torch
from torch import nn
from contextlib import contextmanager
from v6.model import BeliefEncoder, make_backbone, normalize


class RepresentationPrior(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self._signature = None

    @contextmanager
    def cached_signature(self, signature):
        """A single loss call may reuse an immutable history-only representation."""
        if self._signature is not None:
            raise RuntimeError('Nested signature overrides are not supported')
        self._signature = signature
        try:
            yield
        finally:
            self._signature = None

    def forward(self, x, floor):
        normalized, center, scale = normalize(x, floor)
        signature = self.adapter.encode(x) if self._signature is None else self._signature
        if signature.shape != (len(x), self.adapter.module.d_model) or signature.device != x.device:
            raise ValueError('Cached representation has an incompatible shape/device')
        return dict(signature=signature.detach(), normalized=normalized, center=center, scale=scale)


class MoiraiEncoder(BeliefEncoder):
    """Retain V6 heads, history branch, joint vectors and training objective."""
    def __init__(self, c, adapter):
        nn.Module.__init__(self)
        self.c = c
        self.prior = RepresentationPrior(adapter)
        self.prior.requires_grad_(False)
        width, dim = c['model']['hidden'], c['model']['dim']
        signature_dim = adapter.module.d_model
        self.backbone = make_backbone(c)
        self.use_belief = True
        self.readout = nn.Sequential(nn.Linear(width+signature_dim, width), nn.GELU(), nn.Linear(width, dim))
        self.decode_belief = nn.Linear(dim, signature_dim)
        self.decode_future = nn.Linear(dim, max(c['horizons']))
        self.decode_history = nn.Linear(dim, c['model']['history_bins']) if c['model'].get('history_reconstruction', False) else None


def load(checkpoint, device='cpu'):
    from .moirai import MoiraiAdapter
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('version') != 7 or state.get('variant') != 'a':
        raise ValueError('Expected variant A checkpoint; V6 weights cannot be reused as its projection')
    adapter = MoiraiAdapter.empty(state['moirai_architecture'], state['moirai_identity'])
    model = MoiraiEncoder(state['config'], adapter)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), state
