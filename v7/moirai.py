"""Pinned official Moirai 2 API; history-only latent and quantile interfaces."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = 'Salesforce/moirai-2.0-R-small'
MODEL_REVISION = '30f43ff08c8494f4943ae1521e9d4e94a0fbb389'
UNI2TS_REVISION = 'cfd46d4510ed8896f263116f32928eede05b0a75'


class MoiraiAdapter(nn.Module):
    """Frozen pretrained model. Architecture is saved in A checkpoints for offline load."""
    def __init__(self, module, architecture, identity):
        super().__init__()
        self.module = module
        self.architecture = architecture
        self.identity = identity
        # Upstream functional attention uses this value even in eval mode.
        # The official small checkpoint already has 0; enforce it for all frozen adapters.
        for layer in self.module.modules():
            if hasattr(layer, 'attn_dropout_p'):
                layer.attn_dropout_p = 0.0
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def pretrained(cls, settings, device='cpu'):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from safetensors.torch import load_model
        from uni2ts.model.moirai2 import Moirai2Module
        model_id = settings.get('model_id', MODEL_ID)
        revision = settings.get('revision', MODEL_REVISION)
        if len(revision) != 40 or any(x not in '0123456789abcdef' for x in revision):
            raise ValueError('Pin a full 40-character model commit, not main')
        options = dict(repo_id=model_id, revision=revision, cache_dir=settings.get('cache_dir'),
            allow_patterns=['config.json', 'model.safetensors'])
        # A commit hash is immutable: a complete cached snapshot needs no HTTP check.
        try:
            folder = snapshot_download(**options, local_files_only=True)
            if not all((Path(folder)/name).is_file() for name in options['allow_patterns']):
                raise LocalEntryNotFoundError('Incomplete local Moirai snapshot')
        except LocalEntryNotFoundError:
            if settings.get('local_files_only', False):
                raise
            folder = snapshot_download(**options, local_files_only=False)
        arch = json.loads((Path(folder)/'config.json').read_text())
        module = Moirai2Module(**arch)
        load_model(module, str(Path(folder)/'model.safetensors'), strict=True)
        return cls(module, arch, dict(model_id=model_id, revision=revision,
            uni2ts_revision=UNI2TS_REVISION, feature='last_context_token')).to(device)

    @classmethod
    def empty(cls, architecture, identity):
        from uni2ts.model.moirai2 import Moirai2Module
        return cls(Moirai2Module(**architecture), architecture, identity)

    def train(self, mode=True):
        # Freezing gradients alone does not disable dropout. Always freeze mode too.
        return super().train(False)

    def _history(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=next(self.module.parameters()).device)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or not torch.isfinite(x).all():
            raise ValueError('Expected finite, nonempty [batch, history]')
        if math.ceil(x.shape[1]/self.module.patch_size) > self.module.max_seq_len:
            raise ValueError('History exceeds Moirai token limit')
        return x

    @torch.no_grad()
    def encode(self, x):
        """Read the last causal context representation; never encode a true future."""
        x = self._history(x)
        patch = self.module.patch_size
        pad = (-x.shape[1]) % patch
        target = F.pad(x, (pad, 0)).unfold(-1, patch, patch)
        observed = F.pad(torch.ones_like(x, dtype=torch.bool), (pad, 0), value=False).unfold(-1, patch, patch)
        shape = target.shape[:2]
        sample = torch.ones(shape, dtype=torch.long, device=x.device)
        times = torch.arange(shape[1], device=x.device).expand(shape)
        variate = torch.zeros_like(sample)
        prediction = torch.zeros(shape, dtype=torch.bool, device=x.device)
        captured = []
        hook = self.module.encoder.register_forward_hook(lambda _m, _a, out: captured.append(out))
        try:
            # Delegate scaling, mask, projections and Transformer arithmetic to upstream.
            self.module(target, observed, sample, times, variate, prediction, training_mode=False)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != (*shape, self.module.d_model):
            raise RuntimeError('Moirai encoder API changed; check the pinned Uni2TS revision')
        result = captured[0][:, -1].float()
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite Moirai representation')
        return result

    @torch.no_grad()
    def forecast(self, x, horizon):
        """Official recursive quantile forecast, returned as [batch, quantile, time]."""
        from uni2ts.model.moirai2 import Moirai2Forecast
        x = self._history(x)
        if horizon < 1 or math.ceil(x.shape[1]/self.module.patch_size) + math.ceil(horizon/self.module.patch_size) > self.module.max_seq_len:
            raise ValueError('Invalid horizon or total token limit exceeded')
        predictor = Moirai2Forecast(module=self.module, prediction_length=horizon,
            context_length=x.shape[1], target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0).to(x.device).eval()
        result = predictor(past_target=x[..., None],
            past_observed_target=torch.ones_like(x[..., None], dtype=torch.bool),
            past_is_pad=torch.zeros_like(x, dtype=torch.bool)).float()
        if result.shape != (len(x), self.module.num_quantiles, horizon) or not torch.isfinite(result).all():
            raise FloatingPointError('Invalid Moirai quantile forecast')
        return result
