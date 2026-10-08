"""Train A's V6-compatible projection; keep pretrained Moirai fixed."""
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from v6.data import Windows
from v6.model import encoder_loss
from v6.sampling import HistoryBatches
from v6.common import save_torch, environment
from .common import config, fresh_dir, write_json
from .moirai import MoiraiAdapter
from .model import MoiraiEncoder


class FrozenFeatures(Dataset):
    """Cache only x -> frozen Moirai hidden, never labels or trainable activations."""
    def __init__(self, dataset, adapter, device, batch_size=256):
        self.dataset = dataset
        self.store, self.c, self.refs = dataset.store, dataset.c, dataset.refs
        features = []
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            features.append(adapter.encode(batch['x'].to(device)).cpu())
        self.features = torch.cat(features)
        if not torch.isfinite(self.features).all():
            raise FloatingPointError('Invalid frozen representation cache')

    def __len__(self): return len(self.dataset)

    def __getitem__(self, i):
        return dict(self.dataset[i], moirai_signature=self.features[i])


def loss_for_batch(model, batch):
    if 'moirai_signature' not in batch:
        return encoder_loss(model, batch)
    with model.prior.cached_signature(batch['moirai_signature']):
        return encoder_loss(model, batch)


def fit(store, config_path, output, device='cpu', cache_features=True):
    c = config(config_path)
    torch.manual_seed(c['seed'])
    train, val = Windows(store, c, 'train'), Windows(store, c, 'validation')
    out = fresh_dir(output)
    adapter = MoiraiAdapter.pretrained(c['moirai'], device)
    model = MoiraiEncoder(c, adapter).to(device)
    cache_bytes = (len(train)+len(val))*adapter.module.d_model*4
    cache_limit = 512*1024**2
    if cache_features and cache_bytes>cache_limit:
        print(dict(frozen_feature_cache=False, reason='feature cache exceeds 512 MiB; using streaming inference',
            estimated_bytes=cache_bytes),flush=True)
        cache_features = False
    cache_began = time.perf_counter()
    if cache_features:
        # DataLoader iterator creation consumes the CPU RNG even without shuffle.
        # Preserve the original dropout/sampler stream across this optimization.
        with torch.random.fork_rng(devices=[]):
            train, val = (FrozenFeatures(ds, adapter, device, c['index']['batch_size']) for ds in (train, val))
    cache_seconds = time.perf_counter()-cache_began
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=c['training']['lr'], weight_decay=c['training']['weight_decay'])
    sampler = HistoryBatches(train, c['training']['batch_size'], c['seed'], c['training']['history_neighbor_fraction']) if c['training']['history_neighbor_batches'] else None
    history, best = [], float('inf')
    write_json(out/'config.json', c)
    write_json(out/'environment.json', dict(**environment(), torch=str(torch.__version__),
        device=str(device), train_windows=len(train), validation_windows=len(val), precision='float32',
        frozen_feature_cache=cache_features, feature_cache_seconds=cache_seconds,
        feature_cache_estimated_bytes=cache_bytes, feature_cache_limit_bytes=cache_limit))
    try:
        for epoch in range(c['training']['encoder_epochs']):
            began = time.perf_counter()
            model.train()
            if sampler is not None:
                sampler.epoch = epoch
                loader = DataLoader(train, batch_sampler=sampler)
            else:
                loader = DataLoader(train, batch_size=c['training']['batch_size'], shuffle=True,
                    generator=torch.Generator().manual_seed(c['seed']+epoch))
            total, count = 0., 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_for_batch(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(batch['x']); count += len(batch['x'])
            model.eval()
            score, n = 0., 0
            with torch.no_grad():
                for batch in DataLoader(val, batch_size=c['training']['batch_size']):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss, _ = loss_for_batch(model, batch)
                    score += float(loss)*len(batch['x']); n += len(batch['x'])
            score /= n
            if not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError('Nonfinite validation objective')
            improved = score < best; best = min(best, score)
            history.append(dict(epoch=epoch, train_objective=total/count, validation_objective=score,
                seconds=time.perf_counter()-began))
            state = dict(version=7, variant='a', config=c, data_id=train.store.meta['data_id'],
                model=model.state_dict(), moirai_architecture=adapter.architecture,
                moirai_identity=adapter.identity, epoch=epoch, best=best)
            save_torch(out/'last.pt', state)
            if improved:
                save_torch(out/'best.pt', state)
            write_json(out/'history.json', history)
            print(history[-1], flush=True)
    finally:
        train.store.close(); val.store.close()
    return dict(best_validation=best, epochs=len(history), variant='a')
