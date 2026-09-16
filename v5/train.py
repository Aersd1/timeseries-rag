"""Server-only training, deterministic sampling, validation selection, resume."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .common import read_json, write_json, fingerprint, environment, validate_config
from .data import Store
from .teacher import Labels
from .model import UnifiedTokens, objective


class TokenDataset(Dataset):
    def __init__(self, store_path, labels_path, split, config):
        self.store = Store(store_path); self.labels = Labels(labels_path, split); self.split = split; self.c = config; self.epoch = 0
        if self.labels.meta['data_id'] != self.store.meta['data_id']: raise ValueError('Wrong data for labels')
        if self.labels.meta['length'] != config['length'] or self.labels.meta['horizons'] != config['horizons']:
            raise ValueError('Teacher/model context or horizons differ')

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        r = self.labels[i]; sid, p = int(r['sid']), int(r['start']); m = self.c['length']; n = self.c['training']['candidates']
        valid_ids = np.flatnonzero(r['valid']); rng = np.random.default_rng(self.c['seed']+i+self.epoch*len(self))
        if self.split == 'train' and len(valid_ids) > n:
            anchors = valid_ids[:min(8, n)]
            hard = np.flatnonzero(r['hard_negative']); rng.shuffle(hard)
            selected = list(dict.fromkeys([*anchors.tolist(), *hard[:n//4].tolist()]))[:n]
            remaining = np.setdiff1d(valid_ids, selected)
            selected += rng.choice(remaining, size=n-len(selected), replace=False).tolist()
            ids = np.array(selected)
        else: ids = valid_ids[:n]
        # Repeat one eligible candidate for padding; mask removes it from KL/rec.
        ids = np.r_[ids, np.full(n-len(ids), ids[0])].astype(int)
        mask = np.arange(n) < min(n, len(valid_ids))
        return dict(query=self.store.window(sid, p, m), candidates=np.array([self.store.window(sid, pos, m) for pos in r['candidates'][ids]]),
                    target=self.store.window(sid, p+m, max(self.c['horizons'])), valid=mask,
                    past_distance=r['past_distance'][ids], future_nmse=r['future_nmse'][ids],
                    scale_floor=np.float32(max(self.store.series[sid]['memory_std']*1e-4, 1e-6)))


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=True)
    model = UnifiedTokens(state['config']).to(device); model.load_state_dict(state['model']); model.eval()
    return model, state


def fit(store_path, labels_path, output, device='cpu', resume=False, config_path=None):
    original = read_json(Path(store_path)/'config.json')
    c = validate_config(read_json(config_path)) if config_path else original
    for key in ('length', 'horizons', 'split'):
        if c[key] != original[key]: raise ValueError(f'Training override may not change {key}')
    out = Path(output)
    if out.exists() and not resume: raise FileExistsError('Use a fresh training directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    if device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    train = TokenDataset(store_path, labels_path, 'train', c); validation = TokenDataset(store_path, labels_path, 'validation', c)
    if not len(train) or not len(validation): raise ValueError('Need nonempty train AND validation; test is never used for checkpoint selection')
    model = UnifiedTokens(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['lr'], weight_decay=c['training']['weight_decay'])
    start, best = 0, float('inf')
    amp = bool(c['training']['amp'] and device.startswith('cuda'))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if resume:
        ckpt = torch.load(out/'last.pt', map_location=device, weights_only=True)
        if ckpt['data_id'] != train.store.meta['data_id'] or ckpt['config'] != c: raise ValueError('Resume dataset/config mismatch')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer']); scaler.load_state_dict(ckpt['scaler'])
        start, best = ckpt['epoch']+1, ckpt['best_validation']
        torch.set_rng_state(ckpt['rng_cpu'].cpu())
        if device.startswith('cuda') and ckpt['rng_cuda']: torch.cuda.set_rng_state_all([t.cpu() for t in ckpt['rng_cuda']])
    write_json(out/'config.json', c)
    write_json(out/'environment.json', dict(**environment(), torch=torch.__version__, device=device,
                 gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None,
                 parameters=sum(p.numel() for p in model.parameters()), data_id=train.store.meta['data_id']))
    history = read_json(out/'history.json') if resume and (out/'history.json').exists() else []
    for epoch in range(start, c['training']['epochs']):
        began = time.perf_counter(); train.epoch = epoch
        generator = torch.Generator().manual_seed(c['seed']+epoch)
        train_loader = DataLoader(train, batch_size=c['training']['batch_size'], shuffle=True,
                                  generator=generator, num_workers=0)
        model.train(); totals = {}; count = 0
        for batch in train_loader:
            batch = {k:v.to(device) for k,v in batch.items()}; optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                loss, parts = objective(model, batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; inspect input scales/temperatures')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['training']['clip_grad'], error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); count += len(batch['query'])
            for k,v in parts.items(): totals[k] = totals.get(k, 0.)+float(v.detach())*len(batch['query'])
        model.eval(); val_error, val_count = 0., 0
        with torch.no_grad():
            for batch in DataLoader(validation, batch_size=c['training']['batch_size'], num_workers=0):
                q = batch['query'].to(device); target = batch['target'].to(device)
                pred = model(q, batch['scale_floor'].to(device))
                val_error += sum(float(((pred['prediction'][:, :h]-target[:, :h])/pred['scale'][:, None]).square().mean())
                                 for h in c['horizons'])/len(c['horizons'])*len(q); val_count += len(q)
        score = val_error/val_count; improved = score < best; best = min(best, score)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     config=c, data_id=train.store.meta['data_id'], epoch=epoch, best_validation=best,
                     rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
        temp = out/'last.tmp'; torch.save(state, temp); temp.replace(out/'last.pt')
        if improved:
            temp = out/'best.tmp'; torch.save(state, temp); temp.replace(out/'best.pt')
        history.append(dict(epoch=epoch, validation_nmse=score, seconds=time.perf_counter()-began,
                            **{k:v/count for k,v in totals.items()}))
        write_json(out/'history.json', history); print(history[-1], flush=True)
    return dict(best_validation=best, epochs=len(history), training_queries=len(train), validation_queries=len(validation))
