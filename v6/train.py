"""Explicit server training; memory-only prior, later encoder, held-out selection."""
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from .common import config, write_json, read_json, environment, save_torch, sha256
from .data import Windows
from .model import FuturePrior, BeliefEncoder, prior_loss, encoder_loss


def load(path, device='cpu', expected_stage=None):
    # V6 never silently falls back to unrestricted pickle loading.
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get('version') != 6 or (expected_stage and state['stage'] != expected_stage):
        raise ValueError('Wrong checkpoint version/stage')
    model = (FuturePrior if state['stage'] == 'prior' else BeliefEncoder)(state['config']).to(device)
    model.load_state_dict(state['model'])
    return model.eval(), state


def fit(store_path, config_path, output, stage, device='cpu', prior_path=None, resume=False):
    c = config(config_path)
    torch.manual_seed(c['seed'])
    train = Windows(store_path, c, 'prior' if stage == 'prior' else 'train')
    validation = Windows(store_path, c, 'validation')
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('Choose a fresh training output or --resume')
    out.mkdir(parents=True, exist_ok=True)
    prior_sha = None
    if stage == 'prior':
        model = FuturePrior(c).to(device)
    else:
        if prior_path is None:
            raise ValueError('Encoder stage requires --prior')
        prior, ps = load(prior_path, device, 'prior')
        if ps['data_id'] != train.store.meta['data_id']:
            raise ValueError('Prior trained on a different store')
        for key in ('length', 'horizons', 'normalization'):
            if ps['config'][key] != c[key]:
                raise ValueError(f'Prior mismatch: {key}')
        model = BeliefEncoder(c).to(device)
        model.prior.load_state_dict(prior.state_dict())
        prior_sha = sha256(prior_path)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=c['training']['lr'], weight_decay=c['training']['weight_decay'])
    amp = bool(c['training']['amp'] and str(device).startswith('cuda'))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best, history = 0, float('inf'), []
    if resume:
        saved = torch.load(out / 'last.pt', map_location=device, weights_only=True)
        if (saved['config'] != c or saved['stage'] != stage or saved['prior_sha256'] != prior_sha
                or saved['data_id'] != train.store.meta['data_id']):
            raise ValueError('Resume identity/config mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start, best, history = saved['epoch'] + 1, saved['best'], saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if amp and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
    write_json(out / 'config.json', c)
    write_json(out / 'environment.json', dict(**environment(), torch=str(torch.__version__), device=str(device),
                                              train_windows=len(train), validation_windows=len(validation)))

    def loss_for(batch):
        if stage == 'prior':
            loss = prior_loss(model(batch['x'], batch['floor']), batch['y'])
            return loss, {'nll_per_point': loss}
        return encoder_loss(model, batch)

    try:
        for epoch in range(start, c['training'][stage + '_epochs']):
            began, totals, count = time.perf_counter(), {}, 0
            model.train()
            loader = DataLoader(train, batch_size=c['training']['batch_size'], shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(c['seed'] + epoch))
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=torch.device(device).type, enabled=amp):
                    loss, parts = loss_for(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; check input scales')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
                n = len(batch['x']); count += n
                for k, v in parts.items():
                    totals[k] = totals.get(k, 0.) + float(v.detach()) * n
            model.eval(); val_sum, val_count = 0., 0
            with torch.no_grad():
                for batch in DataLoader(validation, batch_size=c['training']['batch_size'], num_workers=0):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    val, _ = loss_for(batch)
                    val_sum += float(val) * len(batch['x']); val_count += len(batch['x'])
            score = val_sum / val_count
            if not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError('Nonfinite validation objective')
            improved = score < best; best = min(best, score)
            history.append(dict(epoch=epoch, validation_objective=score, seconds=time.perf_counter()-began,
                                **{k: v/count for k, v in totals.items()}))
            state = dict(version=6, stage=stage, model=model.state_dict(), config=c,
                         data_id=train.store.meta['data_id'], prior_sha256=prior_sha,
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch, best=best,
                         history=history, rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if amp else [])
            save_torch(out / 'last.pt', state)
            if improved:
                save_torch(out / 'best.pt', state)
            write_json(out / 'history.json', history)
            print(history[-1], flush=True)
    finally:
        train.store.close(); validation.store.close()
    return dict(best_validation=best, epochs=len(history), stage=stage)
