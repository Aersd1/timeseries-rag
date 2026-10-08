"""Train A's V6-compatible projection; keep pretrained Moirai fixed."""
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from v6.data import Windows
from v6.model import encoder_loss
from v6.sampling import HistoryBatches
from v6.common import save_torch, environment
from .common import config, fresh_dir, write_json
from .moirai import MoiraiAdapter
from .model import MoiraiEncoder


def fit(store, config_path, output, device='cpu'):
    c = config(config_path)
    torch.manual_seed(c['seed'])
    train, val = Windows(store, c, 'train'), Windows(store, c, 'validation')
    out = fresh_dir(output)
    adapter = MoiraiAdapter.pretrained(c['moirai'], device)
    model = MoiraiEncoder(c, adapter).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=c['training']['lr'], weight_decay=c['training']['weight_decay'])
    sampler = HistoryBatches(train, c['training']['batch_size'], c['seed'], c['training']['history_neighbor_fraction']) if c['training']['history_neighbor_batches'] else None
    history, best = [], float('inf')
    write_json(out/'config.json', c)
    write_json(out/'environment.json', dict(**environment(), torch=str(torch.__version__),
        device=str(device), train_windows=len(train), validation_windows=len(val), precision='float32'))
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
                loss, _ = encoder_loss(model, batch)
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
                    loss, _ = encoder_loss(model, batch)
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
