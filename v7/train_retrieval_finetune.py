"""Paired lightweight Moirai fine-tuning with and without retrieved memory."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from v6.common import save_torch
from .common import fresh_dir, write_json, sha256
from .moirai import MoiraiAdapter
from .retrieval_finetune import RetrievalFineTune, quantile_loss


class CachedMemory(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.data = {key:np.array(data[key], copy=True) for key in data.files}

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, i):
        return {key:value[i] for key,value in self.data.items()}


def forecast(model, batch, use_retrieval):
    return model(batch['x'], batch['examples'], batch['weights'], batch['valid']) if use_retrieval else model(batch['x'])


def validation_score(model, data, c, stds, batch_size, device, use_retrieval):
    values = []
    median = list(model.module.quantile_levels).index(.5)
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            batch = {key:value.to(device) for key,value in batch.items()}
            pred = forecast(model, batch, use_retrieval)[:, median]
            error = (pred-batch['y']).double().square()
            variance = stds[batch['sid']].square()
            values.extend((sum(error[:, :h].mean(-1)/variance for h in c['horizons'])/len(c['horizons'])).cpu().tolist())
    score = float(np.mean(values))
    if not np.isfinite(score):
        raise FloatingPointError('Nonfinite validation objective')
    return score


def train(root, source, name, variant, device='cuda', epochs=8, batch_size=16):
    dataset_root = root/name
    memory = dataset_root/'memory'
    meta = json.loads((memory/'manifest.json').read_text())
    if not meta['complete'] or meta['history_weight'] != .2 or meta['future_weight'] != .8:
        raise ValueError('Expected complete fixed-weight retrieval memory')
    for split in ['train', 'validation', 'test']:
        if sha256(memory/(split+'.npz')) != meta['splits'][split]['sha256']:
            raise ValueError('Memory artifact changed')
    c = meta['config']
    use_retrieval = variant == 'rag'
    out = fresh_dir(dataset_root/variant)
    torch.manual_seed(c['seed'])
    adapter = MoiraiAdapter.pretrained(c['moirai'], device)
    model = RetrievalFineTune(adapter, c['length'], max(c['horizons']), use_retrieval=use_retrieval).to(device)
    train_data, val_data = CachedMemory(memory/'train.npz'), CachedMemory(memory/'validation.npz')
    catalog = json.loads((source/name/'store/catalog.json').read_text())
    assert catalog['data_id']==meta['data_id']
    stds = torch.tensor([max(s['memory_std'], 1e-6) for s in catalog['series']], dtype=torch.float64, device=device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    run = dict(complete=False, dataset=name, variant=variant, config=c, device=device,
        torch=str(torch.__version__), epochs_limit=epochs, batch_size=batch_size, learning_rate=1e-4,
        lora_rank=8, lora_layers='last two encoder layers, query/value projections',
        trainable_parameters=sum(p.numel() for p in parameters), pretrained_identity=model.identity,
        memory_sha256=sha256(memory/'manifest.json'), data_id=meta['data_id'],
        selection='Validation mean NMSE over horizons; epoch 0 is an eligible unchanged pretrained baseline.',
        train_windows=len(train_data), validation_windows=len(val_data))
    write_json(out/'run.json', run)
    history, best_epoch = [], 0
    began = time.perf_counter()
    best = validation_score(model, val_data, c, stds, batch_size, device, use_retrieval)
    history.append(dict(epoch=0, validation_nmse=best, train_pinball=None, seconds=time.perf_counter()-began))

    def save_best(epoch, score):
        save_torch(out/'best.pt', dict(version=7, variant='moirai_retrieval_finetune',
            adaptation=model.adaptation_state(), config=c, use_retrieval=use_retrieval,
            architecture=model.architecture, identity=model.identity, rank=model.rank,
            epoch=epoch, validation_nmse=score, memory_sha256=run['memory_sha256']))

    save_best(0, best)
    print(name, variant, history[-1], flush=True)
    for epoch in range(1, epochs+1):
        epoch_start = time.perf_counter()
        model.train()
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(c['seed']+epoch))
        total, count = 0., 0
        for batch in loader:
            batch = {key:value.to(device) for key,value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # Upstream recursive inference writes trajectory tensors in place.
            # Autograd clones saved tensors on mutation while preserving its forward arithmetic.
            with torch.autograd.graph.allow_mutation_on_saved_tensors():
                pred = forecast(model, batch, use_retrieval)
                loss = quantile_loss(pred, batch['y'], batch['x'], batch['floor'], model.module.quantile_levels, c['horizons'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite fine-tuning loss')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(batch['x'])
            count += len(batch['x'])
        score = validation_score(model, val_data, c, stds, batch_size, device, use_retrieval)
        if score < best:
            best, best_epoch = score, epoch
            save_best(epoch, score)
        history.append(dict(epoch=epoch, validation_nmse=score, train_pinball=total/count,
            seconds=time.perf_counter()-epoch_start))
        write_json(out/'history.json', history)
        print(name, variant, history[-1], flush=True)
    state = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    model.load_adaptation(state['adaptation'])
    model.eval()
    test_data = CachedMemory(memory/'test.npz')
    median = list(model.module.quantile_levels).index(.5)
    metrics, forecasts, qi = [], [], 0
    with torch.no_grad():
        for batch in DataLoader(test_data, batch_size=batch_size, shuffle=False):
            batch = {key:value.to(device) for key,value in batch.items()}
            qpred = forecast(model, batch, use_retrieval)
            pred = qpred[:, median]
            # Same fine-tuned model, memory disabled: diagnostic, never a selection criterion.
            without = forecast(model, batch, False)[:, median] if use_retrieval else pred
            for i in range(len(pred)):
                y = batch['y'][i].cpu().numpy()
                candidates = {'finetuned':pred[i].cpu().numpy(), 'frozen':batch['frozen_prediction'][i].cpu().numpy(),
                    'analog':batch['analog_prediction'][i].cpu().numpy()}
                if use_retrieval:
                    candidates['memory_disabled'] = without[i].cpu().numpy()
                sid, start = int(batch['sid'][i]), int(batch['start'][i])
                for method, prediction in candidates.items():
                    for h in c['horizons']:
                        error = np.asarray(prediction[:h], dtype=float)-y[:h]
                        mse = float(np.mean(error**2))
                        metrics.append(dict(query=qi, sid=sid, start=start, horizon=h, method=method,
                            mse=mse, mae=float(np.abs(error).mean()), nmse=mse/max(catalog['series'][sid]['memory_std'], 1e-6)**2))
                forecasts.append(dict(query=qi, sid=sid, start=start, prediction=candidates['finetuned'].tolist(),
                    memory_disabled=candidates.get('memory_disabled', candidates['finetuned']).tolist()))
                qi += 1
    with (out/'metrics.jsonl').open('w', encoding='utf-8') as f:
        for record in metrics:
            f.write(json.dumps(record, allow_nan=False)+'\n')
    write_json(out/'predictions.json', forecasts)
    run.update(complete=True, best_epoch=best_epoch, best_validation=best,
        total_seconds=time.perf_counter()-began, checkpoint_sha256=sha256(out/'best.pt'),
        cuda_peak_memory_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
    write_json(out/'run.json', run)
    print('COMPLETE', name, variant, 'best epoch', best_epoch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='v7/runs/moirai_rag_finetune_20261008')
    parser.add_argument('--source', default='v7/runs/all_datasets_20261008')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--variant', choices=['plain','rag'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    train(Path(args.root), Path(args.source), args.dataset, args.variant, args.device, args.epochs, args.batch_size)
