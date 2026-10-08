"""Fit the existing standalone PatchTST and score the saved ETT query protocol."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from v6.patchtst_direct import fit, load_forecast, DenseWindows
from v6.data import Windows
from v6.common import config, sha256, write_json, fresh_dir, environment


def run_dataset(source, root, name, device, epochs, patience):
    src, out = source/name, fresh_dir(root/name)
    c = config(src/'config.json')
    write_json(out/'config.json', c)
    sizes = {}
    for split in ['train', 'validation']:
        ds = DenseWindows(src/'store', c, split, c['training']['sample_stride'],
                          cap_per_series=512 if split == 'validation' else None)
        sizes[split+'_windows'] = len(ds)
        ds.store.close()
    began = time.perf_counter()
    trained = fit(src/'store', c, out/'train', device, epochs, patience, val_cap=512)
    training_seconds = time.perf_counter()-began
    model, state = load_forecast(out/'train/best.pt', device)
    write_json(out/'training.json', dict(**trained, **sizes, best_epoch_zero_based=state['epoch'],
        training_seconds=training_seconds, parameters=sum(p.numel() for p in model.parameters()),
        device=device, torch=str(torch.__version__), environment=environment(),
        train_stride=c['training']['sample_stride'], validation_cap_per_series=512,
        model_source_sha256=sha256('v6/patchtst_direct.py')))
    ds = Windows(src/'store', c, 'test')
    assert ds.store.meta['data_id'] == state['data_id']
    original_run = json.loads((src/'test/run.json').read_text())
    assert original_run['complete'] and original_run['data_id'] == state['data_id'] and original_run['config'] == c
    rows, predictions = [], []
    query = 0
    try:
        with torch.inference_mode():
            for batch in DataLoader(ds, batch_size=c['training']['batch_size'], shuffle=False):
                forecast = model(batch['x'].to(device), batch['floor'].to(device))['raw'].cpu().numpy()
                for i, pred in enumerate(forecast):
                    x, y = batch['x'][i].numpy(), batch['y'][i].numpy()
                    sid, start = int(batch['sid'][i]), int(batch['start'][i])
                    for method, result in [('patchtst_direct', pred), ('persistence', np.full(len(y), x[-1]))]:
                        for h in c['horizons']:
                            error = np.asarray(result[:h], dtype=float)-y[:h]
                            mse = float(np.mean(error**2))
                            rows.append(dict(query=query, sid=sid, start=start, horizon=h, method=method,
                                mse=mse, mae=float(np.abs(error).mean()),
                                nmse=mse/max(ds.store.series[sid]['memory_std'], 1e-6)**2))
                    predictions.append(dict(query=query, sid=sid, start=start, history=x.tolist(), future=y.tolist(),
                        prediction=pred.tolist(), column=ds.store.series[sid]['column']))
                    query += 1
        assert query == len(ds)
        frame = pd.DataFrame(rows)
        assert np.isfinite(frame[['mse', 'mae', 'nmse']]).all().all()
        previous = pd.read_json(src/'test/metrics.jsonl', lines=True)
        keys = ['query', 'sid', 'start', 'horizon']
        matched = frame[frame.method=='persistence'].merge(previous[previous.method=='persistence'], on=keys, validate='one_to_one')
        assert len(matched) == query*len(c['horizons'])
        np.testing.assert_allclose(matched.nmse_x, matched.nmse_y, atol=1e-6, rtol=1e-6)
        out_test = fresh_dir(out/'test')
        with (out_test/'metrics.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        write_json(out_test/'predictions.json', predictions)
        write_json(out_test/'run.json', dict(complete=True, split='test', queries=query,
            config=c, data_id=state['data_id'], checkpoint_sha256=sha256(out/'train/best.pt'),
            best_epoch_zero_based=state['epoch'], training_seconds=training_seconds))
        result = dict(dataset=name, **frame.groupby('method').nmse.mean().to_dict(),
            train_windows=sizes['train_windows'], validation_windows=sizes['validation_windows'],
            epochs=trained['epochs'], best_epoch=state['epoch']+1, training_seconds=training_seconds)
        print('COMPLETED', json.dumps(result), flush=True)
        return result
    finally:
        ds.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='v7/runs/all_datasets_20261008')
    parser.add_argument('--output', default='v7/runs/patchtst_direct_20261008')
    parser.add_argument('--datasets', nargs='+', default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    args = parser.parse_args()
    root = fresh_dir(args.output)
    write_json(root/'protocol.json', dict(source=args.source, datasets=args.datasets,
        max_epochs=args.epochs, patience=args.patience, device=args.device,
        architecture='Existing v6.patchtst_direct.PatchTSTForecast; independent per-series histories with shared weights within each CSV; flatten token forecast head.',
        selection='Multi-horizon validation MSE in query-normalized space. Test queries identical to saved Moirai benchmark.',
        budget_note='All training stride windows, up to 80 epochs. A used at most 64 windows per variable for 8 epochs plus pretrained Moirai; budgets are not matched.',
        code_sha256={p:sha256(p) for p in ['v6/patchtst_direct.py', 'v7/benchmark_patchtst_direct.py']}))
    results = [run_dataset(Path(args.source), root, name, args.device, args.epochs, args.patience) for name in args.datasets]
    write_json(root/'summary.json', dict(complete=True, datasets=results))


if __name__ == '__main__':
    main()
