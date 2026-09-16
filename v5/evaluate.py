"""Held-out compression, full-library recall, predictive utility, forecast tests."""
import math
import time
import json
from pathlib import Path
import numpy as np
import torch
from .common import fresh_dir, write_json, environment, affine, norm_scale, weighted_analog
from .data import Store
from .teacher import Labels
from .train import load_checkpoint
from .token_index import TokenLibrary, file_hash


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def exact_distances(query, past):
    def z(x):
        y = x-x[..., :1]; y -= y.mean(axis=-1, keepdims=True)
        sd = np.sqrt(np.mean(y*y, axis=-1, keepdims=True))
        return np.where(sd > 1e-10, y/np.maximum(sd, 1e-10), 0.)
    return np.sum((z(np.asarray(past, dtype=float))-z(np.asarray(query, dtype=float)))**2, axis=-1)


def evaluate(store_path, labels_path, checkpoint, index_path, output, split='test', device='cpu'):
    if split not in ('validation','test'): raise ValueError('Only held-out splits may be evaluated')
    out = fresh_dir(output); store = Store(store_path); labels = Labels(labels_path, split)
    model, state = load_checkpoint(checkpoint, device); library = TokenLibrary(index_path)
    if not len(labels): raise ValueError('Requested evaluation split is empty')
    identity = store.meta['data_id']
    if state['data_id'] != identity or labels.meta['data_id'] != identity or library.meta['data_id'] != identity:
        raise ValueError('Mismatched dataset artifacts')
    if library.meta['checkpoint_sha256'] != file_hash(checkpoint): raise ValueError('Token index was built with another checkpoint')
    c = state['config']; m = c['length']; kh = c['evaluation']['analog_k']; rng = np.random.default_rng(c['seed']+9001)
    timings, skipped, examples = [], [], []
    reconstruction_nmse, reconstruction_count = 0., 0
    forecasts = (out/'forecasts.jsonl').open('w', encoding='utf-8')
    retrieval = (out/'retrieval.jsonl').open('w', encoding='utf-8')
    def emit_metrics(qid, sid, h, method, prediction, truth, scale, candidates=0, mean_future=None):
        error = prediction-truth
        row = dict(query_id=qid, sid=sid, group=store.series[sid]['group'], horizon=h, method=method,
                   mse=float(np.mean(error**2)), mae=float(np.mean(abs(error))), nmse=float(np.mean(error**2)/(scale*scale)),
                   returned=candidates, mean_candidate_future_nmse=mean_future)
        forecasts.write(json.dumps(row, allow_nan=False)+'\n')
    try:
        with torch.no_grad():
            for qi in range(len(labels)):
                label = labels[qi]; sid, start = int(label['sid']), int(label['start']); s = store.series[sid]
                if sid not in library.series:
                    skipped.append(dict(query=qi, sid=sid, reason='series absent from token library')); continue
                library.select(sid); qid = f'{sid}:{start}'
                q = store.window(sid, start, m); target = store.window(sid, start+m, max(c['horizons']))
                scale = norm_scale(q, s['memory_std']); floor = max(s['memory_std']*1e-4, 1e-6)
                sync(device); began = time.perf_counter()
                result = model(torch.from_numpy(q[None]).to(device), torch.tensor([floor], device=device))
                shape = result['shape'][0].cpu().numpy(); predictive = result['predictive'][0].cpu().numpy()
                codes = result['ids'][0].cpu().numpy(); direct = result['prediction'][0].cpu().numpy()
                sync(device); encode_ms = (time.perf_counter()-began)*1000
                reconstruction_nmse += float(np.mean((result['reconstruction'][0].cpu().numpy()-q)**2)/(scale*scale))
                reconstruction_count += 1
                valid = label['valid']; teacher_pos = label['candidates'][valid]; teacher_dp = label['past_distance'][valid]
                top = teacher_pos[:min(10, len(teacher_pos))]; oracle_set = set(map(int, top))
                indices = np.searchsorted(library.starts, top)
                coverage = np.mean((indices < library.n) & (library.starts[np.minimum(indices,library.n-1)] == top))
                at_boundary = len(teacher_dp) > 10 and abs(float(teacher_dp[9]-teacher_dp[10])) < 1e-6
                budgets = sorted(set(min(library.n, max(1, int(v))) for v in c['evaluation']['topk']) |
                                 {min(library.n, max(1, math.ceil(library.n*c['evaluation']['candidate_fraction'])))})
                main_k = min(library.n, max(1, math.ceil(library.n*c['evaluation']['candidate_fraction'])))
                choices = dict(raw_shape=teacher_pos[:kh], random=library.starts[rng.choice(library.n, min(kh,library.n), replace=False)])
                for h in c['horizons']:
                    truth = target[:h]
                    emit_metrics(qid, sid, h, 'persistence', np.full(h, q[-1]), truth, scale)
                    emit_metrics(qid, sid, h, 'token_forecaster', direct[:h], truth, scale)
                for name, positions in choices.items():
                    past = np.array([store.window(sid, p, m) for p in positions]); ds = exact_distances(q, past)
                    for h in c['horizons']:
                        future = np.array([store.window(sid, p+m, h) for p in positions])
                        prediction = weighted_analog(q, past, future, ds)
                        value = float(np.mean((affine(q, past, future)-target[:h])**2)/(scale*scale))
                        emit_metrics(qid, sid, h, name, prediction, target[:h], scale, len(positions), value)
                for route in ('cosine', 'bm25'):
                    began = time.perf_counter()
                    ids = library.cosine(shape, max(budgets)) if route == 'cosine' else library.bm25(codes, max(budgets))
                    search_ms = (time.perf_counter()-began)*1000
                    positions = np.array(library.starts[ids])
                    for budget in budgets:
                        found = set(map(int, positions[:budget]))
                        retrieval.write(json.dumps(dict(query_id=qid, sid=sid, group=s['group'], route=route, k=budget,
                            returned=min(budget, len(ids)), library_windows=library.n, candidate_fraction=min(budget,len(ids))/library.n,
                            oracle_recall=len(found & oracle_set)/len(oracle_set), oracle_coverage=float(coverage),
                            teacher_tie_at10=bool(at_boundary), encode_ms=encode_ms, search_ms=search_ms,
                            timing_note='Search generated maximum configured budget; smaller K reuse its prefix'))+'\n')
                    chosen_ids = ids[:main_k]; chosen_pos = positions[:main_k]
                    began = time.perf_counter()
                    if len(chosen_pos):
                        past = np.array([store.window(sid,p,m) for p in chosen_pos]); ds = exact_distances(q, past)
                        raw_order = np.argsort(ds, kind='stable')[:kh]
                        pred_order = np.argsort(-(library.predictive[chosen_ids] @ predictive), kind='stable')[:kh]
                    else:
                        past = np.empty((0,m)); ds = np.empty(0); raw_order = pred_order = np.empty(0,dtype=int)
                    verify_ms = (time.perf_counter()-began)*1000
                    for ranker, order in (('exact', raw_order), ('predictive', pred_order)):
                        began = time.perf_counter()
                        for h in c['horizons']:
                            truth = target[:h]
                            if len(order):
                                future = np.array([store.window(sid,int(chosen_pos[j])+m,h) for j in order])
                                prediction = weighted_analog(q,past[order],future,ds[order])
                                future_error = float(np.mean((affine(q,past[order],future)-truth)**2)/(scale*scale))
                            else: prediction = np.full(h,q[-1]); future_error = None
                            emit_metrics(qid,sid,h,f'{route}_{ranker}',prediction,truth,scale,len(order),future_error)
                        future_ms = (time.perf_counter()-began)*1000
                        timings.append(dict(query_id=qid,route=route,ranker=ranker,encode_ms=encode_ms,search_ms=search_ms,
                            verify_ms=verify_ms,future_ms=future_ms,total_ms=encode_ms+search_ms+verify_ms+future_ms,
                            note='future_ms includes offline scoring for all horizons; conservative pipeline estimate'))
                if len(examples) < c['evaluation']['max_examples']:
                    examples.append(dict(query_id=qid,group=s['group'],query=q.tolist(),truth=target.tolist(),prediction=direct.tolist()))
                if qi % 50 == 0:
                    forecasts.flush(); retrieval.flush(); print(f'evaluation {qi+1}/{len(labels)}',flush=True)
        if not reconstruction_count: raise ValueError('No queries had an eligible token library')
        write_json(out/'timing.json', timings)
        write_json(out/'examples_private.json', examples)
        write_json(out/'run.json', dict(complete=True, config=c, environment=environment(), split=split,
                   data_id=identity, checkpoint_sha256=file_hash(checkpoint), queries=len(labels)-len(skipped), skipped=skipped,
                   compression=library.meta['compression'], vocabulary=library.meta['config']['model'],
                   scope='same-series, sampled library vs stride-1 V4 teacher; no ANN or 100ms/10B guarantee',
                   reconstruction_nmse=reconstruction_nmse/reconstruction_count,
                   training_completed=state.get('epoch',-1) >= 0))
    finally:
        forecasts.close(); retrieval.close(); library.close(); store.close()
