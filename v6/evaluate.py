"""Held-out evaluation. True query futures enter scoring/oracle ONLY."""
import heapq
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr
from .common import fresh_dir, write_json, environment, read_json, sha256
from .fusion import FusionStats, check as check_fusion, apply as apply_fusion
from .data import Windows, scale_floor
from .inference import Retriever, analog
from .index import diverse


def probabilistic_scores(p, y, horizons):
    z = (np.asarray(y, dtype=float)-p['center'])/p['scale']
    pi, mu, sd = np.exp(p['logp']).astype(float), p['mu'].astype(float), p['sigma'].astype(float)
    def absolute_normal(delta, sigma):
        u = delta/sigma
        return 2*sigma*np.exp(-u*u/2)/np.sqrt(2*np.pi) + delta*(2*ndtr(u)-1)
    first = (pi[:, None]*absolute_normal(z[None]-mu, sd)).sum(0)
    second = np.zeros(len(z))
    for a in range(len(pi)):
        for b in range(len(pi)):
            second += pi[a]*pi[b]*absolute_normal(mu[a]-mu[b], np.sqrt(sd[a]**2+sd[b]**2))
    crps = first - second/2
    quantiles = []
    for probability in (0.05, 0.95):
        low, high = (mu-10*sd).min(0), (mu+10*sd).max(0)
        for _ in range(50):
            mid = (low+high)/2
            cdf = (pi[:, None]*ndtr((mid[None]-mu)/sd)).sum(0)
            low, high = np.where(cdf<probability, mid, low), np.where(cdf>=probability, mid, high)
        quantiles.append((low+high)/2)
    density = (pi[:, None]*np.exp(-0.5*((z[None]-mu)/sd)**2)/(np.sqrt(2*np.pi)*sd)).sum(0)
    # Marginal NLL, not the joint shared-regime likelihood used in training.
    nll = -np.log(np.maximum(density, 1e-300))
    records = [dict(horizon=h, crps=float(crps[:h].mean()), marginal_nll=float(nll[:h].mean()),
                    coverage90=float(((z[:h]>=quantiles[0][:h]) & (z[:h]<=quantiles[1][:h])).mean()),
                    width90=float((quantiles[1][:h]-quantiles[0][:h]).mean())) for h in horizons]
    return records, [q*p['scale']+p['center'] for q in quantiles]


def brute_audit(r, x, y, sid, start):
    """Streaming exhaustive raw/future oracle over exactly the indexed library.

    Future oracle is a hindsight upper bound and NEVER used by Retriever.
    """
    began = time.perf_counter(); c = r.c; m = c['length']; h = max(c['horizons'])
    qscale = max(float(np.std(x)), scale_floor(r.store.series[sid], c))
    qz = (x-x.mean())/qscale; yz = (y-x.mean())/qscale
    heaps = {'raw': [], 'future': []}; cap = c['evaluation']['k']*c['index']['oversample']
    group = r.store.series[sid]['group']; scope = c['evaluation']['scope']; count = 0
    for shard_id, shard in enumerate(r.library.meta['shards']):
        csid = shard['sid']
        if scope == 'series' and csid != sid or scope == 'group' and shard['group'] != group:
            continue
        starts = r.library.array(shard_id, 'starts.npy')
        floor = scale_floor(r.store.series[csid], c)
        for a in range(0, len(starts), 256):
            ss = starts[a:a+256]
            if csid == sid:
                ss = ss[np.abs(ss-start) >= m+h]
            if not len(ss):
                continue
            values = np.stack([r.store.window(csid, p, m+h) for p in ss]).astype(float)
            center = values[:, :m].mean(-1, keepdims=True)
            scale = np.maximum(values[:, :m].std(-1, keepdims=True), floor)
            past, future = (values[:, :m]-center)/scale, (values[:, m:]-center)/scale
            distances = dict(raw=np.mean((past-qz)**2, axis=1),
                             future=sum(np.mean((future[:, :hh]-yz[:hh])**2, axis=1) for hh in c['horizons'])/len(c['horizons']))
            count += len(ss)
            for method, ds in distances.items():
                heap = heaps[method]
                for p, d in zip(ss, ds):
                    item = (-float(d), -csid, -int(p))
                    if len(heap) < cap:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
    hits = {name: diverse([dict(distance=-d, sid=-s, start=-p) for d, s, p in sorted(heap, reverse=True)], c['evaluation']['k'], m)
            for name, heap in heaps.items()}
    return hits, dict(audit_seconds=time.perf_counter()-began, audited_windows=count)


def evaluate(store_path, checkpoint, index_path, output, split='test', device='cpu', leaf_budgets=None,
             time_budget_ms=None, oversample=None, fusion_path=None, channels=None, audit_queries=None):
    r = Retriever(store_path, checkpoint, index_path, device)
    if channels is not None:
        if not channels or set(channels)-set(r.c['index']['channels']): raise ValueError('Requested channel not built')
        r.c['index']['channels']=list(dict.fromkeys(channels))
    if audit_queries is not None:
        if audit_queries<0: raise ValueError('Negative audit query count')
        r.c['evaluation']['audit_queries']=audit_queries
    if leaf_budgets is not None:
        if not leaf_budgets or min(leaf_budgets) < 0: raise ValueError('Invalid leaf budgets')
        r.c['evaluation']['leaf_budgets'] = list(dict.fromkeys(leaf_budgets))
    if time_budget_ms is not None:
        if time_budget_ms < 0: raise ValueError('Invalid time budget')
        r.c['evaluation']['time_budget_ms'] = time_budget_ms
    if oversample is not None:
        if oversample < 1: raise ValueError('Invalid oversample')
        r.c['index']['oversample'] = oversample
        r.library.c['index']['oversample'] = oversample
    c = r.c; dataset = Windows(store_path, c, split); out = fresh_dir(output)
    fusion=read_json(fusion_path) if fusion_path else None
    index_sha=sha256(Path(index_path)/'manifest.json')
    if fusion:
        check_fusion(fusion,r.state['data_id'],r.library.meta['checkpoint_sha256'],index_sha,c,fusion['method'])
        if int(fusion['method'].split('leaves')[-1]) not in c['evaluation']['leaf_budgets']:
            raise ValueError('Fusion retrieval budget not requested')
    fusion_stats=FusionStats(c['horizons'])
    records, timing, calibration, audits, examples, encodings = [], [], [], [], [], []
    run = dict(version=6, split=split, complete=False, queries=len(dataset), config=c,
               data_id=r.state['data_id'], checkpoint_sha256=r.library.meta['checkpoint_sha256'],
               index_sha256=index_sha, fusion=fusion,
               checkpoint_epoch=r.state.get('epoch', -1), index_windows=r.library.meta['windows'],
               index_bytes=r.library.meta.get('payload_bytes'), environment=environment(),
               timing_note='Independent encode+search+candidate fetch per method; process/model startup excluded. First query marked.',
               causal_scope='per-series chronology; cross-series calendar alignment is NOT verified')
    write_json(out/'run.json', run)
    rng = np.random.default_rng(c['seed'])
    audit_ids = set(rng.choice(len(dataset), min(len(dataset), c['evaluation']['audit_queries']), replace=False).tolist())
    try:
        for qi in range(len(dataset)):
            batch = dataset[qi]; x, y, sid, start = batch['x'], batch['y'], int(batch['sid']), int(batch['start'])
            s = r.store.series[sid]; floor = scale_floor(s, c)
            scale = max(float(x.std()), floor); memory_scale = max(s['memory_std'], 1e-6)
            vectors, p, direct, _ = r.encode(x, sid)
            if 'learned' in vectors: encodings.append(vectors['learned'])
            cal, interval = probabilistic_scores(p, y, c['horizons'])
            for row in cal:
                row['regime_entropy'] = float(-(np.exp(p['logp'])*p['logp']).sum())
            calibration.extend(dict(query=qi, sid=sid, **row) for row in cal)
            forecasts = {'persistence': np.full(len(y), x[-1]),
                         'probabilistic_prior': p['mean']*p['scale']+p['center'], 'encoder_forecast': direct}
            retrieval_details = {}
            for channel in c['index']['channels']:
                for budget in c['evaluation']['leaf_budgets']:
                    name = f'{channel}_leaves{budget}'
                    # retrieve() never receives y or the dataset object.
                    result = r.retrieve(x, sid, start, channel, budget)
                    forecasts[name] = result['prediction']
                    timing.append(dict(query=qi, sid=sid, method=name, first_query=qi==0, **result['stats']))
                    mapped = result['mapped']
                    future_errors = np.mean(((mapped-y)/scale)**2, axis=1) if len(mapped) else np.array([])
                    distances = [hit['distance'] for hit in result['hits']]
                    history_errors=[]
                    for hit in result['hits']:
                        past=r.store.window(hit['sid'],hit['start'],c['length']).astype(float)
                        z=(past-past.mean())/max(float(past.std()),scale_floor(r.store.series[hit['sid']],c))
                        history_errors.append(float(np.mean((z-(x.astype(float)-float(x.mean()))/scale)**2)))
                    corr = None
                    if len(mapped)>2 and np.std(distances)>1e-12 and np.std(future_errors)>1e-12:
                        corr = float(spearmanr(distances, future_errors).statistic)
                    retrieval_details[name] = dict(hits=result['hits'],
                        exact_diverse_topk=result['stats']['exact_diverse_topk'],
                        best_future_nmse=float(future_errors.min()) if len(mapped) else None,
                        mean_future_nmse=float(future_errors.mean()) if len(mapped) else None,
                        mean_history_nmse=float(np.mean(history_errors)) if history_errors else None,
                        max_history_nmse=float(np.max(history_errors)) if history_errors else None,
                        future_nmse_by_horizon={str(h):float(np.mean(((mapped[:,:h]-y[:h])/scale)**2)) for h in c['horizons']} if len(mapped) else {},
                        distance_future_spearman=corr)
            for channel in c['index']['channels']:
                reference = retrieval_details.get(f'{channel}_leaves0')
                if reference and reference['exact_diverse_topk']:
                    ids = {(hit['sid'],hit['start']) for hit in reference['hits']}
                    for budget in c['evaluation']['leaf_budgets']:
                        detail = retrieval_details[f'{channel}_leaves{budget}']
                        detail['embedding_recall_against_full'] = len(ids & {(hit['sid'],hit['start']) for hit in detail['hits']})/len(ids)
            fusion_stats.add(forecasts,y,max(s['memory_std'],1e-6))
            if fusion:
                fusion_began=time.perf_counter()
                forecasts['validation_fusion']=apply_fusion(fusion,forecasts)
                fusion_ms=(time.perf_counter()-fusion_began)*1000
                source_timing=next(row for row in reversed(timing) if row['query']==qi and row['method']==fusion['method'])
                timing.append(dict(source_timing,method='validation_fusion',total_ms=source_timing['total_ms']+fusion_ms,
                                   fusion_ms=fusion_ms,latency_source='retrieval query plus measured fusion arithmetic'))
            if qi in audit_ids:
                hit_sets, audit_stats = brute_audit(r, x, y, sid, start)
                for method, hits in hit_sets.items():
                    prediction, _ = analog(r.store, c, x, floor, hits)
                    forecasts[f'audit_{method}'] = prediction
                audit = dict(query=qi, sid=sid, **audit_stats, raw_hits=hit_sets['raw'], future_oracle_hits=hit_sets['future'])
                oracle_ids = {(v['sid'], v['start']) for v in hit_sets['future']}
                audit['future_oracle_id_recall'] = {name: len(oracle_ids & {(v['sid'], v['start']) for v in detail['hits']})/max(len(oracle_ids), 1)
                                                    for name, detail in retrieval_details.items()}
                audits.append(audit)
            for method, prediction in forecasts.items():
                for h in c['horizons']:
                    error = np.asarray(prediction[:h], dtype=float)-y[:h]
                    mse = float(np.mean(error**2)); base = float(np.mean((x[-1]-y[:h].astype(float))**2))
                    records.append(dict(query=qi, sid=sid, group=s['group'], method=method, horizon=h,
                        mse=mse, mae=float(np.abs(error).mean()), nmse=mse/memory_scale**2,
                        history_nmse=mse/scale**2, persistence_nmse=base/memory_scale**2,
                        audit_query=qi in audit_ids))
            if len(examples) < c['evaluation']['max_examples']:
                detail = next((v for name,v in retrieval_details.items() if name.startswith('joint_leaves')),
                              retrieval_details.get('learned_leaves0', next(iter(retrieval_details.values()))))
                matches = []
                for hit in detail['hits'][:3]:
                    values = r.store.window(hit['sid'],hit['start'],len(x)+len(y)).astype(float)
                    past = values[:len(x)]
                    mapped = float(x.mean())+scale*(values-past.mean())/max(float(past.std()),scale_floor(r.store.series[hit['sid']],c))
                    matches.append(dict(**hit, mapped=mapped.tolist()))
                examples.append(dict(query=qi, sid=sid, start=start, x=x.tolist(), y=y.tolist(),
                    forecasts={k: np.asarray(v).tolist() for k, v in forecasts.items()},
                    lower90=interval[0].tolist(), upper90=interval[1].tolist(), retrieval=retrieval_details, matches=matches))
            with (out/'retrieval.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(query=qi, sid=sid, details=retrieval_details), allow_nan=False)+'\n')
            print(dict(query=qi+1, total=len(dataset)), flush=True)
        for name, rows in [('metrics', records), ('timing', timing), ('calibration', calibration), ('audits', audits)]:
            with (out/f'{name}.jsonl').open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, allow_nan=False)+'\n')
        write_json(out/'examples_private.json', examples)
        if len(encodings) > 1:
            v = np.asarray(encodings, dtype=float)
            eigen = np.maximum(np.linalg.eigvalsh(np.atleast_2d(np.cov(v, rowvar=False))), 0)
            mass = eigen/max(float(eigen.sum()), 1e-30)
            run['embedding_diagnostics'] = dict(mean_dimension_std=float(v.std(0).mean()),
                effective_rank=float(np.exp(-(mass*np.log(np.maximum(mass, 1e-30))).sum())) if eigen.sum()>1e-30 else 0.)
        run['complete'] = True; write_json(out/'run.json', run)
        write_json(out/'fusion_stats.json',fusion_stats.export(run))
    finally:
        dataset.store.close(); r.close()
    return run
