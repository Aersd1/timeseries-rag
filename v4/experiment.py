"""Sections I-VI: analog futures, three transfers, KNN, past/future statistics."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, ConstantInputWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast import ForecastIndex, transfer, aggregate, series_key
from prepare import load_series
from model import znorm

METHODS = ('mean_std', 'endpoint', 'affine')


def rho(x, y):
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConstantInputWarning)
        r = float(spearmanr(x, y).statistic)
    return r if np.isfinite(r) else None


def cluster_ci(rows, field, rng):
    # Whole logical series, not candidate pairs, are bootstrap units.
    groups = {}
    for r in rows:
        if r[field] is not None:
            groups.setdefault(r['series_id'], []).append(r[field])
    if len(groups) < 2:
        return None
    arrays = list(groups.values())
    boot = [np.mean(np.concatenate([arrays[j] for j in rng.integers(len(arrays), size=len(arrays))]))
            for _ in range(1000)]
    return np.percentile(boot, [2.5, 97.5]).tolist()


def summarize(out, rows, forecasts, timing, examples, config):
    rng = np.random.default_rng(2009)
    groups = sorted({r['group'] for r in rows})
    summaries = []
    for scope in ('same_series', 'pooled_train'):
        for group in groups+['ALL']:
            for h in config['horizons']:
                for selection in ('ordinary', 'episodes'):
                    for method in METHODS:
                        rr = [r for r in rows if r['scope'] == scope and (group == 'ALL' or r['group'] == group)
                              and r['horizon'] == h and r['selection'] == selection and r['method'] == method]
                        if not rr:
                            continue
                        correlations = [r['rho'] for r in rr if r['rho'] is not None]
                        summaries.append(dict(scope=scope, group=group, horizon=h, selection=selection, method=method,
                            queries=len(rr), pairs=sum(r['count'] for r in rr), rho_queries=len(correlations),
                            mean_query_spearman=float(np.mean(correlations)) if correlations else None,
                            rho_cluster_ci95=cluster_ci(rr, 'rho', rng),
                            mean_top10_nmse=float(np.mean([r['top10_nmse'] for r in rr])),
                            mean_last10_nmse=float(np.mean([r['last10_nmse'] for r in rr])),
                            mean_random_nmse=float(np.mean([r['random_nmse'] for r in rr])),
                            mean_top10_minus_random=float(np.mean([r['top10_minus_random'] for r in rr])),
                            top10_random_cluster_ci95=cluster_ci(rr, 'top10_minus_random', rng),
                            median_count=float(np.median([r['count'] for r in rr]))))
    fs = []
    keys = sorted({(r['scope'], r['horizon'], r['selection'], r['method'], r['aggregation'], r['k']) for r in forecasts})
    for scope, h, selection, method, agg, k in keys:
        rr = [r for r in forecasts if (r['scope'], r['horizon'], r['selection'], r['method'], r['aggregation'], r['k']) == (scope, h, selection, method, agg, k)]
        nmse = np.mean([r['nmse'] for r in rr]); base = np.mean([r['persistence_nmse'] for r in rr])
        fs.append(dict(scope=scope, horizon=h, selection=selection, method=method, aggregation=agg, k=k,
                       queries=len(rr), mean_nmse=float(nmse), paired_persistence_nmse=float(base),
                       skill_vs_persistence=float(1-nmse/base) if base > 0 else None,
                       win_rate=float(np.mean([r['nmse'] < r['persistence_nmse'] for r in rr]))))
    latency = []
    for scope in ('same_series', 'pooled_train'):
        rr = [r for r in timing if r['scope'] == scope]
        latency.append(dict(scope=scope, queries=len(rr),
            retrieval_p50_ms=float(np.median([r['retrieval_ms'] for r in rr])),
            retrieval_p95_ms=float(np.percentile([r['retrieval_ms'] for r in rr], 95)),
            analysis_p50_ms=float(np.median([r['analysis_ms'] for r in rr])),
            incomplete_episode_queries=sum(not r['episode_complete'] for r in rr),
            self_or_overlap_violations=sum(r['violations'] for r in rr)))
    report = dict(config=config, statistics=summaries, forecasts=fs, latency=latency)
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(2, len(config['horizons']), figsize=(15, 8), squeeze=False)
    for i, h in enumerate(config['horizons']):
        for j, scope in enumerate(('same_series', 'pooled_train')):
            ax = axes[j, i]
            for method in METHODS:
                ss = [r for r in summaries if r['group'] != 'ALL' and r['scope'] == scope and r['horizon'] == h
                      and r['selection'] == 'ordinary' and r['method'] == method and r['mean_query_spearman'] is not None]
                ax.plot([r['group'] for r in ss], [r['mean_query_spearman'] for r in ss], 'o-', label=method)
            ax.axhline(0, color='grey', lw=1); ax.set_ylim(-1, 1)
            ax.set_title(f'{scope}; H={h}'); ax.tick_params(axis='x', rotation=40)
            ax.set_ylabel('Mean within-query Spearman'); ax.legend(fontsize=8)
    fig.suptitle('Past distance vs future MSE: self excluded; positive rho supports association')
    fig.tight_layout(); fig.savefig(out/'correlations.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, len(config['horizons']), figsize=(15, 4.5), squeeze=False)
    for i, h in enumerate(config['horizons']):
        ax = axes[0, i]
        for method in METHODS:
            rr = [r for r in fs if r['scope'] == 'same_series' and r['horizon'] == h and r['selection'] == 'ordinary'
                  and r['method'] == method and r['aggregation'] == 'mean']
            ax.plot([r['k'] for r in rr], [r['skill_vs_persistence'] for r in rr], 'o-', label=method)
        ax.axhline(0, color='grey'); ax.set_title(f'H={h}'); ax.set_xlabel('K'); ax.set_ylabel('Skill vs persistence (higher better)'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'forecast_skill.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3*len(examples)), squeeze=False)
    for ax, example in zip(axes[:, 0], examples):
        q, y, predictions, group, h = example
        ax.plot(np.arange(-len(q), 0), q, color='black', label='query past')
        ax.plot(y, color='black', lw=2, label='true query future')
        for name, pred in predictions.items(): ax.plot(pred, label=name, alpha=.8)
        ax.axvline(-.5, color='grey', ls='--'); ax.set_title(f'{group}, H={h}; first sampled same-series query'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'examples.png', dpi=150); plt.close(fig)
    # Conditional expectation uses within-query distance quintiles and equal
    # query weights. Raw pair distances/errors are retained for alternate bins.
    fig, axes = plt.subplots(2, len(config['horizons']), figsize=(15, 8), squeeze=False)
    for i, h in enumerate(config['horizons']):
        for j, scope in enumerate(('same_series', 'pooled_train')):
            ax = axes[j, i]
            for method in METHODS:
                rr = [r for r in rows if r['scope'] == scope and r['horizon'] == h and r['selection'] == 'ordinary' and r['method'] == method]
                yy = np.mean([r['distance_bin_nmse'] for r in rr], axis=0)
                ax.plot(np.arange(1, 6), yy, 'o-', label=method)
            ax.set_title(f'{scope}; H={h}'); ax.set_xlabel('Within-query past-distance quintile (near -> far)')
            ax.set_ylabel('Mean future MSE / train variance'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'distance_future.png', dpi=160); plt.close(fig)
    lines = ['# V4 实验报告', '', '查询自身在候选资格筛选阶段排除。索引和频率/分箱仅用各序列前 60%；候选历史及未来全部位于训练段。',
             '', f"数据：{config['original_unique_points']:,} 个去重原始点，{config['train_points']:,} 个训练点；不是全量 GPU/10B 验证。",
             '', 'same_series 是同文件、列、设备的严格过去检索。pooled_train 是各序列训练前缀组成的离线外部案例库；未对齐跨文件绝对时间，不能解释为全局在线历史实验。',
             '', '## 六：过去距离与未来误差', '', '正相关表示过去更远通常对应未来误差更大。下表为每个查询 Spearman 的平均值，置信区间按逻辑序列整簇重采样；未把候选对当独立样本。',
             '', '| 范围 | H | 映射 | 查询数 | 有效相关查询 | rho | 95% CI | Top10 NMSE | 随机 NMSE |', '|---|---:|---|---:|---:|---:|---|---:|---:|']
    for r in summaries:
        if r['group'] == 'ALL' and r['selection'] == 'ordinary':
            lines.append(f"| {r['scope']} | {r['horizon']} | {r['method']} | {r['queries']} | {r['rho_queries']} | {r['mean_query_spearman']} | {r['rho_cluster_ci95']} | {r['mean_top10_nmse']:.4g} | {r['mean_random_nmse']:.4g} |")
    lines += ['', '## 时间与排除校验', '', '以下为常驻索引检索时间；analysis 包含三种映射、随机对照、所有 K 的均值/中位数预测、相关性统计和结果写入，不是单次在线预测耗时。', '', '```json', json.dumps(latency, indent=2), '```',
              '', '## 解释限制', '', '- 正相关属于观察性预测信息，不是因果证明；Top100 距离范围窄，弱相关不代表完全无用。',
              '- ordinary 候选彼此可能重叠；episodes 额外要求间隔至少 L+H。两组均排除 query，episode 候选不足不填充。',
              '- 平坦历史不识别未来幅度：三种映射均退回 query 尾值；相关性未定义的查询单独计数。',
              '- NMSE 分母为 query 所属序列训练方差（最小 1e-12），不是未知未来方差。raw MSE 也保存在明细。',
              '- 权重温度 tau=max(median(d_past^2),1e-8)，仅用过去距离。K=1,4,16,32,100 全部报告，不在测试集上挑参数后宣称泛化增益。',
              '- illness 的 H=244 无法同时容纳训练历史及不相交测试上下文/未来，按 skipped 记录。不同 H 的可用查询不完全相同。',
              '- 各数据组包含的独立文件/序列较少，置信区间仅作探索；需要更多独立数据与固定训练/验证/测试划分复验。',
              '', '![相关性](correlations.png)', '', '![条件误差](distance_future.png)', '', '![预测效果](forecast_skill.png)', '', '![示例](examples.png)']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory', default='results/memory')
    ap.add_argument('--output', default='results/study')
    ap.add_argument('--horizons', type=int, nargs='+', default=[24, 96, 244])
    ap.add_argument('--per-group', type=int, default=20)
    ap.add_argument('--capacity', type=int, default=8192)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    config = json.loads((Path(args.memory)/'experiment.json').read_text(encoding='utf-8'))
    config.update(horizons=args.horizons, per_group=args.per_group, seed=20260916,
                  capacity=args.capacity, requested_k=100, skipped=[])
    source = load_series(config['source_index'])
    began = time.perf_counter(); engine = ForecastIndex(Path(args.memory)/'cascade')
    config['startup_ms'] = (time.perf_counter()-began)*1000
    rng = np.random.default_rng(config['seed'])
    rows, forecasts, timing, examples = [], [], [], []
    pairs_file = (out/'pairs.jsonl').open('w', encoding='utf-8')
    forecast_file = (out/'forecasts.jsonl').open('w', encoding='utf-8')
    try:
        for h in args.horizons:
            pools = {}
            for sid, (meta, x) in enumerate(source):
                train = int(len(x)*config['train_fraction']); last = len(x)-engine.m-h
                if last < train:
                    config['skipped'].append(dict(series_id=sid, group=meta['group'], horizon=h, reason='no disjoint test context plus future'))
                    continue
                # Nonoverlapping query episodes within each horizon/series.
                offset = int(rng.integers(min(engine.m+h, last-train+1)))
                for p in range(train+offset, last+1, engine.m+h):
                    pools.setdefault(meta['group'], []).append((sid, p))
            for group, pool in pools.items():
                chosen = rng.choice(len(pool), size=min(args.per_group, len(pool)), replace=False)
                for case, item in enumerate(chosen):
                    sid, p = pool[int(item)]; meta, x = source[sid]
                    q = x[p:p+engine.m].copy(); truth = x[p+engine.m:p+engine.m+h].copy()
                    scale = max(float(np.var(x[:int(len(x)*config['train_fraction'])])), 1e-12)
                    global_start = int(meta['start'])+p
                    query_id = f'{h}:{sid}:{global_start}'
                    persistence = float(np.mean((truth-q[-1])**2))
                    for scope in ('same_series', 'pooled_train'):
                        result = engine.search(q, h, meta, global_start, scope, args.capacity)
                        tick = time.perf_counter(); violations = 0
                        random_hits = engine.random(h, meta, global_start, scope, rng)
                        rp, rf = engine.arrays(random_hits, h)
                        random_errors = {method: float(np.mean((transfer(q, rp, rf, method)-truth)**2))/scale for method in METHODS}
                        for selection in ('ordinary', 'episodes'):
                            hits = result[selection]
                            if not hits: continue
                            past, future = engine.arrays(hits, h)
                            ds = np.array([float(hit[0]) for hit in hits]); dp = np.sqrt(ds)
                            # Direct, independent recomputation on returned candidates.
                            audit_ds = np.sum((znorm(past)-znorm(q))**2, axis=1)
                            if not np.allclose(ds, audit_ds, atol=2e-5, rtol=1e-7):
                                raise AssertionError('Distance audit failed')
                            for _, hs, hp in hits:
                                hm = engine.idx.manifest['series'][int(hs)]
                                end = int(hm.get('start', 0))+int(hp)+engine.m+h
                                bad = end > hm['train_end'] or (series_key(hm) == series_key(meta) and end > global_start)
                                violations += int(bad)
                            if violations: raise AssertionError('Self/temporal leakage')
                            for method in METHODS:
                                mapped = transfer(q, past, future, method)
                                mse = np.mean((mapped-truth)**2, axis=1); nmse = mse/scale
                                record = dict(query_id=query_id, series_id=sid, group=group, scope=scope, horizon=h,
                                    selection=selection, method=method, count=len(hits), rho=rho(dp, mse),
                                    top10_nmse=float(nmse[:10].mean()), last10_nmse=float(nmse[-10:].mean()),
                                    random_nmse=random_errors[method], top10_minus_random=float(nmse[:10].mean())-random_errors[method],
                                    distance_bin_nmse=[float(nmse[a].mean()) for a in np.array_split(np.arange(len(hits)), 5)] if len(hits) >= 5 else [float(nmse.mean())]*5)
                                rows.append(record)
                                pairs_file.write(json.dumps(dict(record, query_start=global_start,
                                    past_distance=dp.tolist(), future_mse=mse.tolist(), future_nmse=nmse.tolist(),
                                    candidates=[dict(sid=int(s), start=int(pp)) for _, s, pp in hits]), allow_nan=False)+'\n')
                                for k in (1, 4, 16, 32, 100):
                                    if len(hits) < k: continue
                                    for agg in ('mean', 'median'):
                                        pred, eff, tau = aggregate(mapped[:k], ds[:k], agg)
                                        error = float(np.mean((pred-truth)**2))
                                        fr = dict(query_id=query_id, series_id=sid, group=group, scope=scope, horizon=h,
                                            selection=selection, method=method, aggregation=agg, k=k, mse=error,
                                            nmse=error/scale, persistence_mse=persistence, persistence_nmse=persistence/scale,
                                            effective_neighbors=eff, tau=tau)
                                        forecasts.append(fr); forecast_file.write(json.dumps(fr, allow_nan=False)+'\n')
                        if scope == 'same_series' and case == 0 and h == args.horizons[0]:
                            past, future = engine.arrays(result['ordinary'][:16], h)
                            ds = np.array([v[0] for v in result['ordinary'][:16]])
                            preds = {method: aggregate(transfer(q, past, future, method), ds)[0] for method in METHODS}
                            examples.append((q, truth, preds, group, h))
                        timing.append(dict(query_id=query_id, scope=scope, **{k: result[k] for k in ('retrieval_ms', 'eligible', 'retained', 'verified', 'episode_complete')},
                            ordinary_count=len(result['ordinary']), episode_count=len(result['episodes']), violations=violations,
                            analysis_ms=(time.perf_counter()-tick)*1000))
                print(f'H={h} {group}: {len(chosen)} queries done', flush=True)
                (out/'progress.json').write_text(json.dumps(dict(timing=timing, config=config), indent=2), encoding='utf-8')
        summarize(out, rows, forecasts, timing, examples, config)
        (out/'query_statistics.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    finally:
        pairs_file.close(); forecast_file.close(); engine.close()


if __name__ == '__main__':
    main()
