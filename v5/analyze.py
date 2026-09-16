"""Offline analysis needs no GPU/PyTorch; export a compact review bundle."""
import json
import hashlib
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read_json, write_json


def cluster_interval(frame, column, rng, draws=1000):
    group = frame.groupby('sid')[column].agg(['sum','count'])
    if len(group) < 2: return None
    picks = rng.integers(len(group), size=(draws,len(group)))
    sums = group['sum'].to_numpy(); counts = group['count'].to_numpy()
    result = sums[picks].sum(1)/counts[picks].sum(1)
    return np.percentile(result,[2.5,97.5]).tolist()


def analyze(results, training=None):
    out = Path(results); run = read_json(out/'run.json')
    if not run.get('complete'): raise ValueError('Evaluation is incomplete')
    forecast = pd.read_json(out/'forecasts.jsonl',lines=True)
    retrieval = pd.read_json(out/'retrieval.jsonl',lines=True)
    if forecast.empty or retrieval.empty: raise ValueError('No evaluation rows')
    if forecast.duplicated(['query_id','horizon','method']).any(): raise ValueError('Duplicate forecast rows')
    if not np.isfinite(forecast[['mse','mae','nmse']].to_numpy()).all(): raise ValueError('Nonfinite metrics')
    rng = np.random.default_rng(90210); metrics = []
    for group in ['ALL',*sorted(forecast['group'].unique())]:
        frame = forecast if group == 'ALL' else forecast[forecast['group'] == group]
        for (h, method), f in frame.groupby(['horizon','method']):
            base = frame[(frame.horizon == h)&(frame.method == 'persistence')][['query_id','nmse']].rename(columns={'nmse':'base_nmse'})
            joined = f.merge(base,on='query_id',validate='one_to_one')
            if len(joined) != len(f): raise ValueError('Missing paired persistence')
            joined['paired_difference'] = joined.nmse-joined.base_nmse
            bm = float(joined.base_nmse.mean()); nm = float(joined.nmse.mean())
            metrics.append(dict(group=group,horizon=int(h),method=method,queries=len(joined),
                nmse=nm,mse=float(joined.mse.mean()),mae=float(joined.mae.mean()),
                skill=1-nm/bm if bm > 0 else None,win_rate=float((joined.nmse < joined.base_nmse).mean()),
                mean_candidate_future_nmse=float(joined.mean_candidate_future_nmse.mean()) if joined.mean_candidate_future_nmse.notna().any() else None,
                paired_difference_ci95=cluster_interval(joined,'paired_difference',rng),
                mean_returned=float(joined.returned.mean())))
    recall = []
    for (route,k), frame in retrieval.groupby(['route','k']):
        recall.append(dict(route=route,k=int(k),queries=len(frame),recall=float(frame.oracle_recall.mean()),
            library_coverage=float(frame.oracle_coverage.mean()),fraction=float(frame.candidate_fraction.mean()),
            tie_query_fraction=float(frame.teacher_tie_at10.mean()),ci95=cluster_interval(frame,'oracle_recall',rng)))
    timing = pd.DataFrame(read_json(out/'timing.json')); latencies = []
    for (route,ranker), frame in timing.groupby(['route','ranker']):
        latencies.append(dict(route=route,ranker=ranker,queries=len(frame),
            p50_ms=float(frame.total_ms.median()),p95_ms=float(frame.total_ms.quantile(.95)),
            over_100ms=int((frame.total_ms > 100).sum())))
    result = dict(compression=run['compression'],forecasts=metrics,retrieval=recall,latencies=latencies,
        interpretation='Exploratory series-cluster intervals; columns may be dependent. Sampled library coverage limits strict teacher-ID recall. Ties may admit different equally good neighbors.',
        actual_training_results=run.get('training_completed',False),queries=run['queries'],split=run['split'])
    write_json(out/'analysis.json',result)
    pd.DataFrame(metrics).to_csv(out/'forecast_metrics.csv',index=False)
    pd.DataFrame(recall).to_csv(out/'retrieval_metrics.csv',index=False)
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fm = pd.DataFrame(metrics); fm = fm[fm.group == 'ALL']
    for method, f in fm.groupby('method'):
        axes[0].plot(f.horizon, f.nmse,'o-',label=method)
    axes[0].set_yscale('symlog',linthresh=.1); axes[0].set_title('Held-out forecast NMSE'); axes[0].set_xlabel('Horizon'); axes[0].legend(fontsize=7)
    for route, frame in pd.DataFrame(recall).groupby('route'):
        axes[1].plot(frame.fraction,frame.recall,'o-',label=route)
    axes[1].axhline(.95,ls='--',color='grey'); axes[1].set_xlabel('Returned/library candidates'); axes[1].set_ylabel('Strict V4 Top10 ID recall'); axes[1].legend()
    comp = run['compression']
    axes[2].bar(['points/token','raw/token bytes'],[comp['points_per_token'],comp['raw_to_token_payload_ratio']])
    axes[2].set_title('Sequence compression vs stored token payload\nEmbeddings/postings excluded from second bar')
    fig.tight_layout(); fig.savefig(out/'overview.png',dpi=160); plt.close(fig)
    lines = ['# V5 服务器实验分析','',f"有效查询 {run['queries']}；划分 {run['split']}。",'',
        '## 三个问题','',
        f"1. 压缩：{comp['points_per_token']:.3f} 点/Token；包含 code、duration、mean、std 后，原始 float32/Token 负载比仅 {comp['raw_to_token_payload_ratio']:.3f}。整个索引 {comp['index_total_bytes']:,} bytes，还包含重叠窗口、读出向量和 postings。",
        f"2. 码本使用 {comp['used_codes']}/{comp['codebook_size']}；困惑度 {comp['codebook_perplexity']:.2f}。码本熵不是已经实现的熵编码压缩率。",
        '3. 检索和预测：下表报告真正的 sampled library 搜索；不是只在教师候选池里重排后宣称全库召回。先看 coverage，stride 较大时精确起点缺失会限制 Recall。','',
        '## 预测','', '| H | 方法 | 平均 NMSE | 相对 persistence skill | 胜率 |','|---:|---|---:|---:|---:|']
    for r in metrics:
        if r['group'] == 'ALL':
            skill = 'NA' if r['skill'] is None else f"{r['skill']:.3f}"
            lines.append(f"| {r['horizon']} | {r['method']} | {r['nmse']:.5g} | {skill} | {r['win_rate']:.3f} |")
    lines += ['', '## 检索','', '| 路由 | K | 查询数 | 候选比例 | Recall@K | 库内覆盖上限 |','|---|---:|---:|---:|---:|---:|']
    for r in recall:
        lines.append(f"| {r['route']} | {r['k']} | {r['queries']} | {r['fraction']:.4f} | {r['recall']:.4f} | {r['library_coverage']:.4f} |")
    lines += ['', '## 注意解释','',
        '- 不自动挑测试集上最好的 K、模型或 horizon。训练 checkpoint 只由 validation forecast NMSE 选择；检索权衡需在 validation 比较，再固定模型测试。',
        '- 同一输入 Token 共享码本，shape/predictive 两个读出头；predictive rerank 只看历史 Token，不读取 query future。',
        '- NMSE 按 query 历史标准差归一化，尺度下限为 memory_std*1e-4 与 1e-6 的较大者；与 V4 按整序列训练方差的指标数值不能直接比较。',
        '- 当前 cosine 是分块全扫描；BM25 是 SQLite unigram/bigram 倒排检索。没有声称 ANN、10B 或每次 <100ms。',
        '- strict teacher-ID Recall 对零距离并列敏感，tie_query_fraction 另行记录。最终 exact 仅核验候选，不能证明库外无更优片段。',
        '- 不同 K 行可能是不同查询集合（小库预算截断）；比较时查看 queries。相同查询所有方法有配对 persistence。',
        '- 置信区间按逻辑序列簇重采样，列之间仍可能相关。raw MSE/MAE 混合不同单位不宜作主结论；优先看分组与 NMSE。','',
        '![结果概览](overview.png)','', '回传 analysis_bundle.zip 即可继续分析；默认不包含原始数据、模型权重、query/true future 数组。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    if training:
        for name in ('history.json','environment.json'):
            path = Path(training)/name
            if path.exists(): write_json(out/('training_'+name),read_json(path))
    public = dict(run)
    public['config'] = dict(run['config'])
    public['config']['data'] = dict(source_count=len(run['config']['data']['sources']),source_paths='omitted from review bundle')
    write_json(out/'run_public.json',public)
    names = ['REPORT.md','analysis.json','forecast_metrics.csv','retrieval_metrics.csv','overview.png',
             'forecasts.jsonl','retrieval.jsonl','timing.json','run_public.json']
    names += [p.name for p in out.glob('training_*.json')]
    hashes = {name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in names}
    write_json(out/'bundle_manifest.json',dict(version=5,files=hashes,excludes=['raw data','weights','examples_private.json']))
    with zipfile.ZipFile(out/'analysis_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*names,'bundle_manifest.json']: z.write(out/name,name)
    return result
