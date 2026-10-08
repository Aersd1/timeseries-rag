"""Aggregate only completed, paired benchmark outputs; never pool physical units."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .benchmark_datasets import DATASETS, write


def report(root):
    rows, horizons, diagnostics, checks, comparisons, robustness = [], [], [], [], [], []
    for name in DATASETS:
        folder = root/name
        run = json.loads((folder/'test/run.json').read_text(encoding='utf-8'))
        if not run['complete']:
            raise ValueError(f'Incomplete evaluation: {name}')
        records = pd.read_json(folder/'test/metrics.jsonl',lines=True)
        timing = pd.read_json(folder/'test/timing.jsonl',lines=True)
        config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
        quality = json.loads((folder/'quality.json').read_text(encoding='utf-8'))
        lengths = records.groupby('method').size()
        if len(lengths)!=8 or lengths.nunique()!=1 or not np.isfinite(records[['mse','mae','nmse']]).all().all():
            raise ValueError(f'Missing methods or invalid metrics: {name}')
        if records.duplicated(['query','horizon','method']).any():
            raise ValueError(f'Duplicate scores: {name}')
        means = records.groupby('method').nmse.mean().to_dict()
        rows.append(dict(dataset=name, variables=records.sid.nunique(),queries=records['query'].nunique(),**means))
        for (h,method),part in records.groupby(['horizon','method']):
            horizons.append(dict(dataset=name,horizon=int(h),method=method,n=len(part),nmse=part.nmse.mean(),
                # Raw MSE is retained for reproducibility; mixed-unit columns must not be compared using it.
                mean_raw_mse=part.mse.mean(),mean_raw_mae=part.mae.mean()))
        pivot = records.pivot(index=['sid','query','horizon'],columns='method',values='nmse')
        for left,right in [('b_reranked','b_v4_original_order'),('b_reranked','moirai_direct'),
                           ('a_learned','a_history'),('a_joint','a_history'),('a_learned','b_reranked')]:
            delta = pivot[right]-pivot[left]
            clusters = delta.groupby('sid').mean().to_numpy()
            rng = np.random.default_rng(20261008)
            boots = np.array([rng.choice(clusters,len(clusters),replace=True).mean() for _ in range(2000)])
            comparisons.append(dict(dataset=name,left=left,right=right,
                mean_nmse_reduction=float(delta.mean()),relative_reduction_percent=100*(1-means[left]/means[right]),
                query_horizon_win_rate=float((delta>0).mean()),series_mean_win_rate=float((clusters>0).mean()),
                series_cluster_bootstrap_95=np.quantile(boots,[.025,.975]).tolist()))
        for method,part in timing.groupby('method'):
            diagnostics.append(dict(dataset=name,method=method,queries=len(part),
                complete_topk_fraction=float(part.complete_topk.mean()),p50_ms=float(part.total_ms.median()),
                p95_ms=float(part.total_ms.quantile(.95))))
        cat = json.loads((folder/'store/catalog.json').read_text(encoding='utf-8'))
        series_means = records.groupby(['sid','method']).nmse.mean().unstack()
        prefix_stds = {}
        for s in cat['series']:
            raw = np.memmap(folder/'store'/s['file'],dtype='<f4',mode='r')
            prefix = np.asarray(raw[:s['train_end']],dtype=float)
            prefix_stds[s['sid']] = max(float(prefix[np.isfinite(prefix)].std()),1e-6)
            del raw
        # Diagnostic only: does an unusually small early-memory variance dominate?
        alternate = records.assign(prefix_train_nmse=records.mse/records.sid.map(prefix_stds)**2)
        ordered = series_means.b_v4_original_order.sort_values(ascending=False)
        robustness.append(dict(dataset=name, median_series_nmse=series_means.median().to_dict(),
            prefix_train_nmse=alternate.groupby('method').prefix_train_nmse.mean().to_dict(),
            dominant_v4_series=[dict(sid=int(sid),column=cat['series'][int(sid)]['column'],
                share_of_mean=float(value/ordered.sum()),memory_std=cat['series'][int(sid)]['memory_std'],
                training_prefix_std=prefix_stds[int(sid)]) for sid,value in ordered.head(3).items()],
            note='Primary memory-variance NMSE is unchanged. These are distribution/normalization sensitivity diagnostics, using no test-fitted statistics.'))
        n_candidates, empty, used, independent = [], {}, [], []
        for line in (folder/'test/retrieval.jsonl').open(encoding='utf-8'):
            r = json.loads(line)
            if not r['hits']: empty[r['method']] = empty.get(r['method'],0)+1
            for hit in r['hits']+r.get('candidates',[]):
                assert hit['sid']==r['sid']
                assert hit['start']+config['length']+max(config['horizons']) <= min(r['start'],cat['series'][r['sid']]['memory_end'])
            if r['method']=='b_reranked':
                n_candidates.append(len(r['candidates']));used.append(len(r['hits']))
                ends, count = -1, 0
                for start in sorted(hit['start'] for hit in r['candidates']):
                    if start>=ends:
                        count+=1; ends=start+config['length']+max(config['horizons'])
                independent.append(count)
        assert len(n_candidates)==run['queries']
        checks.append(dict(dataset=name,variables=quality['variables'],evaluated_variables=records.sid.nunique(),
            queries=run['queries'],complete_20_fraction=float(np.mean(np.array(n_candidates)==20)),
            minimum_b_candidates=min(n_candidates), minimum_b_selected=min(used),empty_queries=empty,
            mean_disjoint_episodes_within_b_pool=float(np.mean(independent)),
            candidate_future_leakage_violations=0,config_matches=run['config']==config,
            original_csv_unchanged=hashlib.sha256(Path(quality['source']).read_bytes()).hexdigest()==quality['sha256'],
            timing_caveat='Separate CPU/GPU execution across datasets; do not compare latency across hardware.'))
        assert quality['variables']==records.sid.nunique() and run['config']==config and checks[-1]['original_csv_unchanged']
    result = dict(complete=True, datasets=rows, horizon_metrics=horizons,diagnostics=diagnostics,
        paired_comparisons=comparisons,checks=checks,robustness=robustness,
        metric='Arithmetic mean of per-query MSE divided by the same series memory-only variance, averaged equally over configured horizons.',
        caveats=['Fixed 8-epoch single-seed A training, at most 64 training windows per variable.',
            'At most four disjoint test windows per variable; illness has two.',
            'B ordinary top20 allows overlapping candidates; this differs from the earlier NREL episodes experiment.',
            'All query horizons share a candidate pool constrained by the longest horizon.',
            'Custom 25/60/80% splits, not the standard published forecasting benchmark protocol.',
            'No V6/CDF control was retrained; these results do not establish that Moirai compresses better than CDF.',
            'Test results were not used for parameter tuning; bootstrap treats series as clusters, although variables can be correlated.'])
    write(root/'summary.json',result)
    pd.DataFrame(rows).to_csv(root/'dataset_metrics.csv',index=False)
    pd.DataFrame(horizons).to_csv(root/'horizon_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(root/'paired_comparisons.csv',index=False)
    pd.DataFrame([dict(dataset=r['dataset'],**r['prefix_train_nmse']) for r in robustness]).to_csv(root/'training_prefix_nmse.csv',index=False)
    pd.DataFrame([dict(dataset=r['dataset'],**r['median_series_nmse']) for r in robustness]).to_csv(root/'median_series_nmse.csv',index=False)
    lines = ['# 八个数据集的 Moirai 双方案实验结果','',
        '所有数值列均参与。A 保留纯学习向量与联合检索两种输出；B 为 V4 一次取 20 个候选后用 Moirai 重排、取前 5 个。',
        '', '指标为 NMSE，越低越好：每个预测窗口的 MSE 除以该变量检索库数据的方差，再平均查询及预测长度。不同变量原始单位不同，因此不直接平均原始 MSE 作主要比较。',
        '', '|数据集|变量|查询|A 学习向量|A 联合检索|B 重排|V4 原排序|Moirai 直接预测|最后值|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|'+ '|'.join([r['dataset'],str(r['variables']),str(r['queries'])]+[f'{r[m]:.6f}' for m in
            ['a_learned','a_joint','b_reranked','b_v4_original_order','moirai_direct','persistence']])+'|')
    lines += ['', '## B 重排相对 V4 原排序', '', '|数据集|平均 NMSE 降低比例（负值表示退化）|获益变量比例|', '|---|---:|---:|']
    for r in comparisons:
        if r['left']=='b_reranked' and r['right']=='b_v4_original_order':
            lines.append(f"|{r['dataset']}|{r['relative_reduction_percent']:.2f}%|{r['series_mean_win_rate']:.2%}|")
    lines += ['', '平均误差下降不表示所有变量都获益，获益变量比例见上表。此处比较的是 B 的重排与未来分数加权整体，与原排序及历史距离加权的对照。']
    lines += ['', '## 候选与缺失检查', '', '|数据集|B 凑齐 20 个比例|A 联合检索无候选查询数|', '|---|---:|---:|']
    for r in checks:
        lines.append(f"|{r['dataset']}|{r['complete_20_fraction']:.1%}|{r['empty_queries'].get('a_joint',0)}/{r['queries']}|")
    gate_path = root/'ett_gate_audit.json'
    if gate_path.exists():
        gate_audit = json.loads(gate_path.read_text(encoding='utf-8'))
        result['ett_gate_audit'] = [{k:v for k,v in r.items() if k!='queries_detail'} for r in gate_audit]
        write(root/'summary.json',result)
        if all(r['empty_despite_eligible_windows']==0 for r in gate_audit):
            lines += ['', '对 ETT 全部检索库窗口另行核查：联合检索为空的查询均没有任何窗口通过 0.5 的历史 NMSE 门槛，未发现存在合格窗口却返回空结果的情况。明细见 ett_gate_audit.json。']
    lines += ['', '## 归一化敏感性', '',
        '主指标完整保留所有变量，不删离群序列。同时保存 median_series_nmse.csv（先算每变量平均再取变量中位数）与 training_prefix_nmse.csv（使用前 60% 数据的方差归一化同一组预测误差）。两者为诊断，未用测试集拟合尺度。']
    for r in robustness:
        d=r['dominant_v4_series'][0]
        if d['share_of_mean']>.5:
            lines.append(f"- {r['dataset']}：变量 {d['column']} 占 V4 平均 NMSE 的 {d['share_of_mean']:.2%}；检索库标准差 {d['memory_std']:.6g}，训练前缀标准差 {d['training_prefix_std']:.6g}。该数据集的均值受到单变量明显影响。")
    lines += ['', '## 实验边界','',
        '- 前 25% 固定检索库；25%–60% 训练 A；60%–80% 选择验证目标最优轮次；最后 20% 测试。',
        '- 非 illness：历史 244，预测 24/96/244 点；illness：历史 36，预测 24/36/48/60 点。',
        '- 每 CSV 训练一个共享 A 编码器；逐列单变量预测与同序列检索。A 训练 8 轮，每变量最多 64 个训练窗口，单随机种子。',
        '- 每变量最多 4 个不重叠的测试窗口；illness 每变量仅 2 个。A 最多 5 个邻居，B 20 选 5；A 历史门控和不重叠筛选保持不变。',
        '- weather 去除完全重复记录，补齐时间轴并将 -9999 标记缺失；具体计数见 quality.json。训练和评估均排除跨缺失窗口，无插值。原始 CSV 不变。',
        '- 此轮不是论文标准数据划分，不应与文献 MSE 表格直接对比；未训练 V6/CDF 对照，不能证明表示压缩优于 CDF。',
        '- 所有检索候选的已知未来均位于检索库内。测试真实未来只参与误差计算。',
        '- 未用测试结果调参。序列级 bootstrap 未完全消除不同变量间相关性，置信区间仅作探索参考。',
        '', '逐预测长度指标见 horizon_metrics.csv，逐查询记录见各数据集 test/metrics.jsonl，配置、文件指纹与数据检查见 protocol.json 和 quality.json。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('v7/runs/all_datasets_20261008'))
    report(p.parse_args().output)
