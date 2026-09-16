"""Readable conclusions and literal past-distance / future-error plots."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results/study')
    args = ap.parse_args(); out = Path(args.results)
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    pairs = [json.loads(line) for line in (out/'pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    horizons = summary['config']['horizons']; methods = ('mean_std', 'endpoint', 'affine')
    fig, axes = plt.subplots(3, len(horizons), figsize=(15, 11), squeeze=False)
    bins = []
    for row, method in enumerate(methods):
        for col, h in enumerate(horizons):
            rr = [r for r in pairs if r['scope'] == 'same_series' and r['selection'] == 'ordinary'
                  and r['method'] == method and r['horizon'] == h]
            x = np.concatenate([r['past_distance'] for r in rr]); y = np.concatenate([r['future_nmse'] for r in rr])
            ax = axes[row, col]
            ax.scatter(x, y, s=3, alpha=.08, rasterized=True, color='steelblue')
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
            means_x, means_y = [], []
            for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (x >= a) & ((x <= b) if j == len(edges)-2 else (x < b))
                if mask.any():
                    means_x.append(float(x[mask].mean())); means_y.append(float(y[mask].mean()))
                    bins.append(dict(method=method, horizon=h, low=float(a), high=float(b),
                                     count=int(mask.sum()), mean_future_nmse=means_y[-1]))
            ax.plot(means_x, means_y, 'o-', color='darkorange', label='Conditional mean by distance bin')
            ax.set_yscale('symlog', linthresh=.01)
            ax.set_title(f'{method}; H={h}')
            ax.set_xlabel('Past z-normalized Euclidean distance')
            ax.set_ylabel('Future MSE / train variance (symlog)'); ax.legend(fontsize=7)
    fig.suptitle('Same-series historical retrieval, ordinary Top100, self excluded\nPooled bins are descriptive; within-query correlations control query difficulty')
    fig.tight_layout(); fig.savefig(out/'past_future_scatter.png', dpi=160); plt.close(fig)
    (out/'distance_bins.json').write_text(json.dumps(bins, indent=2), encoding='utf-8')
    cfg = summary['config']
    ordinary = [r for r in summary['statistics'] if r['scope'] == 'same_series' and r['group'] == 'ALL' and r['selection'] == 'ordinary']
    positive = sum(r['mean_query_spearman'] is not None and r['mean_query_spearman'] > 0 for r in ordinary)
    lines = ['# V4 结论：相似过去能否预测未来？', '',
             f'同序列 {len(ordinary)} 个 H/映射组合中，{positive} 个平均相关系数为正。**“过去非常像”不能保证“未来非常像”。** 原片段已在检索前排除；当前第一名是真正的其他历史案例。', '',
             '## 范围', '',
             f"- {cfg['logical_series']} 条逻辑序列，{cfg['original_unique_points']:,} 个去重原始采样点；训练记忆 {cfg['train_points']:,} 点。范围限于输入的既有索引，不代表全部原 CSV / 全量 GPU。",
             '- 各 H 查询数：'+', '.join(f"H={h}: {next(r['queries'] for r in ordinary if r['horizon'] == h)}" for h in horizons)+f"；两个检索范围共 {sum(r['queries'] for r in summary['latency'])} 次。",
             f"- 上下文 L={cfg['length']}；训练为每条序列前 {cfg['train_fraction']:.0%}。候选 past+future 都在训练段。只取其他合法历史，排除自身及重叠。",
             f"- 无合法不相交测试段的序列/H 组合共 {len(cfg['skipped'])} 个，明细见 summary.json 的 skipped。小样本组不能当成充分验证。", '',
             '## 第六项的直接回答', '',
             '同序列、普通 Top100：对每个查询先计算过去距离和未来误差的 Spearman，再平均。正值说明距离越小，未来误差有偏小的趋势。', '',
             '| H | mean/std rho | endpoint rho | affine rho |', '|---:|---:|---:|---:|']
    for h in horizons:
        ss = [next(r for r in summary['statistics'] if r['scope'] == 'same_series' and r['group'] == 'ALL' and r['horizon'] == h
                   and r['selection'] == 'ordinary' and r['method'] == method) for method in methods]
        lines.append('| '+str(h)+' | '+' | '.join(f"{r['mean_query_spearman']:.3f}" for r in ss)+' |')
    negative = [f"{r['group']}/H={r['horizon']}" for r in summary['statistics'] if r['scope'] == 'same_series' and r['group'] != 'ALL'
                and r['selection'] == 'ordinary' and r['method'] == 'endpoint' and r['mean_query_spearman'] is not None and r['mean_query_spearman'] < 0]
    episodes = [r for r in summary['statistics'] if r['scope'] == 'same_series' and r['group'] == 'ALL' and r['selection'] == 'episodes' and r['method'] == 'mean_std']
    random_comparison = [r for r in ordinary if r['method'] == 'mean_std']
    lines += ['', '相关性大小与置信区间应结合查看，不能解释为每次最相似候选都提供最好的未来。endpoint 出现负相关的同序列分组：'+(', '.join(negative) or '无')+'。', '',
              '去重后 mean/std 结果：'+'；'.join(f"H={r['horizon']}: rho={r['mean_query_spearman']}, 邻居数中位数={r['median_count']}" for r in episodes)+'。距离范围和候选数量不同，不能归因成纯粹的去重效果。', '',
              '最近十个 vs 随机历史，mean/std 平均 NMSE：'+'；'.join(f"H={r['horizon']}: {r['mean_top10_nmse']:.3f} vs {r['mean_random_nmse']:.3f}" for r in random_comparison)+'。', '',
              '## 实际零训练预测', '',
              '固定展示普通 Top-16、加权均值；指标为跨查询平均 NMSE 相对 persistence 的降幅（分母为训练方差）。这里不是 raw MSE 混合不同单位，也没有在测试结果上自动选 K。', '',
              '| H | mean/std 误差下降 | endpoint 误差下降 | affine 误差下降 |', '|---:|---:|---:|---:|']
    for h in horizons:
        ss = [next(r for r in summary['forecasts'] if r['scope'] == 'same_series' and r['horizon'] == h and r['selection'] == 'ordinary'
                   and r['method'] == method and r['aggregation'] == 'mean' and r['k'] == 16) for method in methods]
        lines.append('| '+str(h)+' | '+' | '.join(f"{100*r['skill_vs_persistence']:.1f}%" for r in ss)+' |')
    lines += ['', '这支持继续研究 analog forecasting，但并非每条查询都提升，也尚未与训练型预测模型比较。三种映射中 affine 在这个固定 K 的汇总较好，不能仅凭本次测试将其认定为所有数据集最优。', '',
              '## 效率与不足', '', '| 范围 | 检索 P50 | 检索 P95 |', '|---|---:|---:|']
    for r in summary['latency']:
        lines.append(f"| {r['scope']} | {r['retrieval_p50_ms']:.2f} ms | {r['retrieval_p95_ms']:.2f} ms |")
    lines += ['', '时间是常驻索引 + 有界 8192 候选池 + 普通/去重结果构造，不含冷启动，也不含离线评分。跨序列池未满足每次 100ms；没有做 10B 外推。', '',
              '自匹配/时间重叠违规数：'+str(sum(r['self_or_overlap_violations'] for r in summary['latency']))+'。去重结果未完整认证：'+'；'.join(f"{r['scope']} {r['incomplete_episode_queries']}/{r['queries']}" for r in summary['latency'])+'。这部分只能解释为已返回前缀，不能宣称找到 100 个独立案例。', '',
              '## 下一步判断', '',
              '可以进入 predictive reranker 的研究，但应先固定独立验证/测试划分，用未来误差监督训练排序器，并将跨序列时间可用性按真实时间戳建模。当前阶段完成的是一至六的零训练检验；没有用测试 future 去训练或选择检索器。', '',
              '[详细分组统计和置信区间](REPORT.md) · [全部机器可读结果](summary.json)', '',
              '![过去距离与未来误差](past_future_scatter.png)', '', '![三种映射的相关性](correlations.png)', '', '![预测误差改善](forecast_skill.png)']
    (out/'CONCLUSIONS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__': main()
