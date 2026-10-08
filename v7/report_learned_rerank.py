"""Audit saved A50 outputs and compare matched original-A test predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report(root):
    summary = read(root/'summary.json')
    protocol = read(root/'protocol.json')
    source = Path(protocol['source'])
    policy = read(root/'policy.json')
    assert summary['complete'] and sha256(root/'policy.json') == summary['policy_sha256']
    method = f"blend_{policy['history_weight']:g}"
    rows, audits, horizons, comparisons = [], [], [], []
    for name in protocol['datasets']:
        test = root/name/'test'
        run = read(test/'run.json')
        c = run['config']
        original_run = read(source/name/'test/run.json')
        assert run['complete'] and run['data_id'] == original_run['data_id'] and c == original_run['config']
        assert sha256(root/name/'validation/metrics.jsonl') == policy['validation_artifacts'][name]
        assert run['checkpoint_sha256'] == sha256(source/name/'train/best.pt')
        assert run['index_sha256'] == sha256(source/name/'index_a/manifest.json')
        quality = read(source/name/'quality.json')
        assert sha256(quality['source']) == quality['sha256']
        metrics = pd.read_json(test/'metrics.jsonl', lines=True)
        previous = pd.read_json(source/name/'test/metrics.jsonl', lines=True)
        keys = ['query', 'sid', 'start', 'horizon']
        assert not metrics.duplicated(keys+['method']).any()
        assert np.isfinite(metrics[['nmse', 'mse', 'mae']]).all().all()
        expected = set(map(tuple, metrics[metrics.method == 'a_original_diverse'][keys].to_numpy()))
        for _, part in metrics.groupby('method'):
            assert set(map(tuple, part[keys].to_numpy())) == expected
        for now, old in [('a_original_diverse', 'a_learned'), ('moirai_direct', 'moirai_direct'), ('persistence', 'persistence')]:
            matched = metrics[metrics.method == now].merge(previous[previous.method == old], on=keys, validate='one_to_one')
            assert len(matched) == len(expected)
            np.testing.assert_allclose(matched.nmse_x, matched.nmse_y, rtol=1e-5, atol=1e-5)
        means = metrics.groupby('method').nmse.mean().to_dict()
        rows.append(dict(dataset=name, queries=run['queries'], original_a=means['a_original_diverse'],
            same_pool_original_order=means['a50_original_order'], future_only=means['blend_0'],
            history_future=means[method], history_only=means['blend_1'], moirai_direct=means['moirai_direct'],
            reduction_vs_original_a_percent=100*(1-means[method]/means['a_original_diverse']),
            reduction_vs_same_pool_percent=100*(1-means[method]/means['a50_original_order']),
            reduction_vs_future_only_percent=100*(1-means[method]/means['blend_0'])))
        pivot = metrics.pivot(index=keys, columns='method', values='nmse')
        for baseline in ['a_original_diverse', 'a50_original_order', 'blend_0']:
            delta = pivot[baseline]-pivot[method]
            series = delta.groupby('sid').mean()
            comparisons.append(dict(dataset=name, baseline=baseline, compared=method,
                query_horizon_win_rate=float((delta > 0).mean()), series_mean_win_rate=float((series > 0).mean())))
        for (h, label), part in metrics.groupby(['horizon', 'method']):
            horizons.append(dict(dataset=name, horizon=h, method=label, nmse=part.nmse.mean()))
        cat = read(source/name/'store/catalog.json')
        details = [json.loads(line) for line in (test/'retrieval.jsonl').read_text().splitlines()]
        assert len(details) == run['queries'] and len({d['query'] for d in details}) == run['queries']
        counts, independent, selected_history = [], [], {'blend_0':[], method:[]}
        for item in details:
            candidates = item['candidates']
            counts.append(len(candidates)); independent.append(item['independent_episodes'])
            identities = {(hit['sid'], hit['start']) for hit in candidates}
            assert len(identities) == len(candidates)
            for hit in candidates:
                assert hit['sid'] == item['sid']
                assert hit['start']+c['length']+max(c['horizons']) <= min(item['start'], cat['series'][hit['sid']]['memory_end'])
            for label, selected in item['selected'].items():
                assert len(selected) == min(5, len(candidates))
                assert all((h['sid'], h['start']) in identities for h in selected)
                if selected:
                    assert np.isclose(sum(h['weight'] for h in selected), 1.)
            for label in selected_history:
                selected_history[label].append(float(np.mean([h['history_nmse'] for h in item['selected'][label]])))
        audits.append(dict(dataset=name, queries=len(details), minimum_candidates=min(counts),
            fraction_full_50=float(np.mean(np.array(counts)==50)), mean_independent_episodes=float(np.mean(independent)),
            history_nmse_future_only=float(np.mean(selected_history['blend_0'])),
            history_nmse_blended=float(np.mean(selected_history[method])), candidate_boundary_violations=0,
            original_a_and_direct_predictions_reproduced=True, original_csv_unchanged=True))
    result = dict(complete=True, policy=policy, results=rows, audits=audits, comparisons=comparisons,
        caveats=['Single seed; 28 test queries per dataset; project-specific 25/60/80% chronological splits.',
            '50 candidates and the selected 5 can overlap; compare against the same-pool baseline to isolate reranking.',
            'The shared history weight was selected on validation; test results did not select weights.',
            'No new encoder training. Moirai is used in both learned representations and future compatibility scoring.'])
    write_json(root/'audit_report.json', result)
    pd.DataFrame(rows).to_csv(root/'comparison.csv', index=False)
    pd.DataFrame(horizons).to_csv(root/'horizon_metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(root/'paired_comparisons.csv', index=False)
    lines = ['# A 学习向量召回 50 个，再按历史与未来重排', '',
        f"验证集在 {policy['candidate_weights']} 中选出统一历史权重 {policy['history_weight']:.0%}、未来权重 {policy['future_weight']:.0%}，随后冻结并测试。所有方法使用相同的 {sum(r['queries'] for r in rows)} 个测试查询。", '',
        '历史分数 H 为分别标准化后的原始历史曲线逐点均方误差。未来分数 F 为候选已知后续走势相对 Moirai 预测分位数的损失，先对配置的预测长度等权平均。', '',
        f"S = {policy['future_weight']:g} × F / max(median(F), 1e-6) + {policy['history_weight']:g} × H / max(median(H), 1e-6)。分数越低越好，取前 5 个，按分数加权预测。中位数在本次 50 个候选内计算。", '',
        '历史项是软约束，没有 0.5 的硬过滤门槛。该权重是归一化评分的系数，不是历史误差上限。', '',
        '## 测试结果', '', 'NMSE 越低越好：每个查询 MSE 除以该变量早期记忆库方差，再对查询及预测长度等权平均。', '',
        '|数据集|原 A|50 候选原排序|50 候选仅未来重排|50 候选历史+未来|相对原 A 误差降低|', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['dataset']}|{r['original_a']:.6f}|{r['same_pool_original_order']:.6f}|{r['future_only']:.6f}|{r['history_future']:.6f}|{r['reduction_vs_original_a_percent']:.2f}%|")
    lines += ['', '原 A 使用相互不重叠的历史候选；本次普通 top50 允许重叠。因此也给出同一 50 候选池中、按学习向量原排序取前 5 个的对照，避免把候选去重策略变化误算成重排收益。', '',
        '## 历史相似度与候选多样性', '',
        '|数据集|凑齐 50 个比例|平均独立历史+未来片段数|仅未来重排：所选历史 NMSE|加入历史项：所选历史 NMSE|', '|---|---:|---:|---:|---:|']
    for r in audits:
        lines.append(f"|{r['dataset']}|{r['fraction_full_50']:.0%}|{r['mean_independent_episodes']:.2f}|{r['history_nmse_future_only']:.4f}|{r['history_nmse_blended']:.4f}|")
    lines += ['', '独立片段数按历史长度+最长未来长度贪心计算；50 个窗口不代表 50 段独立经验。所选历史误差对每个查询的 5 个候选等权平均，再平均查询。所有候选未来均在记忆库内，未发现越界。', '',
        '## 验证集权重选择', '', '|历史权重|等权数据集相对 NMSE 目标（越低越好）|', '|---:|---:|']
    for weight, objective in policy['objective_values'].items():
        lines.append(f'|{float(weight):.0%}|{objective:.6f}|')
    lines += ['', '## 限制', '',
        '- 单一随机种子，每个变量至多 4 个测试窗口。这是初步对照，不是标准论文 ETT 全量基准。',
        '- 本次复用已训练的 A，不重新训练；历史权重只在验证集选一次，测试后不调参。',
        '- 加入历史项可能改善形状匹配，但不能保证真实未来更接近。不同数据集可能受益或退化。',
        '- 原始 A、Moirai 直接预测和最后值预测均与已有逐查询指标在浮点容差内复现；原始 CSV 未改变。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='v7/runs/a50_rerank_20261008')
    report(Path(parser.parse_args().root))
