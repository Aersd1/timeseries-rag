"""Paired standalone PatchTST comparison using existing immutable test records."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report(root, a50_root, v6_root):
    completed = read(root/'summary.json')
    protocol = read(root/'protocol.json')
    assert completed['complete']
    source = Path(protocol['source'])
    policy = read(a50_root/'policy.json')
    blend = f"blend_{policy['history_weight']:g}"
    results, horizons, paired, training, checks, curves = [], [], [], [], [], []
    for name in protocol['datasets']:
        src, out = source/name, root/name
        run = read(out/'test/run.json')
        assert run['complete']
        c = run['config']
        assert sha256(out/'train/best.pt') == run['checkpoint_sha256']
        t = read(out/'training.json')
        training.append(dict(dataset=name, **{k:t[k] for k in ['epochs', 'train_windows', 'validation_windows', 'parameters', 'training_seconds']}, best_epoch=t['best_epoch_zero_based']+1))
        training_history = read(out/'train/history.json')
        selected_val = training_history[t['best_epoch_zero_based']]['validation_objective']
        assert np.isclose(selected_val, t['best_validation'])
        assert selected_val <= min(h['validation_objective'] for h in training_history)+1e-4
        direct = pd.read_json(out/'test/metrics.jsonl', lines=True)
        old = pd.read_json(src/'test/metrics.jsonl', lines=True)
        rerank = pd.read_json(a50_root/name/'test/metrics.jsonl', lines=True)
        original_v6 = pd.read_json(v6_root/name/'test/metrics.jsonl', lines=True)
        for p in [src/'test/run.json', a50_root/name/'test/run.json', v6_root/name/'test/run.json']:
            saved = read(p)
            assert saved['complete'] and saved['data_id'] == run['data_id'] and saved['config'] == c
        additional = [old[old.method.isin(['a_learned', 'a_joint', 'b_reranked', 'moirai_direct'])],
            rerank[rerank.method == blend].assign(method='a50_history_future'),
            original_v6[original_v6.method == 'v6_learned']]
        data = pd.concat([direct, *additional], ignore_index=True)
        keys = ['query', 'sid', 'start', 'horizon']
        assert not data.duplicated(keys+['method']).any()
        assert np.isfinite(data[['mse', 'nmse']]).all().all()
        expected = set(map(tuple, direct[direct.method == 'persistence'][keys].to_numpy()))
        for method, part in data.groupby('method'):
            assert set(map(tuple, part[keys].to_numpy())) == expected
        catalog = read(src/'store/catalog.json')
        predictions = read(out/'test/predictions.json')
        denominators = {}
        for q in predictions:
            floor = max(catalog['series'][q['sid']]['memory_std']*c['normalization']['memory_std_floor'], 1e-6)
            denominators[q['query']] = max(float(np.std(q['history'])), floor)**2
            saved = data[(data['query'] == q['query']) & (data.method == 'patchtst_direct')]
            for h in c['horizons']:
                mse = float(np.mean((np.array(q['prediction'][:h])-np.array(q['future'][:h]))**2))
                assert np.isclose(mse, saved[saved.horizon==h].mse.iloc[0], rtol=1e-6, atol=1e-7)
        data['history_nmse'] = data.mse/data['query'].map(denominators)
        means = data.groupby('method').nmse.mean().to_dict()
        history_means = data.groupby('method').history_nmse.mean().to_dict()
        results.append(dict(dataset=name, queries=run['queries'], **means, history_nmse=history_means))
        for (h, method), part in data.groupby(['horizon', 'method']):
            horizons.append(dict(dataset=name, horizon=int(h), method=method, nmse=float(part.nmse.mean()), history_nmse=float(part.history_nmse.mean())))
        pivot = data.pivot(index=keys, columns='method', values='nmse')
        for baseline in ['a_learned', 'a50_history_future', 'v6_learned', 'moirai_direct', 'persistence']:
            delta = pivot[baseline]-pivot.patchtst_direct
            series = delta.groupby('sid').mean()
            paired.append(dict(dataset=name, baseline=baseline, method='patchtst_direct',
                relative_nmse_reduction_percent=100*(1-means['patchtst_direct']/means[baseline]),
                query_horizon_win_rate=float((delta>0).mean()), series_mean_win_rate=float((series>0).mean())))
        quality = read(src/'quality.json')
        assert sha256(quality['source']) == quality['sha256']
        checks.append(dict(dataset=name, queries=run['queries'], query_keys_matched=True, source_unchanged=True,
            source_sha256=quality['sha256'], train_test_boundary_verified=True))
        for q in predictions:
            s = catalog['series'][q['sid']]
            assert q['start'] >= s['validation_end']
            assert q['start']+c['length']+max(c['horizons']) <= s['n']
        # Fixed illustrative query: earliest OT test window, irrespective of error.
        example = min((q for q in predictions if q['column']=='OT'), key=lambda q:q['start'])
        from v5.data import Store
        from v6.inference import analog
        from v6.data import scale_floor
        store = Store(src/'store')
        try:
            records = [json.loads(line) for line in (a50_root/name/'test/retrieval.jsonl').read_text().splitlines()]
            selected = next(r for r in records if r['query']==example['query'])['selected'][blend]
            x = np.array(example['history'], dtype=np.float32)
            _, mapped = analog(store, c, x, scale_floor(store.series[example['sid']], c), selected)
            pred50 = np.array([h['weight'] for h in selected])@mapped
            saved = data[(data['query']==example['query']) & (data.method=='a50_history_future')]
            for h in c['horizons']:
                mse = np.mean((pred50[:h]-np.array(example['future'][:h]))**2)
                assert np.isclose(mse, saved[saved.horizon==h].mse.iloc[0], rtol=1e-6, atol=1e-7)
            curves.append(dict(dataset=name, **example, a50_prediction=pred50.tolist()))
        finally:
            store.close()
    pd.DataFrame([{k:v for k,v in r.items() if k!='history_nmse'} for r in results]).to_csv(root/'comparison.csv', index=False)
    pd.DataFrame(horizons).to_csv(root/'horizon_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(root/'paired_comparisons.csv', index=False)
    pd.DataFrame(training).to_csv(root/'training_summary.csv', index=False)
    audit = dict(complete=True, results=results, training=training, comparisons=paired, checks=checks,
        a50_policy=policy, caveats=['Single seed and at most four held-out queries per variable.',
            'Local standalone PatchTST implementation, not an official-paper benchmark reproduction.',
            'Standalone PatchTST uses all stride-32 train windows and validation early stopping; A has a different training budget and pretrained Moirai.'])
    write_json(root/'audit_report.json', audit)
    write_json(root/'forecast_examples.json', curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for ax, example in zip(axes.flat, curves):
        x, y = example['history'], example['future']
        ax.plot(np.arange(-96, 0), x[-96:], color='#929aa5', lw=1, label='History')
        ax.plot(np.arange(len(y)), y, color='#18202c', lw=1.8, label='Actual future')
        ax.plot(example['prediction'], color='#d36b26', lw=1.3, label='PatchTST direct')
        ax.plot(example['a50_prediction'], color='#187b94', lw=1.3, label='A50 history + future')
        ax.axvline(0, color='#929aa5', linestyle=':', lw=.8)
        ax.set(title=f"{example['dataset']} / OT / first test query", xlabel='Steps from forecast origin', ylabel='OT (original scale)')
        ax.grid(alpha=.15)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Fixed examples, selected by time rather than prediction error')
    fig.savefig(root/'forecast_examples.png', dpi=160)
    plt.close(fig)
    lines = ['# 独立 PatchTST 直接预测与检索方案的 ETT 对照', '',
        '本次从随机初始化训练仓库已有的独立 PatchTSTForecast。它不使用 Moirai、不使用 CDF、不检索候选，直接输出未来数值。', '',
        '## 配对测试结果', '', 'NMSE 越低越好。按每个查询的 MSE / 该变量早期记忆库方差计算，再对查询和预测长度 24、96、244 等权平均。各子集 7 个变量、28 个测试窗口。', '',
        '|数据集|独立 PatchTST|原 V6 学习向量|A 学习向量|A50 历史+未来重排|Moirai 直接预测|', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('|'+r['dataset']+'|'+'|'.join(f'{r[m]:.6f}' for m in ['patchtst_direct','v6_learned','a_learned','a50_history_future','moirai_direct'])+'|')
    lines += ['', '## 训练设置', '',
        '- 每个 CSV 单独训练一个共享模型；每次只输入一个变量的历史，不输入其他变量，也不使用时间戳特征。',
        '- 历史长度 244，patch 长度 16、步长 8，隐藏维度 64，2 层、4 头，dropout 0.1；约 57 万参数。',
        '- 展开所有 patch token 后通过线性预测头，一次输出 244 步；24、96 步取该输出的前缀。',
        '- 损失为查询历史标准化空间内的多预测长度 MSE。AdamW，初始学习率 0.0003，余弦衰减，batch size 32。',
        '- 训练区间为 25%–60%，验证区间为 60%–80%，测试为最后 20%。训练和验证按步长 32 取完整滑窗。',
        '- 最多训练 80 轮；验证目标连续 12 轮未改善则早停，按验证目标保存最佳模型。未使用测试结果选择轮数。', '',
        '|数据集|训练窗口|验证窗口|实际训练轮数|选用轮次（从 1 开始）|训练秒数|', '|---|---:|---:|---:|---:|---:|']
    for r in training:
        lines.append(f"|{r['dataset']}|{r['train_windows']}|{r['validation_windows']}|{r['epochs']}|{r['best_epoch']}|{r['training_seconds']:.1f}|")
    lines += ['', '## 按预测长度分解', '', '|数据集|预测长度|PatchTST|A|A50 重排|Moirai 直接预测|', '|---|---:|---:|---:|---:|---:|']
    hframe = pd.DataFrame(horizons)
    for name in protocol['datasets']:
        for h in [24,96,244]:
            values = hframe[(hframe.dataset==name)&(hframe.horizon==h)].set_index('method').nmse
            lines.append(f'|{name}|{h}|{values.patchtst_direct:.6f}|{values.a_learned:.6f}|{values.a50_history_future:.6f}|{values.moirai_direct:.6f}|')
    lines += ['', '## 曲线示例', '', '固定取各子集 OT 变量最早的测试窗口，没有按误差挑选；总体效果以上表所有变量为准。', '', '![固定窗口的实际预测](forecast_examples.png)', '',
        '## 比较边界', '',
        '- 训练预算不同：A 每变量最多 64 个窗口、8 轮，并使用冻结的预训练 Moirai；独立 PatchTST 使用全部训练滑窗并验证早停。这是已训练方案的效果对照，不是相同算力或相同预训练数据的消融。',
        '- 原 V6 和 A 都已含 PatchTST 历史编码分支，但它们通过检索预测；本次独立 PatchTST 是专门学习直接预测的模型，不能与检索模型的辅助预测头混同。',
        '- 本地实现和自定义划分不同于官方论文全量基准，不能将当前结果解释为 PatchTST 架构的一般结论。',
        '- 单一随机种子，每变量 4 个测试窗口；未在测试集上调参。另存查询历史方差归一化指标作为分母敏感性检查，不改变主指标。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='v7/runs/patchtst_direct_20261008')
    parser.add_argument('--a50-root', default='v7/runs/a50_rerank_20261008')
    parser.add_argument('--v6-root', default='v7/runs/v6_baseline_20261008')
    args = parser.parse_args()
    report(Path(args.root), Path(args.a50_root), Path(args.v6_root))
