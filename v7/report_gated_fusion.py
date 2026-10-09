"""Audit and report gated, paired-history/continuation fusion ablations."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256, write_json
from .report_retrieval_finetune import report as audit_report


def report(root,source):
    protocol=json.loads((root/'protocol.json').read_text())
    for name,variants in protocol['reused_controls'].items():
        for variant,files in variants.items():
            for filename,digest in files.items():
                assert sha256(root/name/variant/filename)==digest
    audit_report(root,source)
    summary=json.loads((root/'summary.json').read_text())
    gates=[]
    for name in protocol['datasets']:
        for v in ['rag_gated','rag_paired','rag']:
            run=json.loads((root/name/v/'run.json').read_text())
            expected=next(e for e in protocol['experiments'] if e['variant']==v)
            for key in ['fusion','candidate_dropout','memory_dropout']:
                assert run[key]==expected[key]
            assert run['epochs_limit']==protocol['epochs'] and run['batch_size']==protocol['batch_size']
            controls=json.loads((root/name/'plain/run.json').read_text())
            assert run['config']==controls['config'] and run['train_windows']==controls['train_windows']
            records=json.loads((root/name/v/'predictions.json').read_text())
            g=np.asarray([r['gate_mean'] for r in records])
            assert np.isfinite(g).all() and ((g>=0)&(g<=1)).all()
            gates.append(dict(dataset=name,variant=v,mean=float(g.mean()),min=float(g.min()),max=float(g.max())))
    summary['gates']=gates
    summary['caveats'] += ['Gate-only, paired-memory and dropout ablations have different trainable parameter counts.',
        'Gate diagnostics average hidden tokens and quantile trajectories in the final recursive encoder call; they are not calibrated confidence scores.']
    write_json(root/'summary.json',summary)
    labels=[('moirai_frozen','原始 Moirai'),('moirai_lora','仅 LoRA'),('moirai_legacy','旧融合'),
        ('moirai_gated','只加门控'),('moirai_paired_nodrop','历史/后续分离＋门控'),('moirai_rag_lora','分离＋门控＋随机屏蔽')]
    lines=['# Moirai 2 历史候选融合机制对照','',
        '使用完全相同的 A50 候选缓存、未来 0.8/历史 0.2 的重排权重及前 5 个候选。旧融合与仅 LoRA 使用前次不可变对照结果；三个新增方案重新训练。','',
        '## 相同测试窗口的 NMSE','',
        'MSE 除以变量早期记忆库方差，再对查询及 24、96、244 三个预测长度等权平均，越低越好。','',
        '|数据集|'+'|'.join(label for _,label in labels)+'|','|---|'+'---:|'*len(labels)]
    for r in summary['results']:
        lines.append('|'+r['dataset']+'|'+'|'.join(f'{r[key]:.6f}' for key,_ in labels)+'|')
    lines+=['','## 如何使用历史数值','',
        '- 旧融合：候选历史及后续 patch 共同作为 Key/Value，编码器输出直接加上注意力残差。',
        '- 只加门控：保留旧 token 融合，加入依赖当前表示、辅助表示及候选质量的逐 token 门控。',
        '- 分离融合：每个候选历史用有序窗口 MLP 编码为 Key，其已发生后续用独立 MLP 编码为 Value；按历史相关性读取对应后续。',
        '- 门控质量特征：有效候选比例、重排权重熵、候选后续分歧及候选历史与查询的标准化误差。所有特征只使用查询历史和旧候选。',
        '- 完整版本训练时独立以 20% 概率屏蔽单个候选，以 10% 概率关闭一条查询的全部候选；验证和测试保留所有有效候选。',
        '- 残差输出仍零初始化，微调前与原 Moirai 完全一致；全无有效候选时辅助更新严格为零。门控初始化偏向较小辅助强度。','',
        '## 训练控制','',
        '所有方案使用相同训练窗口、随机种子、批次顺序、学习率及 8 轮预算，按验证集 NMSE 选模型。Moirai 原始参数冻结，更新最后两层 Q/V 的 rank-8 LoRA 和辅助模块。','',
        '|数据集|方案|参数量|最佳轮次|最佳验证 NMSE|','|---|---|---:|---:|---:|']
    for r in summary['training']:
        lines.append(f"|{r['dataset']}|{r['variant']}|{r['trainable_parameters']}|{r['best_epoch']}|{r['best_validation']:.6f}|")
    lines+=['','## 门控诊断','',
        '下表平均值为最后一次递归编码调用中隐藏 token 和分位数轨迹的门控平均，不是预测置信度或候选贡献百分比。','',
        '|数据集|方案|门控均值|最小查询均值|最大查询均值|','|---|---|---:|---:|---:|']
    for g in gates:
        lines.append(f"|{g['dataset']}|{g['variant']}|{g['mean']:.4f}|{g['min']:.4f}|{g['max']:.4f}|")
    lines+=['','## 验证与限制','',
        '- 复用对照文件逐个哈希核对；缓存的查询历史、真实目标及候选数值从原始存储逐条重建核对，候选不得越过记忆边界或查询起点。',
        '- 当前真实未来仅参与训练损失和评价，不能输入预测、候选匹配或门控。测试不选择权重、轮次或超参数。',
        '- 单随机种子，每个子集 28 个测试查询，采用项目自定义划分；本次不构成官方全量基准。',
        '- 各融合方案参数量不同；分离融合还改变了候选编码方式，结果不能仅归因于 Key/Value 分离。',
        '- 早期验证集选择的模型可能在测试区间退化，门控不保证每次检索都能改善预测。','',
        '## 固定预测示例','',
        '每个子集 OT 变量最早的测试窗口，按时间选择。','',
        '![融合方式对照](fusion_examples.png)']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    examples=json.loads((root/'forecast_examples.json').read_text())
    for ax,e in zip(axes.flat,examples):
        name=e['dataset']; qi=e['query']; past=min(96,len(e['history']))
        ax.plot(np.arange(-past,0),e['history'][-past:],color='#999999',lw=1,label='History')
        ax.plot(e['future'],color='#202020',lw=1.7,label='Actual')
        for v,label,color in [('rag_legacy','Legacy','#8a7a69'),('rag_gated','Gate only','#3578b0'),
                ('rag_paired','Paired + gate','#7e60a5'),('rag','Paired + gate + dropout','#d2652d')]:
            rows=json.loads((root/name/v/'predictions.json').read_text()); r=rows[qi]
            assert (r['sid'],r['start'])==(e['sid'],e['start'])
            ax.plot(r['prediction'],color=color,lw=1.1,label=label)
        ax.axvline(0,color='#999999',lw=.8,ls=':'); ax.grid(alpha=.15)
        ax.set(title=name+' / OT / first test query',xlabel='Steps from forecast origin',ylabel='Original scale')
    handles,names=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(root/'fusion_examples.png',dpi=160); plt.close(fig)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'results':summary['results'],'gates':gates},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    args=p.parse_args(); report(Path(args.root),Path(args.source))
