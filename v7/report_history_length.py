"""Build a local, source-backed Markdown readout; never publish run artifacts."""
import argparse
from pathlib import Path
from .benchmark_memory_access import read


def gain(base,value):return 100*(1-value/base)


def build(root):
    summary=read(root/'summary.json');audit=read(root/'verification.json')
    if not summary['complete'] or not audit['passed']:raise ValueError('Need complete audited experiment')
    records=summary['records'];out=['# Moirai历史长度诊断',
        '', '固定同一Moirai2权重，预测长度96。所有数据集先完成VAL长度选择并冻结，再运行TEST。原固定5+5默认入口未修改。',
        '', '以下改善均为同一共同队列上原始MSE的相对下降，正值更好。主结果严格使用VAL选择，不拼接TEST赢家。',
        '', '|数据集|TEST查询/起点|VAL选长|M96 MSE|所选 MSE|较M96改善|旧5+5较M96改善|旧5+5较所选改善|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        b=r['metrics']['96']['mse'];s=r['selected_metrics']['mse'];old=r['original5plus5_metrics']['mse']
        out.append(f"|{r['dataset']}|{r['queries']}/{r['origins']}|{r['selected_length']}|{b:.7g}|{s:.7g}|{gain(b,s):+.2f}%|{gain(b,old):+.2f}%|{gain(s,old):+.2f}%|")
    out+=['','## 所有长度的曲线','','TEST曲线用于诊断，不能再据此挑选本轮主策略。',
        '', '|数据集|512较96|2016较96|4096较96|4096较2016（探索性95%区间）|饱和诊断|',
        '|---|---:|---:|---:|---|---|']
    labels={'within_predeclared_one_percent_band':'落在预设±1%带内',
        '4096_still_improves_over2016':'仍有改善','4096_worse_than2016':'4096退化，不能解释为信息已耗尽',
        'insufficient_evidence_of_plateau':'证据不足以判定饱和'}
    for r in records:
        b=r['metrics']['96']['mse'];c=r['comparisons']['length4096_vs2016'];lo,hi=c['exploratory_interval95']
        vals='|'.join(f"{gain(b,r['metrics'][str(k)]['mse']):+.2f}%" for k in (512,2016,4096))
        out.append(f"|{r['dataset']}|{vals}|{c['gain_percent']:+.2f}% [{lo:+.2f}, {hi:+.2f}]，{c['weekly_clusters']}周簇|{labels[r['plateau']['classification']]}|")
    out+=['','## 时间稳定性与尺度核对','','|数据集|四个时间块：所选较96的MSE改善|所选MAE改善|所选TRAIN尺度NMSE改善|',
        '|---|---|---:|---:|']
    for r in records:
        b=r['metrics']['96'];s=r['selected_metrics'];blocks=' / '.join(f"{v['selected_gain_vs96_percent']:+.2f}%" for v in r['blocks'])
        out.append(f"|{r['dataset']}|{blocks}|{gain(b['mae'],s['mae']):+.2f}%|{gain(b['train_nmse'],s['train_nmse']):+.2f}%|")
    out+=['','## 样本、审计与结论边界','',
        '- 覆盖原已冻结抽样队列中具有完整4096点历史的所有查询，不是所有滑动起点。列数不能当作独立时间重复。',
        '- 既有TEST曾多次用于探索；周簇区间仅用于描述不确定性，没有校正多重比较，也不保证周间完全独立。',
        '- 四块是同一固定策略的分期结果，不是四折重新训练。原始MSE、MAE和按TRAIN标准差归一的MSE应一起阅读。',
        '- 旧5+5包含其他训练与输入机制。较长历史超过旧方案能说明旧短历史基线偏弱；旧方案超过长历史不能单独证明检索贡献。',
        '- 本轮只测更多连续目标历史。不能据此排除相关变量、日历条件、误差校准或远期案例的信息，也不能证明剩余误差来自混沌。',
        '- 即使4096不再改善，也只限制当前模型、长度与队列；不足以证明“充分历史”或信息论意义的饱和。',
        '- 审计重建了选择、指标、区间及误差贡献，并核对物理序列、来源哈希和未来区间污染不影响历史输入。',
        '', '|数据集|VAL保留/排除|TEST保留/排除|VAL次要稳健选长（仅诊断）|', '|---|---:|---:|---:|']
    for r in records:
        v=read(root/r['dataset']/'validation_cohort.json');t=read(root/r['dataset']/'test_cohort.json')
        out.append(f"|{r['dataset']}|{v['queries']}/{v['filtered']}|{t['queries']}/{t['filtered']}|{r['robust_secondary_length']}|")
    # Quantify concentration rather than silently excluding unusual Electricity columns.
    e=read(root/'electricity'/'variable_error_contributions.json') if (root/'electricity'/'result.json').exists() else []
    if e:
        total=sum(v['baseline_sse'] for v in e);largest=max(e,key=lambda v:v['baseline_sse'])
        out+=['',f"Electricity原始MSE最大贡献列sid={largest['sid']}，占96点基线SSE的{100*largest['baseline_sse']/total:.2f}%。保留该列，没有按TEST排除。"]
    out+=['','本地证据：`protocol.json`、`frozen.json`、各数据集`policy.json`、`result.json`、`test_metrics.json`与`verification.json`。数据、预测、指标及本报告不发布。','']
    target=root/'REPORT.md';target.write_text('\n'.join(out),encoding='utf-8');return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(build(Path(p.parse_args().root)))
