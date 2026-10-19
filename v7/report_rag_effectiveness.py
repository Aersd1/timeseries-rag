"""Local mechanism report, separating primary, secondary and post-hoc evidence."""
import argparse
from pathlib import Path
import numpy as np
from .benchmark_memory_access import read
from .rag_effectiveness import FEATURES


def pct(base,pred):return 100*(1-pred/max(base,1e-30))


def interval(c):
    lo,hi=c['exploratory_interval95']
    return f"{c['gain_percent']:+.2f}% [{lo:+.2f}, {hi:+.2f}]"


def build(root):
    s=read(root/'summary.json');v=read(root/'verification.json');post=read(root/'sensitivity_posthoc.json')
    if not s['complete'] or not v['passed']:raise ValueError('Need audited complete experiment')
    records=s['records'];out=['# RAG有效条件：机制先导实验','',
        '冻结Moirai2、原5+5案例和融合强度0.25。不增加预测网络。验证集确定阈值和符号预测器，再冻结全部七数据集后运行TEST。主试验运行约'+f"{s['seconds']:.1f}秒，复用已审计缓存而不是重新进行基础模型推理。",
        '', '结论：历史案例有效性必须相对于当前基础预测和融合机制定义；相似性本身不足。当前简单可观测特征尚不能可靠跨数据集识别收益符号。部分收益可以被不检索的融合解释，精确配对的额外价值总体较小。事后方向干预提示冗余修正值得追查，但未建立稳定超越强基线的新方案。',
        '', '## 主结果：同样的误差迁移，在不同基础预测上表现不同',
        '', '正值为MSE改善。原始MSE是主指标；区间为探索性周簇95%区间，未校正多重比较。',
        '', '|数据集|查询/起点|长历史长度|加到M96|加到长历史Moirai|真实优于匹配随机|真实优于错配均损失|',
        '|---|---:|---:|---:|---|---|---|']
    for r in records:
        c=r['comparisons'];out.append(f"|{r['dataset']}|{r['queries']}/{r['origins']}|{r['baseline_length']}|{c['short_residual']['real_vs_base']['gain_percent']:+.2f}%|{interval(c['long_residual']['real_vs_base'])}|{interval(c['long_residual']['real_vs_random'])}|{interval(c['long_residual']['real_vs_deranged'])}|")
    out+=['','真实优于随机并不蕴含优于基础预测。例如ETTm1真实案例优于随机，但加到长历史基础预测仍退化。错配对照对每组五个使用四种无固定点循环置换，并平均损失而非预测，避免把集成收益当作错配效应。',
        '', '## 次要算子：融合方式本身改变结论',
        '', '直接将历史对齐的案例后续与当前预测融合，和迁移旧Moirai96残差使用完全相同的案例、历史权重与强度。该算子不是根据TEST逐数据集挑出的替代方案。',
        '', '|数据集|直接后续融合较长历史基线|较匹配随机|原始MSE/归一MSE结论差异|', '|---|---|---|---|']
    for r in records:
        c=r['comparisons']['long_continuation']
        out.append(f"|{r['dataset']}|{interval(c['real_vs_base'])}|{interval(c['real_vs_random'])}|归一MSE变化{c['real_gain_nmse_percent']:+.2f}%|")
    out+=['','该算子在部分数据集有改善，但Electricity、Traffic明显退化；它不能作为通用替代。旧案例参考是96点Moirai，当前查询基线更长，故误差迁移包含参考不匹配风险，不能将失败直接归因于“历史没有信息”。',
        '', '## 预测前能否甄别：排除目标数据集标签的符号预测',
        '', '系数仅用其他六数据集VAL拟合。模型长度选择仍用过自身VAL，因此不是整个算法的零样本外推。所有符号模型只针对主误差迁移算子；概率超过1/2不保证降低MSE。',
        '', '|数据集|查询侧AUROC|加入案例特征AUROC|案例侧平衡准确率|案例侧门控MSE改善|事后oracle改善（不可部署）|',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        q=r['sign_prediction']['query_only'];c=r['sign_prediction']['case_aware']
        out.append(f"|{r['dataset']}|{q['auc']:.3f}|{c['auc']:.3f}|{c['balanced_accuracy']:.3f}|{r['gates']['case_aware']['gain_vs_no_rag']['gain_percent']:+.2f}%|{r['gates']['oracle_future_only']['gain_vs_no_rag']['gain_percent']:+.2f}%|")
    out+=['','事后oracle只在真实未来显示该校正有益时使用，表示固定候选/算子的机会，不是可实现性能。简单案例自检、低分歧规则和逻辑回归都没有建立稳定的普适收益判据。',
        '', '## 预设因素分组：不能只看数据集平均数',
        '', '分组阈值来自VAL。以下所有指标为TEST按TRAIN标准差归一的MSE改善。原方案一列与M96比较；固定算子一列与长历史Moirai比较。各组是描述性关联，不是独立随机试验。']
    for feature in FEATURES:
        out+=['',f'### {feature}','','|数据集|低组/高组查询数|固定算子低组/高组收益|原5+5低组/高组收益|','|---|---:|---:|---:|']
        for r in records:
            g=r['factor_groups'][feature];lo=g.get('low');hi=g.get('high')
            if lo and hi:out.append(f"|{r['dataset']}|{lo['queries']}/{hi['queries']}|{lo['gain_nmse_percent']:+.2f}% / {hi['gain_nmse_percent']:+.2f}%|{lo['original_gain_vs96_nmse_percent']:+.2f}% / {hi['original_gain_vs96_nmse_percent']:+.2f}%|")
    out+=['','## 合成干预：受控反例，不是Moirai实验',
        '', '20个配对种子，每种256查询。历史分布、检索索引和噪声抽样固定，分别改变历史—后续关系、案例是否过时、基础模型已知状态比例。基础预测使用已知玩具生成函数，不能将本表数字当成Moirai收益。',
        '', '|条件|MSE改善|配对种子95%区间|','|---|---:|---|']
    for r in read(root/'synthetic.json')['results']:
        lo,hi=r['paired_seed_interval95'];out.append(f"|{r['condition']}|{r['gain_mse_percent']:+.2f}%|[{lo:+.2f}, {hi:+.2f}]|")
    out+=['','## 事后敏感性分析：单独标记，不改写主结果',
        '', '以下控制在主结果查看后增加。统一用于全部数据集，不挑选每个数据集的赢家。',
        '', '|数据集|历史均值融合|末16点均值融合|重复96融合|短模型预测融合|去掉重复方向|同能量缩小校正|去方向优于同能量缩小|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    keys=('history_mean','last16_level','repeat96','short_forecast','remove_positive_shared_direction','norm_matched_shrinkage')
    for r in post['records']:
        values='|'.join(f"{r['query_only_and_direction_controls'][k]['gain_vs_long']['gain_percent']:+.2f}%" for k in keys)
        out.append(f"|{r['dataset']}|{values}|{interval(r['direction_vs_equal_norm_shrinkage'])}|")
    out+=['','前三种和短模型融合完全不检索。去方向干预仅去除校正与长短模型差值的正向投影；同能量控制用于排除简单缩小校正。方向干预也可能只是基础预测校准，仍必须看随机/错配同样处理后的对照。',
        '', '|数据集|去方向真实优于去方向随机|去方向真实优于去方向错配|原5+5历史距离双向控制相关|固定算子冗余方向双向控制相关|',
        '|---|---|---|---:|---:|']
    for r in post['records']:
        a=r['within_variable_and_origin_correlations']
        out.append(f"|{r['dataset']}|{interval(r['projected_real_vs_random'])}|{interval(r['projected_real_vs_deranged'])}|{a['history_distance']['original_pipeline']:.3f}|{a['redundancy_alignment']['fixed_long_residual']:.3f}|")
    out+=['','双向控制剔除固定变量效应和具体预测起点效应。粗分组中“相似组更有效”不能直接解释为相似度的因果作用。逐变量VAL阈值分组及覆盖列数另见`sensitivity_posthoc.json`，部分数据集能同时覆盖两端的变量很少，不能概括全部列。',
        '', '## 审计、范围与下一项可证伪问题','',
        f"- 已复核{v['physical_case_pairs']:,}个真实/随机历史—后续对及不与查询96/未来96重叠的时间边界，重建全部主预测、指标、门控、阈值、分类器和合成种子。",
        '- 测试期已被多轮探索、时间起点有限，列数不等于独立时间重复。当前结果是机制先导证据，不是新盲测的普适结论。',
        '- 5+5和基础模型权重均未修改，未将任何新算子或门控设为默认。原完整方案含其他学习分支，不能把其收益单独归因于检索。',
        '- 主要支持的后续问题是“给定当前基础预测，历史案例是否提供互补校正”，不是单独判断序列是否重复。应在新时间段比较真实/随机/错配的基模型条件化融合，并加入同容量查询侧校准。',
        '- 本轮没有找到可靠的通用事前选择器；没有证明剩余误差来自混沌，也没有证明RAG本身无用。',
        '', '相关研究：[When Does Retrieval Help Time-Series Forecasting?](https://arxiv.org/abs/2609.20193)、[ROAR](https://arxiv.org/abs/2605.24911)。问题本身已有研究，创新性应来自可复现的新边界、可迁移判断和严格对照，而不是换个门控名字。',
        '', '本地原始证据：`protocol.json`、`frozen.json`、`summary.json`、`synthetic.json`、`verification.json`、`sensitivity_posthoc.json`及每数据集的行级结果。数据、指标、系数和报告不上传。','']
    target=root/'REPORT.md';target.write_text('\n'.join(out),encoding='utf-8');return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(build(Path(p.parse_args().root)))
