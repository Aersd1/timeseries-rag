"""Source-backed H1 report; never label planned mechanism/gate tests completed."""
import argparse
from pathlib import Path
import numpy as np
from .benchmark_memory_access import read
from .benchmark_multimodel_residual import model_path


def pair(test,key='interval95'):
    lo,hi=test[key];return f"{test['gain_percent']:+.2f}% [{lo:+.2f}, {hi:+.2f}]"


def build(root):
    s=read(root/'summary.json');a=read(root/'analysis.json');v=read(root/'verification.json')
    if not s['complete'] or not v['passed']:raise ValueError('Need complete audited H1 matrix')
    p=s['protocol'];out=['# 多模型剩余误差检索：H1主矩阵结果','',
        '**范围：**已完成四个模型、七个数据集、三种子的H1固定探针矩阵。它检验同模型历史案例是否在查询侧校准之外提供增量；不等于H2–H5全部机制干预或可部署门控已验证。所有模型历史512、预测96。',
        '', 'Moirai2为冻结预训练模型；PatchTST、DLinear为本地监督实现；岭回归为闭式直接预测。四模型属于三个家族，不声称跨多个TSFM家族泛化。预训练数据与训练预算不同，只在固定模型内做配对干预。',
        '', '## 正式论证链','',
        '1. 定义增量：真实案例误差预测器R相对同容量查询侧校准器Q的风险下降；同时要求比较随机N与错配P。',
        '2. 排除自拟合：基础模型只训练/选择于前40%，记忆库40–55%，误差预测器55–70%，诊断70–80%，测试80–100%；完整608点窗口各处于单一区间。',
        '3. 统一参考：每个模型同时计算其当前预测与独立旧案例的512→96预测，用旧未来减本模型旧预测作为案例值。',
        '4. 控制容量：所有臂读取完整查询、相同状态信息，128固定随机特征、96输出，非截距有效自由度匹配为min64,N_D/2。案例侧明确读取512点旧历史。',
        '5. 风险判定：R超过Q不自动表示值得使用；还需超过基础B0。细粒度匹配的必要性还需区别于全局记忆均值M。',
        '', '平方误差下降精确等于`2 mean(真实剩余误差×校正)-mean(校正²)`。局部条件下可写为可预测残差信号减迁移偏差平方与估计方差。历史相似度不是这个条件的替代品；不能用真实未来作为事前判据。',
        '', '## 跨数据集主汇总','',
        '正值表示R的归一MSE较参照下降。先在变量内评分、再平均数据集；不是挑选各数据集赢家。下表区间是探索性配对时间块95%区间；显著/实际阈值另按12个主检验的Holm与保守同时区间判定。',
        '', '|模型|R较基础B0|R较查询Q|R较随机N|R较错配P均损失|三主比较都达标准|',
        '|---|---:|---|---|---|---|']
    for m in a['macro']:
        out.append(f"|{m['model']}|{m['real_gain_vs_base_percent']:+.2f}%|{pair(m['tests']['query_only'])}|{pair(m['tests']['matched_random'])}|{pair(m['tests']['deranged'])}|{'是（仍为探索证据）' if m['supports_rag_extra_in_this_probe'] else '否'}|")
    out+=['','当前结果是已探索数据上的协议式复核，不能称新的盲确认；小时数据的独立时间覆盖不足，也限制显著性解释。探针失败不证明其他融合结构或历史信息不可能有效。',
        '', '## 主检验与全局记忆对照','',
        '|模型|比较|点改善|Holm p|保守同时区间|ETT合为一域的点改善|', '|---|---|---:|---:|---|---:|']
    for m in a['macro']:
        for arm,t in m['tests'].items():
            lo,hi=t['familywise_interval95']
            out.append(f"|{m['model']}|R/{arm}|{t['gain_percent']:+.2f}%|{t['holm_p']:.4g}|[{lo:+.2f}, {hi:+.2f}]|{t['domain_sensitivity_gain_percent']:+.2f}%|")
    out+=['','|模型|R较全局记忆M|次要检验族Holm p|达到细粒度匹配增量标准|','|---|---|---:|---|']
    for m in a['macro']:
        t=m['memory_matching_necessity_test'];out.append(f"|{m['model']}|{pair(t)}|{t['holm_p_separate_secondary_family']:.4g}|{'是' if t['supports_matching_beyond_global_memory'] else '否'}|")
    out+=['','## 单元异质性：同一输入长度，不同模型','',
        '各值为R相对Q的归一MSE变化。单元表只描述异质性，不能以某一列显著、另一列不显著推断交互。',
        '', '|数据集|Moirai2|PatchTST|DLinear|岭回归|','|---|---:|---:|---:|---:|']
    for name in p['datasets']:
        vals=[]
        for model in p['models']:
            cell=next(c for c in a['cells'] if c['dataset']==name and c['model']==model)
            vals.append(f"{cell['real_vs']['query_only']['gain_percent']:+.2f}%")
        out.append('|'+name+'|'+'|'.join(vals)+'|')
    out+=['','## 直接检验模型×RAG交互','',
        '比较绝对归一MSE增量`(Q-R)_模型1-(Q-R)_模型2`，使用相同种子与时间重采样。六个双侧检验为独立次要Holm族；模型训练来源不同，不能解释成纯架构因果效应。',
        '', '|模型1/模型2|增量差|95%区间|Holm p|','|---|---:|---|---:|']
    for t in a['paired_model_interactions']:
        lo,hi=t['interval95'];out.append(f"|{t['first']}/{t['second']}|{t['delta_nmse_difference']:+.5g}|[{lo:+.5g}, {hi:+.5g}]|{t['holm_p_separate_secondary_family']:.4g}|")
    out+=['','## 时间覆盖、种子与原始误差','',
        '同步重采样所有变量；块长至少覆盖完整608点跨度：小时数据28天，15分钟/10分钟数据7天。少于8块标记不足，不能用大量通道补足独立时间信息。正式外部确认应规划至少12有效块。',
        '', '|数据集|查询/变量/起点|时间块数/天数|时间覆盖不足|','|---|---:|---:|---|']
    for name in p['datasets']:
        c=next(c for c in a['cells'] if c['dataset']==name)
        out.append(f"|{name}|{c['queries']}/{c['variables']}/{c['origins']}|{c['time_blocks']}/{c['block_days']:g}|{'是' if c['insufficient_time_blocks'] else '否'}|")
    out+=['','|数据集|模型|R较基础原始MSE变化|R较Q三种子归一改善|','|---|---|---:|---|']
    for c in a['cells']:
        raw=100*(1-c['raw_mse']['real']/max(c['raw_mse']['base'],1e-30));seed=' / '.join(f'{x:+.2f}%' for x in c['real_vs']['query_only']['seed_gains_percent'])
        out.append(f"|{c['dataset']}|{c['model']}|{raw:+.2f}%|{seed}|")
    out+=['','## 预算与实现限制','',
        '- 本表重建了时间划分和记忆库，不能与之前固定5+5或不同查询队列的平均分直接相减。原默认预测入口没有变化。',
        '- v1因补齐共有状态信息而终止；v2因发现各阶段日内网格不一致而弃用，已有评分不进入本主表。v3全部窗口、模型选择和探针重新生成，并对齐全局预测时钟；`amendment.json`记录了结构修订及弃用单元数量。',
        '- 共享检索是无训练的形状5+原值5，用于隔离模型条件；它不是原完整学习重排系统的复现。随机臂的时序重叠约束并非与真实臂完全一致，不能只凭R/N做全部因果归因。',
        '- DF匹配控制有限探针的复杂度，不能保证表示/优化等价，也不证明无限函数类的信息论结论。',
        '- H2残差源交换、H3–H5完整机制干预及四级期望效用门控尚未由本主矩阵执行，不将它们列为已证明。',
        '', f"审计：物理窗口{v['physical_windows']:,}；主损失行重建{v['primary_loss_rows_reconstructed']:,}；实际基础预测抽查{v['actual_backbone_forecasts_spot_checked']:,}；各臂探针预测抽查{v['all_arm_probe_forecasts_spot_checked']:,}；D全量Q/R重拟合{v['full_D_query_and_real_probe_refits']}次。",'',
        '正式布局见仓库`v7/MULTI_MODEL_RAG_PROTOCOL.md`，数学命题与F前统计补充见`v7/MULTI_MODEL_RAG_ARGUMENT.md`。本地完整证据为`protocol.json`、`analysis_plan.json`、`frozen.json`、`analysis.json`、`verification.json`及各单元记录。结果、数据和权重不上传。','']
    target=root/'REPORT.md';target.write_text('\n'.join(out),encoding='utf-8');return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(build(Path(p.parse_args().root)))
