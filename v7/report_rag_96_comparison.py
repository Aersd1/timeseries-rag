"""Compare failed decomposition and two tested retrieval correction mechanisms."""
import argparse
import json
from pathlib import Path
from .common import write_json


def report(split_root,error_root,online_root):
    split_root,error_root,online_root=map(Path,(split_root,error_root,online_root))
    first=json.loads((split_root/'summary.json').read_text());second=json.loads((error_root/'summary.json').read_text())
    third=json.loads((online_root/'summary.json').read_text());rows=[]
    for a,b,c in zip(first['results'],second['results'],third['results']):
        assert a['dataset']==b['dataset']==c['dataset']
        rows.append(dict(dataset=a['dataset'],queries=a['queries'],direct=a['moirai_direct']['nmse'],
            split_rag=a['decomposed_rag']['nmse'],historical_error_rag=b['test_nmse']['rag_adopted'],
            online_innovation_rag=c['test_nmse']['rag_adopted'],error_gain_percent=b['gain_vs_direct_percent'],
            online_gain_percent=c['gain_vs_direct_percent']))
    write_json(error_root/'comparison.json',dict(length=96,horizon=96,results=rows,exploratory=True))
    lines=['# RAG96点预测实验：结果与机制比较','',
        f"同一批{sum(r['queries'] for r in rows)}个测试查询，历史96、预测96，包含各输入数据集的全部数值变量。NMSE越低越好。",'',
        '|子集|查询|原始Moirai|趋势/波动两路+RAG|固定库历史误差修正|在线库波动修正|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+r['dataset']+'|'+str(r['queries'])+'|'+'|'.join(f'{r[m]:.6f}' for m in ['direct','split_rag','historical_error_rag','online_innovation_rag'])+'|')
    lines += ['','查看分解后的冻结预测，识别分解影响：原始预训练模型的输入分布被改变，两条分支独立预测又增加了误差来源。',
        '25点因果移动平均会引入滞后，且不等于真实趋势。趋势/波动的预测误差也可能互相放大；分项误差见第一组报告。',
        '历史波动相似仍不足以保证未来相似；用Moirai兼容性重排不能验证未知真实未来。',
        '修正方案保留原始Moirai预测，监督一个小的外部修正头学习训练查询的真实预测误差，避免破坏基础模型输入。',
        '固定库模式从旧案例的基础模型误差迁移信息；在线模式将当前原始预测投影为趋势/波动，由已知旧案例的波动后续给出修正建议。',
        '100候选、0.8未来兼容性/0.2历史相似、选5保持不变。检索案例历史和后续完整结束于当前历史开始之前，训练也同样约束。','',
        '需要分别检查各子集是否超过直接模型、普通全库校准及打乱匹配的对照；单一子集的收益不能推断为普遍有效。',
        '学习头相对固定修正也不一定更好；验证选择的学习头可能在测试上退化。不能依据测试结果改选。',
        '在线库允许使用已经观测的早期验证/测试记录，索引步长由1改16，属于不同的信息协议，不能与固定训练库混称。',
        '当前结果是单种子、自定义25/60/80%划分、同一测试区间多轮方法开发的探索性结果，不能承诺未知未来上的提升。','',
        '验证选择的部署方式与独立核对：']
    for root,label in [(split_root,'两路分解'),(error_root,'历史误差修正'),(online_root,'在线波动修正')]:
        audit=root/'verification.json'
        if audit.exists():
            data=json.loads(audit.read_text());lines.append('')
            lines.append(label+'：')
            for item in data['audits']:
                interval=item['rag_gain_bootstrap_95_interval_percent']
                lines.append(f"- {item['dataset']}：候选重叠违规{item['overlap_violations']}，配对时间窗口重采样的改善区间[{interval[0]:.2f}%, {interval[1]:.2f}%]。")
    lines+=['','区间未校正多个方案/子集比较；不能等同于预注册独立外部验证。模型数据、完整预测和权重保留本地。']
    (error_root/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split-root',default='v7/runs/trend_rag_96_20261013')
    p.add_argument('--error-root',default='v7/runs/rag_error_correction_96_20261013');p.add_argument('--online-root',default='v7/runs/rag_innovation_online_96_20261013')
    a=p.parse_args();report(a.split_root,a.error_root,a.online_root)
