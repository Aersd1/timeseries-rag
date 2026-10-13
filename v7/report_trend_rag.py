"""Local, reproducible trend-RAG experiment report; never publish data artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import write_json


def report(root):
    root=Path(root); summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    lines=['# 96点历史、96点预测：去趋势后的内部RAG实验','',
        'Moirai直接预测与分解方案使用相同的原始查询。趋势是25点因果滑动平均；波动=原值−趋势。',
        '趋势分支使用冻结Moirai；波动分支通过LoRA及历史/后续配对注意力微调。检索信息进入Moirai隐藏表示后再产生预测，最终趋势与波动相加。没有在模型外混合两条最终预测。','',
        '检索：同变量、固定25%历史库、归一化波动历史的精确距离前100；0.8未来分位兼容性+0.2历史相似；选5，并抑制候选之间重叠。',
        '每个查询历史96、未来96。训练区间25%–60%，验证60%–80%，测试80%–100%；验证和测试使用全部完整且互不重叠的192点窗口。',
        '训练候选的完整192点必须结束于查询开始之前。分解历史的计算不接受查询未来；训练未来只进入损失目标。','',
        'NMSE=每条变量的查询MSE除以历史库变量方差，再等权平均。下表越低越好。原始MSE/MAE也保存在summary.json和query_metrics.csv。','',
        '|子集|测试查询|直接Moirai|分解、冻结|分解、微调无RAG|分解、内部RAG|关闭同一模型的检索|RAG相对直接改善|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['results']:
        vals=[row[m]['nmse'] for m in ['moirai_direct','decomposed_frozen','decomposed_plain_ft','decomposed_rag','rag_memory_disabled']]
        lines.append('|'+row['dataset']+'|'+str(row['queries'])+'|'+'|'.join(f'{x:.6f}' for x in vals)+f"|{row['rag_gain_vs_direct_percent']:.2f}%|")
    lines += ['','正改善表示RAG误差更低，负改善表示退化。不得将负结果解释为已经提升。','',
        '“分解、冻结”用于识别分解影响；“微调无RAG”用于识别微调影响；关闭同一RAG模型的检索用于检验已训练模型是否实际使用了检索。',
        '各模型检查点由重构后的原值验证NMSE选择，包含epoch0；测试结果不能改变检查点。趋势分支未微调。',
        '单种子、自定义划分，尚不是标准ETT榜单结果。25点在小时和15分钟数据中代表不同时间跨度，趋势定义也未专门调参。','',
        '|子集|无RAG最佳轮次|RAG最佳轮次|RAG平均门控|', '|---|---:|---:|---:|']
    for row in summary['results']:
        with np.load(root/row['dataset']/'rag/predictions.npz',allow_pickle=False) as d: gate=float(d['gate_mean'].mean())
        lines.append(f"|{row['dataset']}|{row['runs']['plain']['best_epoch']}|{row['runs']['rag']['best_epoch']}|{gate:.5f}|")
    lines += ['','趋势分支诊断：使用同一因果算子得到的真实未来趋势作为评估目标；这是事后诊断，不进入模型输入。',
        '|子集|趋势NMSE|冻结波动NMSE|RAG波动NMSE|趋势/波动误差交叉项（RAG）|',
        '|---|---:|---:|---:|---:|']
    from .trend_rag import decompose
    diagnostics=[]
    for row in summary['results']:
        name=row['dataset'];catalog=json.loads((root/name/'catalog.json').read_text())
        with np.load(root/name/'test.npz',allow_pickle=False) as d,np.load(root/name/'rag/predictions.npz',allow_pickle=False) as r:
            window=json.loads((root/name/'manifest.json').read_text())['window']
            true_t,true_r=decompose(np.concatenate([d['raw_x'],d['raw_y']],axis=1),window)
            te=d['trend_forecast'].astype(float)-true_t[:,96:]
            re_f=d['frozen'].astype(float)-d['trend_forecast']-true_r[:,96:]
            re_r=r['prediction'].astype(float)-d['trend_forecast']-true_r[:,96:]
            variance=np.array([max(catalog['series'][int(s)]['memory_std'],1e-6)**2 for s in d['sid']])
            vals=[np.mean(np.mean(a,axis=1)/variance) for a in [te**2,re_f**2,re_r**2,2*te*re_r]]
            diagnostics.append(dict(dataset=name,trend_nmse=float(vals[0]),frozen_residual_nmse=float(vals[1]),
                rag_residual_nmse=float(vals[2]),rag_cross_term=float(vals[3])))
            lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
    write_json(root/'component_diagnostics.json',diagnostics)
    verification=root/'verification.json'
    if verification.exists():
        audit=json.loads(verification.read_text()); lines+=['','独立核对：'+json.dumps(audit,ensure_ascii=False),'']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,8),squeeze=False)
    for ax,row in zip(axes.flat,summary['results']):
        name=row['dataset']
        with np.load(root/name/'test.npz',allow_pickle=False) as d, np.load(root/name/'rag/predictions.npz',allow_pickle=False) as r:
            ids=np.flatnonzero(d['sid']==6); i=int(ids[0] if len(ids) else 0)
            ax.plot(np.arange(-96,0),d['raw_x'][i],label='Observed history',color='gray')
            ax.plot(d['raw_y'][i],label='Actual',color='black')
            ax.plot(d['direct'][i],label='Direct Moirai',color='tab:blue')
            ax.plot(d['frozen'][i],label='Decomposed frozen',color='tab:orange',alpha=.6)
            ax.plot(r['prediction'][i],label='Decomposed internal RAG',color='tab:green')
            ax.set_title(name+' / first OT query'); ax.axvline(0,color='gray',linestyle=':'); ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.suptitle('96 context / 96 forecast — fixed first OT examples, not selected by forecast quality')
    fig.tight_layout(rect=(0,.055,1,.96)); fig.savefig(root/'forecast_examples.png',dpi=150); plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(p.parse_args().root)
