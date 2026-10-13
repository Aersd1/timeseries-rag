"""Train true-error prediction rather than noisy candidate utility labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import write_json,sha256
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval
from .residual_adapter import features,train
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,choose_policy,apply_policy
from .moirai import MODEL_ID,MODEL_REVISION


def feature(data,ids,s,period,use_rag):
    rag=(data['weights'][ids,:,None]*data['errors'][ids]).sum(1)
    return features(data['raw_x'][ids],data['base'][ids],data['raw_quantiles'][ids],rag,data['start'][ids],
        data['std'][ids],np.full(len(ids),s['memory_mean']),period,use_rag)


def experiment(cache,source,out,baselines):
    out.mkdir(parents=True,exist_ok=False);store=Store(source/'store')
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if meta['source_data_id']!=store.meta['data_id']:raise ValueError('Wrong source store')
    if meta['model_identity']['model_id']!=MODEL_ID or meta['model_identity']['revision']!=MODEL_REVISION:
        raise ValueError('Expected pinned Moirai model')
    if meta.get('value_mode','errors')!='errors' or meta.get('memory_scope','prefix')!='prefix':
        raise ValueError('Expected fixed-prefix model errors')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'));seconds=cfg['data']['sources'][0]['expected_interval']
    period=86400/seconds if seconds<=43200 else None;data={sp:load(cache,sp) for sp in ['train','validation','test']}
    deltas={kind:{sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']} for kind in ['with_rag','without_rag']}
    selections={}
    try:
        for s in store.series:
            sid=s['sid'];selections[str(sid)]={}
            ids=np.flatnonzero(data['train']['sid']==sid)
            for kind in ['with_rag','without_rag']:
                x=feature(data['train'],ids,s,period,kind=='with_rag')
                state,selection=train(x,data['train']['target'][ids],data['train']['start'][ids])
                selections[str(sid)][kind]=selection;np.savez(out/f'adapter_{sid}_{kind}.npz',**state)
                for split in ['validation','test']:
                    current=data[split];rows=np.flatnonzero(current['sid']==sid)
                    prediction=covariance(state,feature(current,rows,s,period,kind=='with_rag'))
                    raw=prediction*current['std'][rows,None]
                    deltas[kind][split][rows]=bound_delta(raw,current['raw_quantiles'][rows])/current['std'][rows,None]
            if sid%40==0 or sid==len(store.series)-1:print(cache.name,'residual adapters',sid+1,'/',len(store.series),flush=True)
        write_json(out/'selection.json',selections);policies={};forecasts={'moirai_direct':data['test']['base']}
        for kind in deltas:
            val=data['validation'];policy=choose_policy(val['target'],{kind:deltas[kind]['validation']},val['sid'],val['start'])
            policies[kind]=policy
        write_json(out/'policy.json',policies)
        test=data['test']
        for kind in deltas:
            change=apply_policy(policies[kind],{kind:deltas[kind]['test']},test['sid'])
            forecasts['adapter_'+kind]=test['base']+change*test['std'][:,None]
        for baseline in baselines:
            path=baseline/cache.name/'predictions.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as f:
                    for k in ['raw_y','std','sid','start']:np.testing.assert_array_equal(f[k],test[k])
                    for k in ['prior_stable','utility_shape','utility_state','online_local','safe_rag']:
                        if k in f:forecasts[k]=f[k].copy()
        np.savez_compressed(out/'predictions.npz',**forecasts,raw_y=test['raw_y'],std=test['std'],sid=test['sid'],start=test['start'])
        metrics={k:metric(p,test) for k,p in forecasts.items()}
        run=dict(complete=True,dataset=cache.name,metrics=metrics,queries=len(test['sid']),variables=len(store.series),
            gain_vs_direct_percent=100*(1-metrics['adapter_with_rag']['nmse']/metrics['moirai_direct']['nmse']),
            gain_interval=interval(test['base'],forecasts['adapter_with_rag'],test),
            rag_gain_vs_ablation_percent=100*(1-metrics['adapter_with_rag']['nmse']/metrics['adapter_without_rag']['nmse']),
            adopted_variables={k:[int(s) for s,p in policies[k].items() if p['beta']>0] for k in policies},
            source_cache=str(cache),source_data_id=store.meta['data_id'],source_manifest_sha256=sha256(cache/'manifest.json'),
            selection_sha256=sha256(out/'selection.json'),policy_sha256=sha256(out/'policy.json'),
            predictions_sha256=sha256(out/'predictions.npz'),period=period,
            adapter_hashes={p.name:sha256(p) for p in out.glob('adapter_*.npz')},
            mechanism='Per-variable14-feature residual ridge (11 without retrieval); raw Moirai remains frozen',
            caveat='Exploratory development on already-inspected test periods. Thresholds are not externally validated.')
        write_json(out/'run.json',run);print('RESIDUAL RESULT',run,flush=True)
    finally:store.close()


def report(root):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/run.json'))]
    write_json(root/'summary.json',dict(complete=True,results=records,length=96,horizon=96))
    lines=['# 直接学习基础模型预测误差的轻量适配','',
        '逐变量拟合14维岭回归，输入原始历史状态、当前Moirai预测路径与区间、已知日内相位、5个旧案例的加权误差及平滑误差。',
        '目标是当前训练查询的真实Moirai预测误差，不再间接训练候选收益。岭惩罚只由训练区间两次192点清除间隔的向前验证选择。',
        '用Moirai分位数跨度限制修正，再通过前两块验证的多数窗口/收益下界筛选系数；最后一块只能确认或拒绝。',
        '另训练不含3个RAG特征的11维适配器，检验收益是否确实来自案例信息。Moirai本体仍冻结，不是内部微调。','',
        '|数据集|直接Moirai|无RAG残差适配|有RAG残差适配|相对直接改善|RAG相对消融改善|',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['metrics'];lines.append(f"|{r['dataset']}|{m['moirai_direct']['nmse']:.6f}|{m['adapter_without_rag']['nmse']:.6f}|{m['adapter_with_rag']['nmse']:.6f}|{r['gain_vs_direct_percent']:.2f}%|{r['rag_gain_vs_ablation_percent']:.2f}%|")
    lines+=['','指标为按初始历史库方差归一化NMSE。当前查询真未来不进入输入；旧案例库仍固定且完整结束于查询开始前。',
        '原始MSE、MAE、区间和逐变量采用情况均保留。若移除RAG不变或更好，不能把收益归因于检索。',
        '同一测试段此前已被查看，这是探索性修改；尚未多种子或新测试段验证，不承诺普遍改善。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='v7/runs/state_raw_other_96_20261013')
    p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',default=['weather','electricity']);p.add_argument('--baselines',nargs='+',
        default=['v7/runs/state_recall_96_20261013','v7/runs/safe_error_rag_96_20261013']);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for name in a.datasets:experiment(Path(a.cache)/name,Path(a.source)/name,root/name,list(map(Path,a.baselines)))
    report(root)
