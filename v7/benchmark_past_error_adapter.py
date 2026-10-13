"""Train true-error prediction rather than noisy candidate utility labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import write_json,sha256
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval
from .residual_adapter import train
from .contextual_adapter import context_plan
from .past_error_adapter import feature,rerank,WEIGHTS,fit_completed
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,choose_policy,apply_policy
from .moirai import MODEL_ID,MODEL_REVISION


def experiment(cache,source,out,baselines,probes,online=False,adaptive_scale=False,annual_calendar=False):
    out.mkdir(parents=True,exist_ok=False);store=Store(source/'store')
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if meta['source_data_id']!=store.meta['data_id']:raise ValueError('Wrong source store')
    if meta['model_identity']['model_id']!=MODEL_ID or meta['model_identity']['revision']!=MODEL_REVISION:
        raise ValueError('Expected pinned Moirai model')
    if meta.get('value_mode','errors')!='errors' or meta.get('memory_scope','prefix')!='prefix':
        raise ValueError('Expected fixed-prefix model errors')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'));seconds=cfg['data']['sources'][0]['expected_interval']
    period=86400/seconds if seconds<=43200 else None;data={sp:load(cache,sp) for sp in ['train','validation','test']}
    probe_meta=json.loads((probes/'manifest.json').read_text(encoding='utf-8'))
    if probe_meta['contract']['source_manifest_sha256']!=sha256(cache/'manifest.json'):raise ValueError('Wrong probe cache')
    probe={}
    for sp in data:
        if sha256(probes/f'{sp}.npz')!=probe_meta['files'][sp]:raise ValueError('Changed probe cache')
        with np.load(probes/f'{sp}.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['sid'],data[sp]['sid']);np.testing.assert_array_equal(f['start'],data[sp]['start'])
            probe[sp]=f['error'].copy()
    metric_test={k:v.copy() for k,v in data['test'].items() if k in ['raw_y','std','sid','start']}
    if annual_calendar:
        for current in data.values():current['annual_calendar']=True
    if adaptive_scale:
        for current in data.values():
            memory_std=current['std'].copy()
            current['std']=np.maximum(current['raw_x'].std(1),.1*memory_std)
            current['scale_ratio']=current['std']/memory_std
            current['target']=(current['raw_y']-current['base'])/current['std'][:,None]
    plan=context_plan(store);write_json(out/'context_plan.json',plan)
    deltas={kind:{sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']} for kind in ['with_rag','without_rag']}
    selections={}
    try:
        for s in store.series:
            sid=s['sid'];selections[str(sid)]={}
            ids=np.flatnonzero(data['train']['sid']==sid)
            path=cache/f'memory_errors_{sid}.npz'
            if sha256(path)!=meta['memory_files'][str(sid)]:raise ValueError('Changed memory')
            with np.load(path,allow_pickle=False) as f:memory_starts=f['starts'].copy();memory_errors=f['errors'].copy()
            if adaptive_scale:
                old_std=np.array([store.window(sid,int(p),96).std() for p in memory_starts])
                memory_errors=memory_errors*(max(s['memory_std'],1e-6)/np.maximum(old_std,.1*max(s['memory_std'],1e-6)))[:,None]
            for kind in ['with_rag','without_rag']:
                trials=[]
                for weight in (WEIGHTS if kind=='with_rag' else [0.]):
                    ranked=rerank(data['train'],ids,probe['train'][ids],memory_starts,memory_errors,weight) if kind=='with_rag' else None
                    x=feature(store,data['train'],ids,s,period,plan,probe['train'][ids],kind=='with_rag',ranked)
                    state,selection=train(x,data['train']['target'][ids],data['train']['start'][ids])
                    score=float(np.mean(selection['losses'][str(selection['penalty'])]))
                    trials.append((score,weight,state,selection))
                _,weight,state,selection=min(trials,key=lambda t:(t[0],t[1]))
                selection.update(error_state_weight=weight,weight_losses={str(w):score for score,w,_,_ in trials})
                selections[str(sid)][kind]=selection;np.savez(out/f'past_{sid}_{kind}.npz',**state)
                ranked=rerank(data['train'],ids,probe['train'][ids],memory_starts,memory_errors,weight) if kind=='with_rag' else None
                available_x=feature(store,data['train'],ids,s,period,plan,probe['train'][ids],kind=='with_rag',ranked)
                available_y=data['train']['target'][ids].copy();available_start=data['train']['start'][ids].copy()
                for split in ['validation','test']:
                    current=data[split];rows=np.flatnonzero(current['sid']==sid)
                    ranked=rerank(current,rows,probe[split][rows],memory_starts,memory_errors,weight) if kind=='with_rag' else None
                    query_features=feature(store,current,rows,s,period,plan,probe[split][rows],kind=='with_rag',ranked)
                    if online:
                        prediction=np.zeros_like(current['base'][rows],dtype=float)
                        for j in np.argsort(current['start'][rows]):
                            clock=current['start'][rows[j]]
                            fitted=fit_completed(available_x,available_y,available_start,clock,selection['penalty'])
                            prediction[j]=covariance(fitted,query_features[j])
                            # This record is buffered, never eligible before its full192-point episode ends.
                            available_x=np.concatenate([available_x,query_features[j:j+1]])
                            available_y=np.concatenate([available_y,current['target'][rows[j:j+1]]])
                            available_start=np.r_[available_start,clock]
                    else:prediction=covariance(state,query_features)
                    raw=prediction*current['std'][rows,None]
                    deltas[kind][split][rows]=bound_delta(raw,current['raw_quantiles'][rows])/current['std'][rows,None]
            if sid%40==0 or sid==len(store.series)-1:print(cache.name,'past-error adapters',sid+1,'/',len(store.series),flush=True)
        write_json(out/'selection.json',selections);policies={};forecasts={'moirai_direct':data['test']['base']}
        for kind in deltas:
            val=data['validation'];policy=choose_policy(val['target'],{kind:deltas[kind]['validation']},val['sid'],val['start'],require_mae=True)
            policies[kind]=policy
        write_json(out/'policy.json',policies)
        test=data['test']
        for kind in deltas:
            change=apply_policy(policies[kind],{kind:deltas[kind]['test']},test['sid'])
            forecasts['past_'+kind]=test['base']+change*test['std'][:,None]
        for baseline in baselines:
            path=baseline/cache.name/'predictions.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as f:
                    for k in ['raw_y','std','sid','start']:np.testing.assert_array_equal(f[k],metric_test[k])
                    for k in ['prior_stable','utility_shape','utility_state','online_local','safe_rag','adapter_with_rag','adapter_without_rag','contextual_with_rag','contextual_without_rag']:
                        if k in f:forecasts[k]=f[k].copy()
        np.savez_compressed(out/'predictions.npz',**forecasts,**metric_test)
        metrics={k:metric(p,metric_test) for k,p in forecasts.items()}
        run=dict(complete=True,dataset=cache.name,metrics=metrics,queries=len(test['sid']),variables=len(store.series),
            gain_vs_direct_percent=100*(1-metrics['past_with_rag']['nmse']/metrics['moirai_direct']['nmse']),
            gain_interval=interval(test['base'],forecasts['past_with_rag'],metric_test),
            rag_gain_vs_ablation_percent=100*(1-metrics['past_with_rag']['nmse']/metrics['past_without_rag']['nmse']),
            adopted_variables={k:[int(s) for s,p in policies[k].items() if p['beta']>0] for k in policies},
            source_cache=str(cache),source_data_id=store.meta['data_id'],source_manifest_sha256=sha256(cache/'manifest.json'),
            selection_sha256=sha256(out/'selection.json'),policy_sha256=sha256(out/'policy.json'),
            predictions_sha256=sha256(out/'predictions.npz'),period=period,source_probes=str(probes),source_probe_manifest_sha256=sha256(probes/'manifest.json'),context_plan_sha256=sha256(out/'context_plan.json'),
            past_hashes={p.name:sha256(p) for p in out.glob('past_*.npz')},
            annual_calendar=annual_calendar,adaptive_scale=adaptive_scale,online=online,feedback_protocol='Completed earlier train/validation/test episodes can refit the head, only after episode_end <= query_history_start' if online else 'Training-only fixed adapter',
            mechanism='Observed64-to32 Moirai probe error conditions100-candidate reranking and residual adaptation; history-only context; frozen Moirai',
            caveat='Exploratory development on already-inspected test periods. Thresholds are not externally validated.')
        write_json(out/'run.json',run);print('PAST ERROR RESULT',cache.name,{k:v for k,v in run['metrics'].items() if k in ['moirai_direct','past_with_rag','past_without_rag']},flush=True)
    finally:store.close()


def report(root):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/run.json'))]
    write_json(root/'summary.json',dict(complete=True,results=records,length=96,horizon=96))
    lines=['# 历史内部检验误差状态驱动的检索适配','',
        '用历史前64点预测已观测后32点，误差作为状态特征；候选误差匹配强度0/.25/1只在清除重叠的训练折选择。逐变量拟合岭回归，额外输入记忆前缀选定的相关变量历史、周内相位、5个完整旧案例实际未来的水平与幅度对齐延续、案例分歧。',
        '目标是当前训练查询的真实Moirai预测误差，不再间接训练候选收益。岭惩罚只由训练区间两次192点清除间隔的向前验证选择。',
        '用Moirai分位数跨度限制修正，再通过前两块验证的多数窗口/收益下界筛选系数；最后一块只能确认或拒绝。',
        '无RAG对照保留同一批多变量历史和周内特征，只移除旧案例特征。采用系数还要求验证MSE和MAE同时改善。Moirai本体仍冻结。','',
        '|数据集|直接Moirai|无RAG残差适配|有RAG残差适配|相对直接改善|RAG相对消融改善|',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['metrics'];lines.append(f"|{r['dataset']}|{m['moirai_direct']['nmse']:.6f}|{m['past_without_rag']['nmse']:.6f}|{m['past_with_rag']['nmse']:.6f}|{r['gain_vs_direct_percent']:.2f}%|{r['rag_gain_vs_ablation_percent']:.2f}%|")
    lines+=['','本目录的实验开关：']
    for r in records:
        lines.append(f"- {r['dataset']}：按查询幅度训练={r.get('adaptive_scale',False)}；年内周期={r.get('annual_calendar',False)}；在线更新={r.get('online',False)}。{r.get('feedback_protocol','Training-only fixed adapter')}")
    lines+=['','指标为按初始历史库方差归一化NMSE。当前查询真未来不进入输入；旧案例库仍固定且完整结束于查询开始前。',
        '原始MSE、MAE、区间和逐变量采用情况均保留。若移除RAG不变或更好，不能把收益归因于检索。',
        '同一测试段此前已被查看，这是探索性修改；尚未多种子或新测试段验证，不承诺普遍改善。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='v7/runs/state_raw_other_96_20261013')
    p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--output',required=True)
    p.add_argument('--annual-calendar',action='store_true');p.add_argument('--adaptive-scale',action='store_true');p.add_argument('--online',action='store_true');p.add_argument('--probes',required=True);p.add_argument('--datasets',nargs='+',default=['weather','electricity']);p.add_argument('--baselines',nargs='+',
        default=['v7/runs/state_recall_96_20261013','v7/runs/residual_adapter_96_20261013','v7/runs/contextual_adapter_96_20261013']);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for name in a.datasets:experiment(Path(a.cache)/name,Path(a.source)/name,root/name,list(map(Path,a.baselines)),Path(a.probes)/name,a.online,a.adaptive_scale,a.annual_calendar)
    report(root)
