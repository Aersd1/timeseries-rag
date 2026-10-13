"""Conservative validation routing of prefix and causal-online retrieval nudges."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import write_json,sha256
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval
from .safe_error_policy import bound_delta,choose_policy,apply_policy


def inputs(cache,online,split):
    data=load(cache,split)
    with np.load(online/f'{split}_trace.npz',allow_pickle=False) as f:pred=f['prediction'].copy()
    old=(data['weights'][:,:,None]*data['errors']).sum(1)*data['std'][:,None]
    modern=pred-data['base']
    deltas=dict(prefix_compatibility=bound_delta(old,data['raw_quantiles'])/data['std'][:,None],
        online_local=bound_delta(modern,data['raw_quantiles'])/data['std'][:,None])
    return data,deltas


def experiment(cache,online,out):
    out.mkdir(parents=True,exist_ok=False)
    val,deltas=inputs(cache,online,'validation');policy=choose_policy(val['target'],deltas,val['sid'],val['start'])
    write_json(out/'policy.json',policy)
    # Expert/scale decisions are persisted before the test inputs are loaded.
    test,td=inputs(cache,online,'test');delta=apply_policy(policy,td,test['sid'])
    forecast=test['base']+delta*test['std'][:,None]
    with np.load(online/'predictions.npz',allow_pickle=False) as f:predictions={k:f[k].copy() for k in f.files}
    predictions['safe_rag']=forecast
    np.savez_compressed(out/'predictions.npz',**predictions)
    metrics={k:metric(p,test) for k,p in predictions.items() if np.asarray(p).shape==test['base'].shape and k!='raw_y'}
    run=dict(complete=True,dataset=cache.name,metrics=metrics,queries=len(test['sid']),
        variables=len(np.unique(test['sid'])),adopted_variables=[int(s) for s,p in policy.items() if p['beta']>0],
        gain_vs_direct_percent=100*(1-metrics['safe_rag']['nmse']/metrics['moirai_direct']['nmse']),
        gain_interval=interval(test['base'],forecast,test),policy_sha256=sha256(out/'policy.json'),
        predictions_sha256=sha256(out/'predictions.npz'),source_cache=str(cache),source_online=str(online),
        protocol='Validation-calibrated bounded retrieval; online suggestions may use only completed past episodes')
    write_json(out/'run.json',run);print('SAFE RESULT',run,flush=True)
    return run


def report(root):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/run.json'))]
    write_json(root/'summary.json',dict(complete=True,results=rows,length=96,horizon=96,
        caveats=['Exploratory modification after prior results were observed; this is not a fresh blind test.',
        'Validation selects one expert/scale; its final block can reject only. No current test target selects a policy.']))
    lines=['# 有界修正与保守验证路由','',
        '候选专家为固定库兼容排序与在线局部方案。先将修正限制在当前Moirai10%–90%分位数跨度的四分之一内。',
        '每变量前两块验证选专家和系数：要求至少75%窗口改善，收益bootstrap下界超过基础验证误差的1%。',
        '最后一块只能确认或拒绝此前选中的专家，要求均值改善且至少75%窗口改善。不满足则使用直接Moirai。','',
        '|数据集|直接Moirai|固定库兼容排序|固定库收益重排|状态召回|在线局部方案|有界保守方案|改善|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics'];lines.append('|'+r['dataset']+'|'+'|'.join(f"{m[k]['nmse']:.6f}" for k in
            ['moirai_direct','prior_stable','utility_shape','utility_state','online_local','safe_rag'])+f"|{r['gain_vs_direct_percent']:.2f}%|")
    lines+=['','采用回退导致的持平不表示RAG有效。当前测试区间此前已经用于开发，阈值与幅度限制属于新增探索性方案，不能据此承诺泛化效果。',
        '在线建议可使用此前已完整观测的测试案例，当前查询与后续真实值始终排除；与固定库离线方案的数据访问协议不同。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='v7/runs/state_raw_other_96_20261013')
    p.add_argument('--online',default='v7/runs/local_error_rag_96_20261013');p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',default=['weather','electricity']);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for name in a.datasets:experiment(Path(a.cache)/name,Path(a.online)/name,root/name)
    report(root)
