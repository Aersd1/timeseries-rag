"""Train-only hyperparameter choice, then causal online validation/test feedback."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import write_json,sha256
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval
from .trend_rag import ResidualPool
from .local_error_rag import LocalErrorRAG,PARAMETERS
from .moirai import MODEL_ID,MODEL_REVISION


def make_model(store,meta,cache,sid,period,parameters):
    pool=ResidualPool(store,sid,25,stride=meta.get('memory_stride',1))
    raw=np.stack([store.window(sid,p,96) for p in pool.starts]);path=cache/f'memory_errors_{sid}.npz'
    assert sha256(path)==meta['memory_files'][str(sid)]
    with np.load(path,allow_pickle=False) as f:
        np.testing.assert_array_equal(f['starts'],pool.starts);errors=f['errors'].copy()
    return LocalErrorRAG(pool,raw,errors,store.series[sid]['memory_std'],period,parameters)


def walk(model,data,indices,previous=None):
    pending=[] if previous is None else list(previous);outputs=[];trace=[]
    for i in indices:
        start=int(data['start'][i]);remaining=[]
        for old,old_data,potential in pending:
            if int(old_data['start'][old])+192<=start:
                model.observe(int(old_data['start'][old]),old_data['raw_x'][old],old_data['raw_y'][old],
                    old_data['base'][old],potential,start)
            else:remaining.append((old,old_data,potential))
        pending=remaining
        result=model.predict(data['raw_x'][i],data['raw_quantiles'][i],start)
        prediction=data['base'][i]+result['beta'][:,None]*result['potentials']*data['std'][i]
        outputs.append(prediction);trace.append(result)
        # The current target is not read by prediction or calibration. Pending
        # episodes become observable only when their FULL192 points are past.
        pending.append((int(i),data,result['potentials']))
    return np.stack(outputs),trace,pending


def experiment(cache,source,out,baseline):
    out.mkdir(parents=True,exist_ok=False);meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('memory_scope','prefix')!='prefix' or meta.get('value_mode','errors')!='errors':raise ValueError('Need prefix-error source')
    if meta['model_identity']['model_id']!=MODEL_ID or meta['model_identity']['revision']!=MODEL_REVISION:
        raise ValueError('Expected pinned Moirai model')
    store=Store(source/'store');assert store.meta['data_id']==meta['source_data_id']
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'));seconds=cfg['data']['sources'][0]['expected_interval']
    period=86400/seconds if seconds<=43200 else None
    train=load(cache,'train');val=load(cache,'validation');test=load(cache,'test');traces={};losses=[];times=[]
    try:
        for s in store.series:
            sid=s['sid'];ids=np.flatnonzero(train['sid']==sid);ids=ids[np.argsort(train['start'][ids])]
            model=make_model(store,meta,cache,sid,period,PARAMETERS)
            predictions,trace,_=walk(model,train,ids)
            errors=(predictions.astype(float)-train['raw_y'][ids,None,:])/train['std'][ids,None,None]
            losses.extend(np.mean(errors**2,axis=-1));times.extend(train['start'][ids])
            potentials=np.stack([r['potentials'] for r in trace]);traces[sid]=(ids,potentials)
            np.savez_compressed(out/f'train_trace_{sid}.npz',indices=ids,potentials=potentials,
                beta=np.stack([r['beta'] for r in trace]))
            del model;gc.collect()
            if sid%20==0 or sid==len(store.series)-1:print(cache.name,'local training',sid+1,'/',len(store.series),flush=True)
        losses=np.asarray(losses);times=np.asarray(times);edges=np.quantile(times,[.5,.75,1.]);scores=[]
        for j in range(2):
            held=(times>=edges[j])&(times<=edges[j+1] if j==1 else times<edges[j+1])
            scores.append(losses[held].mean(0))
        scores=np.stack(scores);selected=int(np.argmin(scores.mean(0)))
        choice=PARAMETERS[selected]
        write_json(out/'selection.json',dict(parameters=PARAMETERS,selected_index=selected,selected=choice,
            forward_train_nmse=scores.tolist(),selection='two final chronological training quarters only',
            protocol='Causal online feedback: past complete train episodes can enter subsequent train queries'))
        outputs={};details={}
        for split,data in [('validation',val),('test',test)]:outputs[split]=np.zeros_like(data['base']);details[split]={}
        for s in store.series:
            sid=s['sid'];model=make_model(store,meta,cache,sid,period,[choice]);ids,potentials=traces[sid]
            pending=[(int(i),train,potentials[j,selected:selected+1]) for j,i in enumerate(ids)]
            for split,data in [('validation',val),('test',test)]:
                ids=np.flatnonzero(data['sid']==sid);ids=ids[np.argsort(data['start'][ids])]
                predicted,trace,pending=walk(model,data,ids,pending)
                outputs[split][ids]=predicted[:,0]
                for j,i in enumerate(ids):
                    r=trace[j]
                    for key,value in [('index',int(i)),('beta',float(r['beta'][0])),('selected_starts',r['selected_starts'][0]),
                        ('recent_candidates',int(r['recent_candidates'][0])),('observed_queries',r['observed_queries'])]:
                        details[split].setdefault(key,[]).append(value)
            del model;gc.collect()
            if sid%20==0 or sid==len(store.series)-1:print(cache.name,'local validation/test',sid+1,'/',len(store.series),flush=True)
        # Hyperparameters were sealed before validation/test processing began.
        forecasts=dict(moirai_direct=test['base'],online_local=outputs['test'])
        if baseline and (baseline/'predictions.npz').exists():
            with np.load(baseline/'predictions.npz',allow_pickle=False) as f:
                for key in ['raw_y','std','sid','start']:np.testing.assert_array_equal(test[key],f[key])
                for key in ['prior_stable','utility_shape','utility_state']:forecasts[key]=f[key].copy()
        np.savez_compressed(out/'predictions.npz',**forecasts,raw_y=test['raw_y'],std=test['std'],sid=test['sid'],start=test['start'])
        for split in ['validation','test']:np.savez_compressed(out/f'{split}_trace.npz',**details[split],prediction=outputs[split])
        metrics={k:metric(p,test) for k,p in forecasts.items()};gain=100*(1-metrics['online_local']['nmse']/metrics['moirai_direct']['nmse'])
        run=dict(complete=True,dataset=cache.name,selected=choice,metrics=metrics,queries=len(test['sid']),variables=len(store.series),
            gain_vs_direct_percent=gain,gain_interval=interval(test['base'],forecasts['online_local'],test),
            validation_metrics=metric(outputs['validation'],val),source_cache=str(cache),source_data_id=store.meta['data_id'],
            source_manifest_sha256=sha256(cache/'manifest.json'),selection_sha256=sha256(out/'selection.json'),
            predictions_sha256=sha256(out/'predictions.npz'),protocol='Causal online: fully observed past evaluation episodes may supply feedback',
            memory_stride=meta.get('memory_stride',1),model_frozen=True)
        write_json(out/'run.json',run);print('LOCAL RESULT',run,flush=True);return run
    finally:store.close()


def report(root):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/run.json'))]
    write_json(root/'summary.json',dict(complete=True,results=records,length=96,horizon=96,
        caveats=['New online protocol; includes earlier evaluation episodes only after192-point observation is complete.',
        'Same previously inspected test periods; exploratory development, not a fresh blind test.',
        'Candidates100/top5 unchanged; library coverage, ranking and causal calibration change together.']))
    lines=['# 在线历史误差案例与局部校准：96→96','',
        '只复用冻结Moirai已有的历史预测。完整旧查询的真实后续在其192点全部结束于当前历史开始前，才加入案例库和校准反馈。',
        '形状/状态召回100个案例；0.8预测未来兼容与0.2历史形状作为先验。可用最近且状态相近的独立旧查询误差，估计当前基础模型误差并重排预期收益，取5个误差案例。',
        '每次查询重新选择过去最多12个独立、相近的已完成查询，利用当时按因果顺序生成的潜在修正进行三块校准。前两块选系数，最后一块只确认或回退。',
        '仅训练区间两次向前回测选择兼容/局部收益重排、状态权重、误差参考数；验证与测试不选超参数。',
        '这是在线反馈协议，旧验证/测试标签在完全观测后可以供后续查询使用，与固定历史库的离线协议不同。','',
        '|数据集|直接Moirai|原兼容排序|固定库收益重排|固定库状态召回|在线局部方案|相对直接改善|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['metrics'];vals=[m[k]['nmse'] if k in m else float('nan') for k in ['moirai_direct','prior_stable','utility_shape','utility_state','online_local']]
        lines.append('|'+r['dataset']+'|'+'|'.join(f'{v:.6f}' for v in vals)+f"|{r['gain_vs_direct_percent']:.2f}%|")
    lines+=['','指标为初始历史库方差归一化NMSE。测试未来没有进入当前查询预测；历史案例和校准查询均必须完整结束于当前历史开始前。',
        '这是联合机制实验，不能将变化归因于单一组件；没有承诺超过基础模型。每变量窗口采样沿用同一缓存，原始预测可独立复算。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='v7/runs/state_raw_other_96_20261013')
    p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--output',required=True)
    p.add_argument('--baseline',default='v7/runs/state_recall_96_20261013');p.add_argument('--datasets',nargs='+',default=['weather','electricity'])
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for name in a.datasets:
        experiment(Path(a.cache)/name,Path(a.source)/name,root/name,Path(a.baseline)/name)
    report(root)


if __name__=='__main__':main()
