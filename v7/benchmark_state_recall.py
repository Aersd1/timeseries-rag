"""Isolate state-preserving recall from the unchanged benefit-ranking mechanism."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import write_json,sha256
from .moirai import MODEL_ID,MODEL_REVISION
from .trend_rag import ResidualPool
from .state_recall import StateRecall,STATE_WEIGHTS
from .benchmark_error_correction import load
from .benefit_rerank import RIDGES,pair_features,fit_ridge,covariance,guarded_policy,apply_policy


def corrections(state,table):
    """Vectorized equivalent of select_candidates; preserve its tie ordering."""
    cov=covariance(state,table['features']);e=table['values'].astype(float)
    gain=.5*cov-.0625*np.mean(e*e,axis=-1)
    scale=np.maximum(np.median(abs(gain),axis=1),1e-4)
    prior=table['prior'];starts=table['starts'];score=gain-.02*scale[:,None]*prior
    tie=np.lexsort((starts,prior),axis=1);rows=np.arange(len(score))
    overlap=np.zeros_like(score);remaining=np.ones_like(score,dtype=bool);selected=[]
    for _ in range(5):
        adjusted=np.where(remaining,score-.1*scale[:,None]*overlap,-np.inf)
        chosen=tie[rows,np.argmax(np.take_along_axis(adjusted,tie,axis=1),axis=1)]
        selected.append(chosen);remaining[rows,chosen]=False
        overlap=np.maximum(overlap,np.maximum(0.,1-abs(starts-starts[rows,chosen,None])/192))
    selected=np.stack(selected,axis=1)
    logits=score[rows[:,None],selected]/scale[:,None]
    weights=np.exp(np.clip(logits-logits.max(1,keepdims=True),-50,0));weights/=weights.sum(1,keepdims=True)
    correction=np.sum(weights[:,:,None]*e[rows[:,None],selected],axis=1)
    expected=np.sum(weights*cov[rows[:,None],selected],axis=1)
    permitted=.5*expected-.0625*np.mean(correction**2,axis=1)>0
    correction[~permitted]=0
    return dict(correction=correction.astype(np.float32),selected=selected,weights=weights.astype(np.float32),permitted=permitted)


def train_ranker(table):
    starts=table['start'];edges=np.quantile(starts,[.5,.75,1.]);scores={};folds=[]
    labels=np.mean(table['values']*table['target'][:,None,:],axis=-1)
    for j in range(2):
        held=np.flatnonzero((starts>=edges[j])&(starts<=edges[j+1] if j==1 else starts<edges[j+1]))
        cutoff=int(starts[held].min());fitted=np.flatnonzero(starts+192<=cutoff)
        if len(fitted)<50 or len(held)<10:raise ValueError('Insufficient purged utility folds')
        folds.append(dict(train_queries=len(fitted),heldout_queries=len(held),
            train_max_episode_end=int((starts[fitted]+192).max()),heldout_first_start=cutoff))
        sub={k:v[held] for k,v in table.items()}
        for penalty in RIDGES:
            state=fit_ridge(table['features'][fitted].reshape(-1,table['features'].shape[-1]),labels[fitted].reshape(-1),penalty)
            r=corrections(state,sub)
            scores.setdefault(str(penalty),[]).append(float(np.mean((sub['target']-.25*r['correction'])**2)))
    penalty=min(RIDGES,key=lambda p:(np.mean(scores[str(p)]),-p))
    state=fit_ridge(table['features'].reshape(-1,table['features'].shape[-1]),labels.reshape(-1),penalty)
    return state,dict(folds=folds,forward_nmse=scores,selected_penalty=penalty,
        selection='purged chronological training folds only')


class Builder:
    def __init__(self,cache,source):
        self.cache=Path(cache);self.meta=json.loads((self.cache/'manifest.json').read_text(encoding='utf-8'))
        if self.meta['model_identity']['model_id']!=MODEL_ID or self.meta['model_identity']['revision']!=MODEL_REVISION:
            raise ValueError('State recall requires the pinned repository Moirai model')
        if self.meta.get('value_mode','errors')!='errors' or self.meta.get('memory_scope','prefix')!='prefix':
            raise ValueError('This isolated experiment requires fixed-prefix model-error memory')
        self.store=Store(Path(source)/'store')
        if self.store.meta['data_id']!=self.meta['source_data_id']:raise ValueError('Data mismatch')
        cfg=json.loads((Path(source)/'config.json').read_text(encoding='utf-8'));interval=cfg['data']['sources'][0]['expected_interval']
        self.period=86400/interval if interval<=43200 else None
        self.libraries={}

    def library(self,sid):
        if sid not in self.libraries:
            # Bound memory on datasets with hundreds of variables.
            if len(self.libraries)>=16:self.libraries.pop(next(iter(self.libraries)))
            s=self.store.series[sid]
            pool=ResidualPool(self.store,sid,self.meta['window'],stride=self.meta.get('memory_stride',1))
            past=np.stack([self.store.window(sid,p,96) for p in pool.starts])
            path=self.cache/f'memory_errors_{sid}.npz'
            if sha256(path)!=self.meta['memory_files'][str(sid)]:raise ValueError('Memory changed')
            with np.load(path,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['starts'],pool.starts);errors=f['errors'].copy()
            self.libraries[sid]=(pool,past,errors,StateRecall(pool,past,s['memory_std'],self.period))
        return self.libraries[sid]

    def table(self,split,weight):
        data=load(self.cache,split);arrays={}
        for i,(sid,start) in enumerate(zip(data['sid'],data['start'])):
            sid,start=int(sid),int(start);s=self.store.series[sid];pool,past,errors,recall=self.library(sid)
            m=recall.retrieve(data['raw_x'][i],data['raw_quantiles'][i],start,weight,self.meta['window'])
            if weight==0:
                np.testing.assert_array_equal(m['candidate_starts'],data['candidate_starts'][i])
                np.testing.assert_array_equal(m['history_scores'],data['history_scores'][i])
                np.testing.assert_allclose(m['future_scores'],data['future_scores'][i],rtol=0,atol=1e-12)
            ids=np.searchsorted(pool.starts,m['candidate_starts']);values=errors[ids]
            fs,hs=m['future_scores'],m['history_scores'];std=max(s['memory_std'],1e-6)
            features=pair_features(data['raw_x'][i],data['base'][i],start,m['candidate_starts'],past[ids],pool.past[ids],
                values,fs,hs,s['memory_mean'],std,self.period or 1,self.meta['window'])
            oldids=np.searchsorted(pool.starts,m['selected_starts'])
            item=dict(features=features,values=values,starts=m['candidate_starts'],prior=m['prior'],
                state_scores=m['state_scores'],base=data['base'][i],raw_y=data['raw_y'][i],target=data['target'][i],
                std=data['std'][i],sid=data['sid'][i],start=data['start'][i],
                prior_correction=(m['weights'][:,None]*errors[oldids]).sum(0))
            for k,v in item.items():arrays.setdefault(k,[]).append(v)
            if (i+1)%2000==0:
                print(self.cache.name,split,'state weight',weight,'built',i+1,'/',len(data['sid']),flush=True)
        table={k:np.asarray(v) for k,v in arrays.items()}
        print(self.cache.name,split,'state weight',weight,'queries',len(table['sid']),flush=True)
        return table

    def close(self):self.store.close()


def metric(pred,table):
    error=pred.astype(float)-table['raw_y']
    return dict(nmse=float(np.mean((error/table['std'][:,None])**2)),mse=float(np.mean(error**2)),mae=float(np.mean(abs(error))))


def interval(base,pred,table):
    direct=np.mean(((base.astype(float)-table['raw_y'])/table['std'][:,None])**2,axis=1)
    final=np.mean(((pred.astype(float)-table['raw_y'])/table['std'][:,None])**2,axis=1)
    loss=np.array([[direct[table['start']==b].mean(),final[table['start']==b].mean()] for b in np.unique(table['start'])])
    rng=np.random.default_rng(20261013);sample=loss[rng.integers(0,len(loss),size=(2000,len(loss)))].mean(1)
    return np.quantile(100*(1-sample[:,1]/sample[:,0]),[.025,.975]).tolist()


def experiment(cache,source,out,baseline=None):
    out.mkdir(parents=True,exist_ok=False);builder=Builder(cache,source);states={};selections={}
    try:
        for weight in STATE_WEIGHTS:
            table=builder.table('train',weight);state,selection=train_ranker(table)
            states[weight]=state;selections[str(weight)]=selection
            del table;gc.collect()
            print(cache.name,'training forward CV',weight,selection['forward_nmse'],flush=True)
        selected=min(STATE_WEIGHTS,key=lambda w:(np.mean(selections[str(w)]['forward_nmse'][str(selections[str(w)]['selected_penalty'])]),w))
        write_json(out/'selection.json',dict(state_weight=selected,candidates=selections,
            selected_before_validation_and_test=True,selection='minimum mean two purged training-fold losses; tie favors smaller weight'))
        forecasts={};run_methods={};reference=None
        for weight in sorted(set([0.,selected])):
            branch=out/f'weight_{weight}';branch.mkdir();state=states[weight]
            val=builder.table('validation',weight);vr=corrections(state,val)
            policy=guarded_policy(val['target'],vr['correction'],val['sid'],val['start'])
            prior_policy=guarded_policy(val['target'],val['prior_correction'],val['sid'],val['start'])
            np.savez(branch/'ranker.npz',**state)
            write_json(branch/'policy.json',dict(utility=policy,prior=prior_policy,state_weight=weight))
            del val,vr;gc.collect()
            test=builder.table('test',weight);tr=corrections(state,test)
            delta=apply_policy(policy,tr['correction'],test['sid']);pred=test['base']+delta*test['std'][:,None]
            label='utility_shape' if weight==0 else 'utility_state'
            forecasts[label]=pred
            np.savez_compressed(branch/'test.npz',**test)
            np.savez_compressed(branch/'predictions.npz',prediction=pred,selected=tr['selected'],weights=tr['weights'],permitted=tr['permitted'])
            run_methods[label]=dict(metrics=metric(pred,test),gain_vs_direct_percent=100*(1-metric(pred,test)['nmse']/metric(test['base'],test)['nmse']),
                gain_interval=interval(test['base'],pred,test),active_variables=[int(s) for s,p in policy.items() if p['beta']>0],
                ranker_sha256=sha256(branch/'ranker.npz'),policy_sha256=sha256(branch/'policy.json'),
                test_sha256=sha256(branch/'test.npz'),predictions_sha256=sha256(branch/'predictions.npz'))
            if weight==0:
                forecasts['moirai_direct']=test['base'];run_methods['moirai_direct']=dict(metrics=metric(test['base'],test))
                prior=test['base']+apply_policy(prior_policy,test['prior_correction'],test['sid'])*test['std'][:,None]
                forecasts['prior_stable']=prior;run_methods['prior_stable']=dict(metrics=metric(prior,test))
                reference={k:test[k].copy() for k in ['base','raw_y','std','sid','start']}
                if baseline and (baseline/'predictions.npz').exists():
                    with np.load(baseline/'predictions.npz',allow_pickle=False) as old:
                        np.testing.assert_allclose(pred,old['utility_stable'],rtol=0,atol=1e-6)
            else:
                for k in reference:np.testing.assert_array_equal(test[k],reference[k])
            del test,tr;gc.collect()
        if selected==0:
            forecasts['utility_state']=forecasts['utility_shape'].copy()
            run_methods['utility_state']=dict(run_methods['utility_shape'])
        run=dict(complete=True,dataset=cache.name,selected_state_weight=selected,methods=run_methods,
            gain_state_vs_shape_percent=100*(1-run_methods['utility_state']['metrics']['nmse']/run_methods['utility_shape']['metrics']['nmse']),
            state_vs_shape_interval=interval(forecasts['utility_shape'],forecasts['utility_state'],reference),
            queries=len(reference['sid']),variables=len(np.unique(reference['sid'])),
            memory_stride=builder.meta.get('memory_stride',1),selection_sha256=sha256(out/'selection.json'),
            source_manifest_sha256=sha256(cache/'manifest.json'),zero_weight_exact_candidate_reproduction=True,
            prior_baseline_replayed=bool(baseline and (baseline/'predictions.npz').exists()),
            source_cache=str(cache),source_data_id=builder.meta['source_data_id'],
            period=builder.period,window=builder.meta['window'])
        np.savez_compressed(out/'predictions.npz',**forecasts,**reference)
        run['predictions_sha256']=sha256(out/'predictions.npz');write_json(out/'run.json',run)
        print('STATE RESULT',cache.name,run,flush=True);return run
    finally:builder.close()


def report(root):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/run.json'))]
    missing=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/unsupported.json'))]
    write_json(root/'summary.json',dict(complete=True,results=rows,unsupported=missing,length=96,horizon=96,
        caveats=['Seeded sampled windows, all variables; not official benchmark splits.',
        'ETT reuses previously inspected test periods. Additional-dataset observations are exploratory.',
        'State weight/ridge selected on train only; validation chooses bounded corrections. No guaranteed improvement.']))
    lines=['# 保留状态的召回：96→96配对比较','',
        '同一候选库、查询、冻结原始Moirai预测和收益重排；唯一主要变量是召回距离中的状态权重。',
        '形状距离加权水平、对数幅度、线性趋势、循环日内相位。只用训练区间两次带192点清除间隔的向前回测选择状态权重与岭参数。',
        '0权重保留原召回；验证前两块选择修正比例、第三块只允许拒绝。测试真实未来只用于指标。',
        '收益重排的59项特征原本就含状态；这里检验状态是否应更早进入100个候选的召回。','',
        '|数据集|变量|测试窗口|库步长|状态权重|直接Moirai NMSE|原0.8/0.2排序＋回退|收益重排|状态召回＋收益重排|状态相对收益重排改善|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['methods'];lines.append(f"|{r['dataset']}|{r['variables']}|{r['queries']}|{r['memory_stride']}|{r['selected_state_weight']}|"+
            '|'.join(f"{m[x]['metrics']['nmse']:.6f}" for x in ['moirai_direct','prior_stable','utility_shape','utility_state'])+
            f"|{r['gain_state_vs_shape_percent']:.2f}%|")
    for r in missing:lines.append(f"\n{r['dataset']}: 直接模型NMSE={r['moirai_direct_nmse']:.6f}，其余方案不适用：{r['reason']}。")
    lines+=['','NMSE以初始历史库各变量方差归一化，越低越好。状态权重0意味着训练选择保留原方案，不代表新机制有效。',
        '其他数据集统一全变量、每变量32个随机训练窗口、12个均匀非重叠验证/测试窗口；库步长16。ETT复用原步长1及原查询。',
        '日内相位是固定等间隔序列索引的循环相位，不是节假日或真实日历状态；weather的缺测边界由store.runs排除。',
        '置信区间按同步时间块重采样，仅作为探索性不确定性估计。未覆盖所有测试窗口，也没有多随机种子或全新盲测。']
    diagnostic=root/'robust_diagnostics.json'
    if diagnostic.exists():
        lines+=['','## 补充：逐变量NMSE中位数','',
            '保留上述预先定义的平均NMSE；中位数只诊断极端变量对均值的影响，不用于选择模型。','',
            '|数据集|直接Moirai|原兼容重排|收益重排|状态召回|状态方案改善/退化/持平变量数|',
            '|---|---:|---:|---:|---:|---|']
        for r in json.loads(diagnostic.read_text(encoding='utf-8')):
            m=r['methods'];state=m['utility_state']
            lines.append('|'+r['dataset']+'|'+'|'.join(f"{m[x]['median_variable_nmse']:.6f}" for x in
                ['moirai_direct','prior_stable','utility_shape','utility_state'])+
                f"|{state['improved_variables']}/{state['worse_variables']}/{state['unchanged_variables']}|")
    quality=root/'electricity_quality.json'
    if quality.exists():
        q=json.loads(quality.read_text(encoding='utf-8'));p=q['profiles']
        lines+=['',f"electricity {q['column']}号变量：历史库标准差{p['memory']['std']:.6f}，测试期标准差{p['test']['std']:.6f}。",
            '原CSV与store数值一致、时间戳无重复且每小时间隔完整。已确认尺度变化；实际原因未知。该变量仍保留在所有主指标中。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--baseline')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    write_json(root/f'protocol_{Path(a.cache).name}.json',dict(state_weights=STATE_WEIGHTS,train_selection_only=True,
        length=96,horizon=96,candidates=100,top_k=5,source_cache=a.cache,datasets=a.datasets,
        code_sha256={p:sha256(p) for p in ['v7/state_recall.py','v7/benchmark_state_recall.py','v7/prepare_state_benchmark.py']}))
    for name in a.datasets:
        cache=Path(a.cache)/name;out=root/name
        if (out/'run.json').exists():continue
        if (cache/'unsupported.json').exists():
            out.mkdir(exist_ok=True);write_json(out/'unsupported.json',json.loads((cache/'unsupported.json').read_text(encoding='utf-8')));continue
        experiment(cache,Path(a.source)/name,out,Path(a.baseline)/name if a.baseline else None)
    report(root)


if __name__=='__main__':main()
