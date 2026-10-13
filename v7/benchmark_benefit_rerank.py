"""Predict correction benefit, then independently guard temporal validation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import fresh_dir,write_json,sha256
from .benchmark_error_correction import load
from .moirai import MODEL_ID,MODEL_REVISION
from .trend_rag import ResidualPool,decompose,normalize
from .benefit_rerank import RIDGES,pair_features,fit_ridge,covariance,select_candidates,guarded_policy,apply_policy


def prepare(cache,source,out,name):
    prior=cache/name;meta=json.loads((prior/'manifest.json').read_text())
    if meta['model_identity']['model_id']!=MODEL_ID or meta['model_identity']['revision']!=MODEL_REVISION:
        raise ValueError('Utility inference requires the pinned repository Moirai model')
    store=Store(source/name/'store');assert store.meta['data_id']==meta['source_data_id']
    cfg=json.loads((source/name/'config.json').read_text())
    interval=cfg['data']['sources'][0]['expected_interval'];period=int(round(86400/interval))
    mode=meta.get('value_mode','errors');window=meta['window'];tables={};libraries={}
    try:
        for s in store.series:
            sid=s['sid'];pool=ResidualPool(store,sid,window,end=s['n'] if meta.get('memory_scope')=='observed' else None,
                stride=meta.get('memory_stride',1))
            raw_past=np.stack([store.window(sid,p,96) for p in pool.starts])
            errors=None
            if mode=='errors':
                path=prior/f'memory_errors_{sid}.npz';assert sha256(path)==meta['memory_files'][str(sid)]
                with np.load(path,allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['starts'],pool.starts);errors=f['errors'].copy()
            libraries[sid]=(pool,raw_past,errors)
        for split in ['train','validation','test']:
            data=load(prior,split);arrays={}
            for i,(sid,start) in enumerate(zip(data['sid'],data['start'])):
                sid,start=int(sid),int(start);s=store.series[sid];pool,past,old_errors=libraries[sid]
                starts=data['candidate_starts'][i];ids=np.searchsorted(pool.starts,starts)
                np.testing.assert_array_equal(pool.starts[ids],starts)
                if np.any(starts+192>min(start,pool.memory_end)):raise AssertionError('Candidate overlaps query')
                x=data['raw_x'][i];std=max(float(s['memory_std']),1e-6)
                if mode=='errors':values=old_errors[ids]
                else:
                    _,r=decompose(x,window);_,mean,scale=normalize(r,pool.floor)
                    _,predicted_r=decompose(np.r_[x,data['base'][i]],window)
                    values=((pool.future[ids]*scale+mean-predicted_r[96:])/std).astype(np.float32)
                fs,hs=data['future_scores'][i],data['history_scores'][i]
                feature=pair_features(x,data['base'][i],start,starts,past[ids],pool.past[ids],values,fs,hs,
                    s['memory_mean'],std,period,window)
                # Labels are attached only after candidate inputs have been built.
                item=dict(features=feature,values=values,starts=starts,
                    prior=.8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6),
                    base=data['base'][i],raw_y=data['raw_y'][i],target=data['target'][i],std=data['std'][i],
                    sid=data['sid'][i],start=data['start'][i],prior_correction=(data['weights'][i,:,None]*data['errors'][i]).sum(0))
                for k,v in item.items():arrays.setdefault(k,[]).append(v)
            table={k:np.asarray(v) for k,v in arrays.items()};tables[split]=table
            np.savez_compressed(out/(split+'.npz'),**table)
            print(cache.name,name,'utility inputs',split,len(table['target']),flush=True)
        write_json(out/'manifest.json',dict(complete=True,source_manifest_sha256=sha256(prior/'manifest.json'),
            source_data_id=meta['source_data_id'],period=period,window=window,memory_scope=meta.get('memory_scope','prefix'),
            memory_stride=meta.get('memory_stride',1),value_mode=mode,seed=meta['seed'],
            splits={s:sha256(out/(s+'.npz')) for s in tables}))
        return tables
    finally:store.close()


def corrections(state,table):
    cov=covariance(state,table['features']);results=[];selected=[];weights=[];permitted=[]
    for i in range(len(table['target'])):
        r=select_candidates(cov[i],table['values'][i],table['prior'][i],table['starts'][i],int(table['sid'][i]))
        results.append(r['correction']);selected.append(r['selected']);weights.append(r['weights']);permitted.append(r['permitted'])
    return dict(correction=np.stack(results),selected=np.stack(selected),weights=np.stack(weights),permitted=np.array(permitted))


def train_ranker(table):
    """Two purged forward folds inside TRAIN only; final validation stays unseen."""
    starts=table['start'];edges=np.quantile(starts,[.5,.75,1.]);scores={};folds=[]
    labels=np.mean(table['values']*table['target'][:,None,:],axis=-1)
    for j in range(2):
        held=np.flatnonzero((starts>=edges[j])&(starts<=edges[j+1] if j==1 else starts<edges[j+1]))
        cutoff=int(starts[held].min());fitted=np.flatnonzero(starts+192<=cutoff)
        if len(fitted)<50 or len(held)<10:raise ValueError('Insufficient purged utility folds')
        folds.append(dict(train_queries=len(fitted),heldout_queries=len(held),train_max_episode_end=int((starts[fitted]+192).max()),
            heldout_first_start=cutoff))
        sub={k:v[held] for k,v in table.items()}
        for penalty in RIDGES:
            state=fit_ridge(table['features'][fitted].reshape(-1,table['features'].shape[-1]),labels[fitted].reshape(-1),penalty)
            r=corrections(state,sub)
            score=float(np.mean((sub['target']-.25*r['correction'])**2))
            scores.setdefault(str(penalty),[]).append(score)
    penalty=min(RIDGES,key=lambda p:(np.mean(scores[str(p)]),-p))
    state=fit_ridge(table['features'].reshape(-1,table['features'].shape[-1]),labels.reshape(-1),penalty)
    return state,dict(folds=folds,forward_nmse=scores,selected_penalty=penalty,selection='purged chronological training folds only')


def experiment(cache,source,out,name):
    tables=prepare(cache,source,out,name);state,selection=train_ranker(tables['train'])
    val,test=tables['validation'],tables['test'];vr=corrections(state,val)
    policy=guarded_policy(val['target'],vr['correction'],val['sid'],val['start'])
    prior_policy=guarded_policy(val['target'],val['prior_correction'],val['sid'],val['start'])
    np.savez(out/'ranker.npz',**state)
    write_json(out/'policy.json',dict(utility=policy,prior=prior_policy,model_selection=selection,
        calibration='first two chronological validation blocks per variable',guard='third block can reject only',
        minimum_calibration_gain=.01,beta_grid=[0.,.1,.25,.5]))
    # No test labels are accessed by either state fit or policy selection above.
    tr=corrections(state,test);delta=apply_policy(policy,tr['correction'],test['sid'])
    old_delta=apply_policy(prior_policy,test['prior_correction'],test['sid'])
    permutation=np.arange(len(test['target']));rng=np.random.default_rng(20261013)
    for sid in np.unique(test['sid']):
        ids=np.flatnonzero(test['sid']==sid);permutation[ids]=rng.permutation(ids)
    wrong=np.stack([tr['weights'][i]@test['values'][permutation[i],tr['selected'][i]] if tr['permitted'][i]
        else np.zeros(96) for i in range(len(permutation))]).astype(np.float32)
    deltas=dict(moirai_direct=np.zeros_like(delta),prior_stable=old_delta,utility_stable=delta,
        utility_fixed_quarter=.25*tr['correction'],utility_shuffled=apply_policy(policy,wrong,test['sid']))
    forecasts={m:test['base']+d*test['std'][:,None] for m,d in deltas.items()}
    np.savez_compressed(out/'predictions.npz',**forecasts,selected=tr['selected'],weights=tr['weights'],permitted=tr['permitted'])
    metrics={m:dict(nmse=float(np.mean(((p.astype(float)-test['raw_y'])/test['std'][:,None])**2)),
        mse=float(np.mean((p.astype(float)-test['raw_y'])**2)),mae=float(np.mean(abs(p.astype(float)-test['raw_y'])))) for m,p in forecasts.items()}
    blocks=np.unique(test['start']);direct=((forecasts['moirai_direct'].astype(float)-test['raw_y'])/test['std'][:,None])**2
    final=((forecasts['utility_stable'].astype(float)-test['raw_y'])/test['std'][:,None])**2
    loss=np.array([[direct[test['start']==b].mean(),final[test['start']==b].mean()] for b in blocks])
    rng=np.random.default_rng(20261013);samples=loss[rng.integers(0,len(blocks),size=(2000,len(blocks)))].mean(1)
    ci=np.quantile(100*(1-samples[:,1]/samples[:,0]),[.025,.975]).tolist()
    run=dict(complete=True,dataset=name,cache=cache.name,queries=len(test['target']),features=test['features'].shape[-1],
        metrics=metrics,gain_percent=100*(1-metrics['utility_stable']['nmse']/metrics['moirai_direct']['nmse']),
        improvement_interval=ci,variables_using_rag=[int(s) for s,p in policy.items() if p['beta']>0],
        ranker_sha256=sha256(out/'ranker.npz'),policy_sha256=sha256(out/'policy.json'),predictions_sha256=sha256(out/'predictions.npz'))
    write_json(out/'run.json',run);print('BENEFIT',cache.name,name,metrics,run['variables_using_rag'],flush=True)
    return run


def report(root):
    root=Path(root);runs=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/run.json'))]
    write_json(root/'summary.json',dict(complete=True,length=96,horizon=96,results=runs,
        caveats=['Same test periods already seen in prior development; exploratory, not independent blind testing.',
        'Guarded fallback reduces validation instability but cannot guarantee every future prediction improves.']))
    lines=['# 96→96：预测收益重排与分段验证回退','',
        '基础模型继续读取原始历史。100候选和候选库保持对应对照不变；0.8未来兼容/0.2历史匹配作为先验。',
        '训练标签是当前基础模型误差与候选修正向量的内积，进而估计加入候选后MSE是否降低。真实未来只在训练标签/指标中使用。',
        '59个历史/预测/案例特征包括原始水平、波动幅度、趋势差、日内相位和误差向量特征，减少去趋势后丢失状态信息的问题。',
        '岭回归参数通过训练区间的两次向前验证选择；每次将训练标签跨度从后续查询完整窗口中清除。',
        '每变量校准：前两个验证时间块选择修正系数，要求两个块都改善且平均至少改善1%；第三块只负责确认，否则回退直接模型。',
        '第三块不能改选另一个系数。系数上限0.5，避免对候选信息的过度依赖。','',
        '|候选库/修正值|子集|直接Moirai|原排序＋稳定回退|收益重排＋稳定回退|改善|启用RAG变量|',
        '|---|---|---:|---:|---:|---:|---|']
    for r in runs:
        m=r['metrics'];lines.append(f"|{r['cache']}|{r['dataset']}|{m['moirai_direct']['nmse']:.6f}|{m['prior_stable']['nmse']:.6f}|{m['utility_stable']['nmse']:.6f}|{r['gain_percent']:.2f}%|{r['variables_using_rag']}|")
    lines+=['','指标是历史库变量方差归一化NMSE，越低越好。未启用RAG的变量输出直接Moirai，不能把回退导致的持平宣称为检索收益。',
        '在线库沿用已有在线协议：只包含查询历史开始前已经完整观测的案例，早期已公开测试观测可成为后续历史库。',
        '同一测试期间曾用于此前开发，此次结果仍属探索性；不能据此保证所有数据集或所有未来窗口超过直接模型。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--caches',nargs='+',default=['v7/runs/rag_error_correction_96_20261013','v7/runs/rag_innovation_online_96_20261013'])
    p.add_argument('--datasets',nargs='+',default=['ETTh1','ETTh2','ETTm1','ETTm2']);args=p.parse_args()
    root=fresh_dir(args.output);source=Path(args.source)
    write_json(root/'protocol.json',dict(length=96,horizon=96,candidates=100,top_k=5,caches=args.caches,
        code_sha256={f:sha256(f) for f in ['v7/benefit_rerank.py','v7/benchmark_benefit_rerank.py']}))
    for cache in map(Path,args.caches):
        for name in args.datasets:experiment(cache,source,fresh_dir(root/cache.name/name),name)
    report(root)


if __name__=='__main__':main()
