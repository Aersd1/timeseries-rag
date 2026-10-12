"""Audit scratch future-regime retrieval, future compatibility and forecasting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v5.data import Store
from .common import sha256,write_json
from .future_cluster import eligible


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def report(root,source):
    complete=read(root/'complete.json')
    assert complete['complete']
    results=[]; quality=[]; training=[]; audits=[]; examples=[]
    for name in complete['datasets']:
        folder=root/name; src=source/name; meta=read(folder/'index_manifest.json')
        c=meta['source_config']; span=c['length']+max(c['horizons'])
        catalog=read(src/'store/catalog.json'); source_quality=read(src/'quality.json')
        assert meta['complete'] and meta['data_id']==catalog['data_id']
        assert meta['feature_policy']=='history_mean_scaled_ordered_future_v1'
        assert sha256(source_quality['source'])==source_quality['sha256']
        assert sha256(folder/'index.npz')==meta['index_sha256']
        assert sha256(folder/'train_future_features.npy')==meta['training_features_sha256']
        ta=read(folder/'training_audit.json')
        assert ta['data_id']==catalog['data_id'] and ta['clusters_fit_split']=='train_only'
        assert ta['catalog_sha256']==sha256(src/'store/catalog.json')
        with np.load(folder/'training_pairs.npz',allow_pickle=False) as pairs:
            qs=pairs['query_sid']; qt=pairs['query_start']; cs=pairs['candidate_sid']; ct=pairs['candidate_start']; valid=pairs['eligible']
            assert valid.shape==ct.shape and valid.sum()==ta['training_pairs']
            assert ((cs==qs[:,None]) & (ct+span<=qt[:,None]))[valid].all()
            for sid in np.unique(qs):
                s=catalog['series'][int(sid)]; assert ((qt[qs==sid]>=s['memory_end']) & (qt[qs==sid]+span<=s['train_end'])).all()
            overlap=(cs==qs[:,None]) & (ct<qt[:,None]+span) & (qt[:,None]<ct+span)
            assert not overlap[valid].any()
        with np.load(folder/'index.npz',allow_pickle=False) as archive:
            index={k:archive[k].copy() for k in archive.files}
        assert len(index['start'])==meta['windows']
        assert np.isfinite(index['past']).all() and np.isfinite(index['future']).all()
        store=Store(src/'store')
        try:
            # Reconstruct EVERY indexed past/continuation from raw memory, not a sample.
            for sid in np.unique(index['sid']):
                s=store.series[int(sid)]; ids=np.flatnonzero(index['sid']==sid); starts=index['start'][ids]
                assert (starts>=0).all() and (starts+span<=s['memory_end']).all()
                assert len(np.unique(starts))==len(starts)
                view=np.lib.stride_tricks.sliding_window_view(store.values(sid),span)
                for a in range(0,len(ids),1024):
                    chosen=ids[a:a+1024]; raw=np.asarray(view[index['start'][chosen]],dtype=np.float64)
                    x=raw[:,:c['length']]; center=x.mean(-1); scale=np.maximum(x.std(-1),max(s['memory_std']*c['normalization']['memory_std_floor'],1e-6))
                    np.testing.assert_allclose(index['past'][chosen],(x-center[:,None])/scale[:,None],rtol=1e-4,atol=2e-5)
                    np.testing.assert_allclose(index['future'][chosen],(raw[:,c['length']:]-center[:,None])/scale[:,None],rtol=1e-4,atol=2e-5)
        finally: store.close()
        with np.load(folder/'test.npz',allow_pickle=False) as archive:
            test={k:archive[k].copy() for k in archive.files}
        forecasts=read(folder/'predictions.json')
        metrics=pd.read_json(folder/'metrics.jsonl',lines=True)
        keys=['query','sid','start','horizon']; assert not metrics.duplicated(keys+['method']).any()
        expected=read(src/'test/run.json'); assert expected['data_id']==catalog['data_id']
        old=pd.read_json(src/'test/metrics.jsonl',lines=True)
        identities=set(map(tuple,old[old.method=='a_learned'][keys].to_numpy()))
        score_lookup={(r['query'],r['method'],r['horizon']):r for r in metrics.to_dict('records')}
        checked=0
        for i,r in enumerate(forecasts):
            assert r['query']==i and (r['sid'],r['start'])==(int(test['sid'][i]),int(test['start'][i]))
            s=catalog['series'][r['sid']]; assert s['validation_end']<=r['start'] and r['start']+span<=s['n']
            variance=max(s['memory_std'],1e-6)**2
            for method,prediction in r['predictions'].items():
                error=np.asarray(prediction,dtype=float)-test['y'][i].astype(float)
                for h in c['horizons']:
                    row=score_lookup[(i,method,h)]
                    mse=float(np.dot(error[:h],error[:h])/h); mae=sum(abs(float(e)) for e in error[:h])/h
                    np.testing.assert_allclose([row['mse'],row['mae'],row['nmse']],[mse,mae,mse/variance],rtol=1e-10,atol=1e-10)
                    checked+=1
        assert checked==len(metrics)
        for method,part in metrics.groupby('method'):
            assert set(map(tuple,part[keys].to_numpy()))==identities
        means=metrics.groupby('method').nmse.mean().to_dict()
        means['previous_a_learned']=float(old[old.method=='a_learned'].nmse.mean())
        previous_path=Path('v7/runs/a50_rerank_20261008')/name/'test'
        if previous_path.exists():
            previous=pd.read_json(previous_path/'metrics.jsonl',lines=True)
            selected=previous[previous.method=='blend_0.2']
            assert read(previous_path/'run.json')['data_id']==catalog['data_id']
            assert set(map(tuple,selected[keys].to_numpy()))==identities
            means['previous_a50_moirai_rerank']=float(selected.nmse.mean())
        results.append(dict(dataset=name,queries=len(forecasts),**means))
        retrieval=[json.loads(line) for line in (folder/'retrieval.jsonl').read_text().splitlines()]
        index_keys={(int(s),int(p)):i for i,(s,p) in enumerate(zip(index['sid'],index['start']))}
        for r in retrieval:
            qi=r['query']; x=test['x'][qi]; y=test['y'][qi]
            center=float(x.mean()); scale=max(float(x.std()),float(test['floor'][qi]))
            yn=(y-center)/scale
            for method,hits in r['methods'].items():
                positions=np.asarray(hits['starts']); assert len(positions)==5
                assert eligible([r['sid']],[r['start']],[r['sid']]*5,positions,span).all()
                ids=[index_keys[(r['sid'],int(p))] for p in positions]
                mismatch=np.mean([np.mean((index['future'][ids,:h]-yn[:h])**2,axis=-1) for h in c['horizons']],axis=0)
                np.testing.assert_allclose([hits['candidate_future_nmse_mean'],hits['candidate_future_nmse_top1']],
                    [mismatch.mean(),mismatch[0]],rtol=1e-6,atol=1e-6)
                prediction=np.asarray(hits['weights'])@index['future'][ids]*scale+center
                np.testing.assert_allclose(prediction,forecasts[qi]['predictions'][method],rtol=1e-5,atol=2e-5)
        for method in retrieval[0]['methods']:
            quality.append(dict(dataset=name,method=method,
                future_mismatch=float(np.mean([r['methods'][method]['candidate_future_nmse_mean'] for r in retrieval])),
                history_mismatch=float(np.mean([r['methods'][method]['candidate_history_nmse_mean'] for r in retrieval]))))
        for variant in ['scratch','cluster']:
            run=read(folder/variant/'run.json'); history=read(folder/variant/'history.json')
            assert run['complete'] and run['data_id']==catalog['data_id']
            assert sha256(folder/variant/'best.pt')==run['checkpoint_sha256']==meta['checkpoints'][variant]
            best=min(history,key=lambda r:r['validation_nmse'])
            assert (best['epoch'],best['validation_nmse'])==(run['best_epoch'],run['best_validation'])
            training.append(dict(dataset=name,variant=variant,training_windows=run['training_windows'],
                epochs=run['epochs'],best_epoch=run['best_epoch'],best_validation=run['best_validation'],parameters=run['trainable_parameters']))
        audits.append(dict(dataset=name,training_pairs=ta['training_pairs'],overlap_pairs=0,chronology_violations=0,
            indexed_windows=meta['windows'],all_index_values_reconstructed=True,metric_rows_recomputed=checked,
            cluster_fit_split='train_only'))
        sid=next(s['sid'] for s in catalog['series'] if s['column']=='OT')
        r=min((r for r in forecasts if r['sid']==sid),key=lambda r:r['start']); qi=r['query']
        examples.append(dict(dataset=name,history=test['x'][qi].tolist(),future=test['y'][qi].tolist(),predictions=r['predictions']))
    summary=dict(complete=True,results=results,retrieval_quality=quality,training=training,audits=audits,
        caveats=['Single seed and 28 test queries per ETT subset; custom chronological splits.',
            'Scratch and cluster controls use the same dense training windows and 20-epoch budget; earlier A used fewer windows and epochs.',
            'Candidate-candidate overlap is permitted; every query-candidate pair is nonoverlapping and strictly historical.',
            'True test futures only measure retrieval quality and forecast errors, never ranking or training.'])
    write_json(root/'summary.json',summary); pd.DataFrame(results).to_csv(root/'dataset_metrics.csv',index=False)
    pd.DataFrame(quality).to_csv(root/'retrieval_quality.csv',index=False)
    write_json(root/'forecast_examples.json',examples)
    lines=['# 自训练未来聚类倾向编码器：ETT 对照','',
        '完全移除 Moirai 编码器。随机初始化的 PatchTST 历史编码器接受未来邻域监督；聚类版本额外学习训练未来所属的软走势簇。预测与检索只输入当前历史。','',
        '## 预测误差：NMSE 越低越好','',
        '|数据集|历史相似检索|自训练向量|聚类监督向量|全库未来倾向|历史50＋未来簇|历史50＋连续未来|原 A50＋Moirai 重排|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        labels=['history_only','scratch_embedding','cluster_embedding','future_cluster','history_future_cluster',
            'history_future_regression','previous_a50_moirai_rerank']
        lines.append('|'+r['dataset']+'|'+'|'.join(f'{r[k]:.6f}' if k in r else '—' for k in labels)+'|')
    lines+=['','## 取回候选的未来是否相似','',
        '候选未来误差用查询历史尺度标准化，先对 24、96、244 三个长度等权，再对取回的 5 个候选和查询平均。该指标与上表的记忆库方差 NMSE 分母不同。真实测试未来仅用于这里的事后诊断。','',
        '|数据集|方案|候选未来误差|候选历史误差|','|---|---|---:|---:|']
    for r in quality: lines.append(f"|{r['dataset']}|{r['method']}|{r['future_mismatch']:.6f}|{r['history_mismatch']:.6f}|")
    lines+=['','## 机制与训练','',
        '- 仅用训练窗口的已知未来拟合 16 个 k-means++ 走势原型。特征保留多预测长度的完整有序数值和一阶变化，尺度仅由历史计算。',
        '- 编码器同时学习连续检索向量、辅助未来预测和未来簇概率。候选对损失用真实训练未来定义相似目标，历史相似但未来不同的候选会得到较低目标权重。',
        '- 每条训练查询取 32 个历史相似困难候选和 32 个随机候选，全部来自最早记忆库；候选的历史＋后续必须结束于查询开始前。',
        '- 聚类倾向检索用查询预测的簇分布匹配历史案例已发生后续的簇分布。对照全库未来倾向召回与历史形状召回 50 个后再按未来倾向 0.8／历史形状 0.2 重排的策略，均取 5 个；不使用当前真实未来。',
        '- 连续未来重排同样只在历史召回的 50 个中选择，以自训练辅助预测头输出的完整未来数值匹配候选后续，评分权重仍为未来 0.8／历史 0.2。',
        '- 对照模型去掉聚类监督，保留相同 PatchTST 主干、未来邻域损失和预测损失。两个模型均以粗记忆库上的验证集检索预测 NMSE 选最佳轮次；聚类模型按历史50＋未来簇模式选择。',
        '- 最终检索库按 stride 1 构建，训练候选库按 stride 32 构建；记忆库只在最早 25%，验证与测试沿用原来的查询。','',
        '|数据集|方案|训练窗口|训练轮数|选用轮次|参数量|','|---|---|---:|---:|---:|---:|']
    for r in training: lines.append(f"|{r['dataset']}|{r['variant']}|{r['training_windows']}|{r['epochs']}|{r['best_epoch']}|{r['parameters']}|")
    lines+=['','## 核对与限制','',
        '- 所有训练检索对均核对完整区间，无自身或重叠候选；所有选中测试候选均严格早于查询。候选彼此之间可以重叠。',
        '- 所有索引数值从原始记忆库重建核对，全部预测指标从预测值和真实目标独立重算。原始 CSV 哈希未变。',
        '- 聚类版本与自训练对照使用相同密集训练窗口和轮数；旧 A 使用更少窗口和轮数，因此与旧 A 的比较不是训练预算相等的消融。',
        '- 单随机种子，每个 ETT 子集 28 个测试查询，项目自定义划分；簇监督不能保证从相似历史唯一确定未来。','',
        '## 固定预测示例','', '每个子集 OT 变量最早测试窗口，按时间选取。','', '![未来倾向检索对照](forecast_examples.png)']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nrows=(len(examples)+1)//2; fig,axes=plt.subplots(nrows,2,figsize=(13,3.7*nrows),squeeze=False,constrained_layout=True)
    for ax,e in zip(axes.flat,examples):
        ax.plot(np.arange(-96,0),e['history'][-96:],color='#999999',lw=1,label='History')
        ax.plot(e['future'],color='#202020',lw=1.7,label='Actual')
        for key,label,color in [('history_only','History nearest','#8a7a69'),('scratch_embedding','Scratch vector','#3578b0'),
                ('history_future_cluster','History50 + regime','#7e60a5'),('history_future_regression','History50 + continuous future','#d2652d')]:
            ax.plot(e['predictions'][key],lw=1.1,color=color,label=label)
        ax.axvline(0,color='#999999',lw=.8,ls=':'); ax.grid(alpha=.15)
        ax.set(title=e['dataset']+' / OT / first test query',xlabel='Steps from forecast origin',ylabel='Original scale')
    for ax in list(axes.flat)[len(examples):]: ax.set_axis_off()
    handles,names=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,names,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(root/'forecast_examples.png',dpi=160); plt.close(fig)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(results=results,retrieval_quality=quality,audits=audits),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    args=p.parse_args(); report(Path(args.root),Path(args.source))
