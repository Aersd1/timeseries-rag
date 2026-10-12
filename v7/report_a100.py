"""Audit frozen-base A100 reliability improvements against reproduced A50."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v5.data import Store
from .common import sha256,write_json
from .benchmark_a100 import load_cache,rank_cache,validation_tables,select_policy
from .quality_gate import quality_features,gate_predict


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def report(root,source):
    complete=read(root/'complete.json'); policy=read(root/'policy.json')
    assert complete['complete'] and sha256(root/'policy.json')==complete['policy_sha256']
    reproduced=select_policy(validation_tables(root,source,complete['datasets']))
    for k,v in reproduced.items(): assert policy[k]==v
    rows=[]; audits=[]; gates=[]; curves=[]
    for name in complete['datasets']:
        folder=root/name; src=source/name; store=Store(src/'store'); catalog=store.meta
        c=read(src/'config.json'); span=c['length']+max(c['horizons'])
        source_quality=read(src/'quality.json'); assert sha256(source_quality['source'])==source_quality['sha256']
        gate=torch.load(folder/'quality_gate.pt',map_location='cpu',weights_only=True)
        gate_run=read(folder/'gate_run.json')
        assert gate['data_id']==catalog['data_id'] and gate['policy_sha256']==sha256(root/'policy.json')
        assert gate_run['complete'] and gate_run['checkpoint_sha256']==sha256(folder/'quality_gate.pt')
        assert gate['training_cache_sha256']==sha256(folder/'train/cache.npz')
        assert gate['validation_cache_sha256']==sha256(folder/'validation/cache.npz')==policy['validation_hashes'][name]
        gate_history=read(folder/'gate_history.json')
        if gate['best_epoch']:
            best=min(gate_history,key=lambda r:r['validation_nmse']); assert best['epoch']==gate['best_epoch']
        assert gate['adopted']==('gate' if gate['validation_gate_nmse']<gate['validation_constant_nmse'] else 'constant')
        counts={}; checked_candidates=0
        try:
            for split in ['train','validation','test']:
                run,data=load_cache(folder/split); counts[split]=len(data['x'])
                assert run['data_id']==catalog['data_id'] and run['config']==c
                assert run['checkpoint_sha256']==sha256(src/'train/best.pt')==gate['checkpoint_sha256']
                assert run['index_sha256']==sha256(src/'index_a/manifest.json')==gate['index_sha256']
                levels=np.asarray(run['quantile_levels']); median=int(np.flatnonzero(np.isclose(levels,.5))[0])
                for i,x in enumerate(data['x']):
                    sid,start=int(data['sid'][i]),int(data['start'][i]); s=store.series[sid]
                    begin,end={'train':(s['memory_end'],s['train_end']),'validation':(s['train_end'],s['validation_end']),
                        'test':(s['validation_end'],s['n'])}[split]
                    assert begin<=start and start+span<=end
                    np.testing.assert_array_equal(x,store.window(sid,start,c['length']))
                    np.testing.assert_array_equal(data['y'][i],store.window(sid,start+c['length'],max(c['horizons'])))
                    positions=data['starts'][i]; assert len(positions)==100 and len(np.unique(positions))==100
                    assert (positions>=0).all() and (positions+span<=min(start,s['memory_end'])).all()
                    past=np.stack([store.window(sid,int(p),c['length']) for p in positions]).astype(float)
                    future=np.stack([store.window(sid,int(p)+c['length'],max(c['horizons'])) for p in positions]).astype(float)
                    floor=max(s['memory_std']*c['normalization']['memory_std_floor'],1e-6)
                    center=float(x.mean()); scale=max(float(x.std()),floor)
                    mapped=center+scale*(future-past.mean(-1)[:,None])/np.maximum(past.std(-1),floor)[:,None]
                    np.testing.assert_allclose(data['mapped'][i],mapped,rtol=1e-12,atol=1e-12)
                    history=((past-past.mean(-1)[:,None])/np.maximum(past.std(-1),floor)[:,None]
                        -(x.astype(float)-float(x.astype(float).mean()))/max(float(x.astype(float).std()),floor))**2
                    np.testing.assert_allclose(data['history_scores'][i],history.mean(-1),rtol=1e-10,atol=1e-10)
                    quantiles=np.sort((data['quantiles'][i].astype(float)-center)/scale,axis=0)
                    normalized=(mapped-center)/scale
                    residual=normalized[:,None,:]-quantiles[None]
                    loss=np.where(residual>=0,levels[None,:,None]*residual,(levels[None,:,None]-1)*residual)
                    fs=sum(loss[:,:,:h].mean((1,2)) for h in c['horizons'])/len(c['horizons'])
                    np.testing.assert_allclose(data['future_scores'][i],fs,rtol=1e-10,atol=1e-10)
                    np.testing.assert_array_equal(data['direct'][i],data['quantiles'][i,median])
                    checked_candidates+=100
            run,data=load_cache(folder/'test'); predictions=read(folder/'test/predictions.json')
            details=read(folder/'test/retrieval.json'); metrics=pd.read_json(folder/'test/metrics.jsonl',lines=True)
            keys=['query','sid','start','horizon']; assert not metrics.duplicated(keys+['method']).any()
            previous=pd.read_json(Path('v7/runs/a50_rerank_20261008')/name/'test/metrics.jsonl',lines=True)
            old=previous[previous.method=='blend_0.2']; new=metrics[metrics.method=='a50']
            paired=old.merge(new,on=keys,validate='one_to_one',suffixes=('_old','_new'))
            assert len(paired)==len(old)==len(new)
            np.testing.assert_allclose(paired.nmse_old,paired.nmse_new,rtol=1e-5,atol=1e-5)
            means=metrics.groupby('method').nmse.mean().to_dict()
            rows.append(dict(dataset=name,queries=len(predictions),**means,
                adaptive_gain_vs_a50_percent=100*(1-means['a100_adaptive']/means['a50']),
                adaptive_gain_vs_direct_percent=100*(1-means['a100_adaptive']/means['moirai_direct'])))
            metric_lookup={(r['query'],r['method'],r['horizon']):r for r in metrics.to_dict('records')}; metric_count=0
            for i,r in enumerate(predictions):
                assert r['query']==i and (r['sid'],r['start'])==(int(data['sid'][i]),int(data['start'][i]))
                variance=max(catalog['series'][r['sid']]['memory_std'],1e-6)**2
                for method,pred in r['predictions'].items():
                    error=np.asarray(pred,dtype=float)-data['y'][i].astype(float)
                    for h in c['horizons']:
                        saved=metric_lookup[(i,method,h)]; mse=np.dot(error[:h],error[:h])/h; mae=np.abs(error[:h]).sum()/h
                        np.testing.assert_allclose([saved['mse'],saved['mae'],saved['nmse']],[mse,mae,mse/variance],rtol=1e-10,atol=1e-10)
                        metric_count+=1
                for method,k,strength in [('a50',50,0.),('a100',100,0.),('a100_diverse',100,policy['diversity_weight'])]:
                    result=rank_cache(data,run,i,k,strength)
                    assert [h['start'] for h in result['hits']]==[h['start'] for h in details[i]['selected'][method]]
                    np.testing.assert_allclose(r['predictions'][method],result['prediction'],rtol=1e-12,atol=1e-12)
                hits=details[i]['selected']['a100_diverse']; lookup={int(s):j for j,s in enumerate(data['starts'][i])}
                ids=[lookup[h['start']] for h in hits]
                f=quality_features(data['x'][i],data['floor'][i],data['quantiles'][i],run['quantile_levels'],
                    data['mapped'][i,ids],hits,span,catalog['series'][r['sid']]['memory_std'])
                raw,deployed=gate_predict(gate,f[None])
                np.testing.assert_allclose([r['raw_gate_alpha'],r['deployed_alpha']],[raw[0],deployed[0]],rtol=1e-6,atol=1e-6)
                rebuilt=(1-float(deployed[0]))*data['direct'][i]+float(deployed[0])*np.asarray(r['predictions']['a100_diverse'])
                np.testing.assert_allclose(rebuilt,r['predictions']['a100_adaptive'],rtol=1e-12,atol=1e-12)
            assert metric_count==len(metrics)
            alphas=np.asarray([r['deployed_alpha'] for r in predictions])
            gates.append(dict(dataset=name,adopted=gate['adopted'],best_epoch=gate['best_epoch'],constant_alpha=gate['constant_alpha'],
                mean_alpha=float(alphas.mean()),min_alpha=float(alphas.min()),max_alpha=float(alphas.max()),
                validation_gate_nmse=gate['validation_gate_nmse'],validation_constant_nmse=gate['validation_constant_nmse']))
            audits.append(dict(dataset=name,queries=counts,checked_candidates=checked_candidates,query_overlap_violations=0,
                baseline_a50_reproduced=True,raw_values_and_scores_reconstructed=True,metric_rows_recomputed=metric_count))
            sid=next((s['sid'] for s in catalog['series'] if s['column']=='OT'),0)
            chosen=min((r for r in predictions if r['sid']==sid),key=lambda r:r['start']); qi=chosen['query']
            curves.append(dict(dataset=name,history=data['x'][qi].tolist(),future=data['y'][qi].tolist(),predictions=chosen['predictions']))
        finally: store.close()
    summary=dict(complete=True,policy=policy,results=rows,gates=gates,audits=audits,
        caveats=['Single seed, 28 test queries per dataset, custom chronological splits.',
            'A/Moirai weights and index remain frozen; only the 113-parameter reliability gate is trained.',
            'Diversity and mixture grid choices, gate epochs and adoption use validation only; repeated benchmark development is exploratory.',
            'Alpha zero is exactly Moirai-only and alpha one is exactly candidate-only. Alpha is not a calibrated confidence probability.'])
    write_json(root/'summary.json',summary); pd.DataFrame(rows).to_csv(root/'dataset_metrics.csv',index=False)
    write_json(root/'forecast_examples.json',curves)
    lines=['# 原 A＋Moirai：100 候选与可靠性门控','',
        '冻结原 A 学习向量编码器、历史索引及 Moirai 2，召回 100 个，以未来 0.8／历史 0.2 重排取 5 个。新增门控直接学习何时参考候选，以及参考多少。','',
        '## 相同查询的 NMSE','',
        '|数据集|原 A50|仅改 A100|A100＋多样性|验证集固定融合|训练可靠性门控|验证集选用方案|原始 Moirai|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|'+r['dataset']+'|'+'|'.join(f'{r[k]:.6f}' for k in
            ['a50','a100','a100_diverse','a100_calibrated','a100_gate','a100_adaptive','moirai_direct'])+'|')
    lines+=['','## 改进机制','',
        '- 原检索表示与分位数未来评分完整保留。A50 对照使用相同 top100 中的前 50 个，并在该 50 个内重新归一化评分，已与旧 A50 逐查询结果核对。',
        '- 多样性：逐个选择候选，对与已选候选完整历史＋后续区间的重叠施加软惩罚。该惩罚只控制选择，聚合仍用原兼容性分数。',
        f"- 多样性惩罚在验证集从 {policy['diversity_grid']} 选出统一系数 {policy['diversity_weight']}，测试前冻结。", 
        '- 固定融合对照：验证集按数据集选择候选比例 α，预测为 (1−α)×Moirai＋α×五个候选的加权预测。α 与重排的 0.8／0.2 是不同参数。',
        '- 自适应门控：12 个历史／候选质量特征进入 113 参数的小网络，按当前查询动态输出 α。特征包含 Moirai 分位数区间宽度、候选后续分歧、模型与候选差异、历史误差、权重熵及候选重叠。',
        '- 门控用每个数据集 448 个训练查询的真实未来计算预测损失；特征不接受当前未来。均值与标准差只从训练特征拟合，200 轮预算仅按验证 NMSE 选轮次。',
        '- 验证集若未支持采用训练门控，回退到固定融合，测试不决定采用哪一个。', '',
        '|数据集|选用|最佳门控轮次|固定候选比例|测试平均比例|最小|最大|相对 A50 误差降低|相对 Moirai 误差降低|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g,r in zip(gates,rows):
        lines.append(f"|{g['dataset']}|{g['adopted']}|{g['best_epoch']}|{g['constant_alpha']:.2f}|{g['mean_alpha']:.4f}|{g['min_alpha']:.4f}|{g['max_alpha']:.4f}|{r['adaptive_gain_vs_a50_percent']:.2f}%|{r['adaptive_gain_vs_direct_percent']:.2f}%|")
    lines+=['','## 核对和限制','',
        '- 训练、验证、测试候选的完整后续必须结束于查询起点及记忆库边界之前，排除自身／重叠；未放宽该规则。候选彼此可重叠。',
        '- 缓存原始历史、真实目标、候选映射、历史误差和分位数未来分数逐条重建；全部测试预测误差独立重算。原始 CSV 和 A 权重、索引哈希核对不变。',
        '- 验证和测试都使用原 stride 1 索引。训练及密集验证的历史编码与分位数预测按 batch 32 加速；原测试查询使用 batch 1 以复现旧 A50。',
        '- 单随机种子，每个子集 28 个测试窗口，项目自定义划分；多轮研发中重复使用过测试集，结果是探索性证据，仍需独立窗口确认。',
        '- α=0 表示选用原 Moirai 直接预测，此时不能将收益归因于检索；α 不是正确性概率。', '',
        '## 固定预测示例','', '每个子集 OT 最早测试查询。','', '![A100 门控对照](forecast_examples.png)']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots((len(curves)+1)//2,2,figsize=(13,7.7),squeeze=False,constrained_layout=True)
    for ax,e in zip(axes.flat,curves):
        n=min(96,len(e['history'])); ax.plot(np.arange(-n,0),e['history'][-n:],color='#999999',lw=1,label='History')
        ax.plot(e['future'],color='#202020',lw=1.7,label='Actual')
        for key,label,color in [('a50','Original A50','#8a7a69'),('a100','A100 only','#7e60a5'),
                ('moirai_direct','Moirai direct','#3578b0'),('a100_adaptive','Validated reliability','#d2652d')]:
            ax.plot(e['predictions'][key],lw=1.1,color=color,label=label)
        ax.axvline(0,color='#999999',lw=.8,ls=':'); ax.grid(alpha=.15)
        ax.set(title=e['dataset']+' / first OT test query',xlabel='Steps from forecast origin',ylabel='Original scale')
    for ax in list(axes.flat)[len(curves):]: ax.set_axis_off()
    handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(root/'forecast_examples.png',dpi=160); plt.close(fig)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008'); args=p.parse_args()
    report(Path(args.root),Path(args.source))
