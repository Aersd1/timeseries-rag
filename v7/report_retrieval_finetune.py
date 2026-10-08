"""Audit chronological retrieval memory and matched fine-tuned Moirai predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report(root, source):
    complete = read(root/'complete.json')
    if not complete['complete']:
        raise ValueError('Incomplete fine-tuning experiment')
    results, training, audits, comparisons, horizons = [], [], [], [], []
    all_curves = []
    for name in complete['datasets']:
        folder = root/name
        memory_meta = read(folder/'memory/manifest.json')
        c = memory_meta['config']
        catalog = read(source/name/'store/catalog.json')
        assert memory_meta['complete'] and catalog['data_id']==memory_meta['data_id']
        assert memory_meta['history_weight']==.2 and memory_meta['future_weight']==.8
        assert memory_meta['checkpoint_sha256']==sha256(source/name/'train/best.pt')
        assert memory_meta['index_sha256']==sha256(source/name/'index_a/manifest.json')
        quality = read(source/name/'quality.json')
        assert sha256(quality['source'])==quality['sha256']
        raw_series={s['sid']:np.memmap(source/name/'store'/s['file'],dtype='<f4',mode='r') for s in catalog['series']}
        full = {}
        for split in ['train','validation','test']:
            path = folder/'memory'/(split+'.npz')
            assert sha256(path)==memory_meta['splits'][split]['sha256']
            with np.load(path, allow_pickle=False) as archive:
                memory={key:np.array(archive[key],copy=True) for key in archive.files}
                assert np.isfinite(memory['examples']).all() and np.isfinite(memory['y']).all()
                records = [json.loads(line) for line in (folder/'memory'/(split+'_retrieval.jsonl')).read_text().splitlines()]
                assert len(records)==len(memory['x'])
                for i,r in enumerate(records):
                    assert r['query']==i and r['sid']==memory['sid'][i] and r['start']==memory['start'][i]
                    series = catalog['series'][r['sid']]
                    begin,end = {'train':(series['memory_end'],series['train_end']),
                        'validation':(series['train_end'],series['validation_end']),
                        'test':(series['validation_end'],series['n'])}[split]
                    assert begin <= r['start'] and r['start']+c['length']+max(c['horizons']) <= end
                    raw=raw_series[r['sid']]
                    length,horizon=c['length'],max(c['horizons'])
                    np.testing.assert_array_equal(memory['x'][i],raw[r['start']:r['start']+length])
                    np.testing.assert_array_equal(memory['y'][i],raw[r['start']+length:r['start']+length+horizon])
                    for j,hit in enumerate(r['hits']):
                        assert hit['sid']==r['sid']
                        assert hit['start']+c['length']+max(c['horizons']) <= min(r['start'],series['memory_end'])
                        past=np.asarray(raw[hit['start']:hit['start']+length],dtype=float)
                        both=np.asarray(raw[hit['start']:hit['start']+length+horizon],dtype=float)
                        floor=max(series['memory_std']*c['normalization']['memory_std_floor'],1e-6)
                        expected=((both-past.mean())/max(float(past.std()),floor)).astype(np.float32)
                        np.testing.assert_array_equal(memory['examples'][i,j],expected)
                    assert len(r['hits'])==int(memory['valid'][i].sum())
                    if r['hits']:
                        assert np.isclose(memory['weights'][i].sum(),1.)
                full[split] = dict(windows=len(records), full_top5=int(memory['valid'].all(1).sum()),
                    full_top50=sum(r['returned_candidates']==50 for r in records))
        keys = ['query','sid','start','horizon']
        previous = pd.read_json(source/name/'test/metrics.jsonl', lines=True)
        reference = previous[previous.method=='moirai_direct']
        tables = []
        curves = {}
        for variant in ['plain','rag']:
            run = read(folder/variant/'run.json')
            assert run['complete'] and run['config']==c and run['data_id']==catalog['data_id']
            assert run['memory_sha256']==sha256(folder/'memory/manifest.json')
            assert run['checkpoint_sha256']==sha256(folder/variant/'best.pt')
            history = read(folder/variant/'history.json')
            best = min(history,key=lambda h:h['validation_nmse'])
            assert best['epoch']==run['best_epoch'] and np.isclose(best['validation_nmse'],run['best_validation'])
            state = torch.load(folder/variant/'best.pt', map_location='cpu', weights_only=True)
            updated = {key:float(value.norm()) for key,value in state['adaptation'].items()
                if '.b.weight' in key or key=='retrieval.output.weight'}
            training.append(dict(dataset=name,variant=variant,best_epoch=run['best_epoch'],
                epochs=run['epochs_limit'], trainable_parameters=run['trainable_parameters'],
                best_validation=run['best_validation'], initial_validation=history[0]['validation_nmse'],
                total_seconds=run['total_seconds'], update_norms=updated))
            metrics = pd.read_json(folder/variant/'metrics.jsonl', lines=True)
            assert np.isfinite(metrics[['mse','mae','nmse']]).all().all()
            assert not metrics.duplicated(keys+['method']).any()
            expected = set(map(tuple,reference[keys].to_numpy()))
            for _,part in metrics.groupby('method'):
                assert set(map(tuple,part[keys].to_numpy()))==expected
            matched = metrics[metrics.method=='frozen'].merge(reference,on=keys,validate='one_to_one')
            np.testing.assert_allclose(matched.nmse_x,matched.nmse_y,rtol=1e-5,atol=1e-5)
            selected = metrics[metrics.method=='finetuned'].copy()
            selected['method'] = 'moirai_lora' if variant=='plain' else 'moirai_rag_lora'
            tables.append(selected)
            if variant=='rag':
                disabled=metrics[metrics.method=='memory_disabled'].copy()
                disabled['method']='rag_memory_disabled'
                tables.append(disabled)
                tables.append(metrics[metrics.method=='analog'].assign(method='a50_analog'))
            curves[variant] = read(folder/variant/'predictions.json')
        data = pd.concat([reference.assign(method='moirai_frozen'), *tables,
            previous[previous.method=='a_learned']],ignore_index=True)
        # Add the earlier standalone PatchTST only when query identities match.
        direct_path = Path('v7/runs/patchtst_direct_20261008')/name/'test/metrics.jsonl'
        direct_run = read(direct_path.parent/'run.json') if direct_path.exists() else None
        if direct_run and direct_run['data_id']==memory_meta['data_id'] and direct_run['config']==c:
            patch = pd.read_json(direct_path,lines=True)
            patch = patch[patch.method=='patchtst_direct']
            assert set(map(tuple,patch[keys].to_numpy()))==set(map(tuple,reference[keys].to_numpy()))
            data = pd.concat([data,patch],ignore_index=True)
        means = data.groupby('method').nmse.mean().to_dict()
        results.append(dict(dataset=name,queries=len(reference)//len(c['horizons']),**means))
        pivot = data.pivot(index=keys,columns='method',values='nmse')
        for baseline in ['moirai_frozen','moirai_lora','a50_analog','rag_memory_disabled']:
            delta = pivot[baseline]-pivot.moirai_rag_lora
            series = delta.groupby('sid').mean()
            comparisons.append(dict(dataset=name,baseline=baseline,method='moirai_rag_lora',
                mean_nmse_reduction_percent=100*(1-means['moirai_rag_lora']/means[baseline]),
                query_horizon_win_rate=float((delta>0).mean()),series_mean_win_rate=float((series>0).mean())))
        for (h,method),part in data.groupby(['horizon','method']):
            horizons.append(dict(dataset=name,horizon=int(h),method=method,nmse=float(part.nmse.mean())))
        change = max(float(np.max(np.abs(np.array(r['prediction'])-np.array(r['memory_disabled'])))) for r in curves['rag'])
        audits.append(dict(dataset=name,splits=full,candidate_boundary_violations=0,original_csv_unchanged=True,
            memory_values_reconstructed=True,frozen_predictions_reproduced=True,max_abs_memory_effect=change))
        sid = next((s['sid'] for s in catalog['series'] if s['column']=='OT'),catalog['series'][0]['sid'])
        chosen = min((r for r in curves['rag'] if r['sid']==sid),key=lambda r:r['start'])
        with np.load(folder/'memory/test.npz',allow_pickle=False) as memory:
            qi = chosen['query']
            all_curves.append(dict(dataset=name,column=catalog['series'][sid]['column'],query=qi,sid=sid,start=chosen['start'],
                history=memory['x'][qi].tolist(),future=memory['y'][qi].tolist(),
                frozen=memory['frozen_prediction'][qi].tolist(),
                lora=curves['plain'][qi]['prediction'],rag=chosen['prediction']))
    summary=dict(complete=True,results=results,training=training,audits=audits,comparisons=comparisons,
        caveats=['Single seed, at most four test windows per variable; custom chronological splits.',
            'Same training windows, learning rate, epochs and batch order for plain/RAG LoRA, but RAG adds attention parameters.',
            'Epoch zero is an eligible pretrained checkpoint; test data selects no model or hyperparameter.',
            'Frozen A retriever was supervised on the training split; candidate data is restricted to earlier memory.'])
    write_json(root/'summary.json',summary)
    write_json(root/'forecast_examples.json',all_curves)
    pd.DataFrame(results).to_csv(root/'dataset_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(root/'paired_comparisons.csv',index=False)
    pd.DataFrame(horizons).to_csv(root/'horizon_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nrows=(len(all_curves)+1)//2
    fig,axes=plt.subplots(nrows,2,figsize=(13,3.5*nrows),constrained_layout=True)
    for ax,example in zip(axes.flat,all_curves):
        past=min(96,len(example['history']))
        ax.plot(np.arange(-past,0),example['history'][-past:],color='#929aa5',lw=1,label='Observed history')
        ax.plot(example['future'],color='#1d2530',lw=1.8,label='Actual future')
        for key,label,color in [('frozen','Frozen Moirai','#8a7a69'),('lora','Moirai + LoRA','#3578b0'),('rag','Moirai + LoRA + memory','#d2652d')]:
            ax.plot(example[key],color=color,lw=1.2,label=label)
        ax.axvline(0,color='#929aa5',lw=.8,linestyle=':')
        ax.set(title=f"{example['dataset']} / {example['column']} / first test query",xlabel='Steps from forecast origin',ylabel='Original scale')
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(all_curves):]:
        ax.set_axis_off()
    handles,labels=list(axes.flat)[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('Fixed examples selected by time, not by forecast error')
    fig.savefig(root/'forecast_examples.png',dpi=160)
    plt.close(fig)
    lines=['# Moirai 2：微调与检索辅助微调的 ETT 对照','',
        'A 学习向量召回 50 个候选，以未来 0.8、历史 0.2 的固定评分重排取 5 个。候选历史及已知后续走势以外部记忆输入 Moirai 的可训练注意力模块。模型自行输出未来预测，不以候选加权平均替代模型。','',
        '## 相同测试窗口的 NMSE','',
        '每个查询 MSE 除以该变量早期记忆库方差，再对查询和预测长度等权平均。越低越好。','',
        '|数据集|原始 Moirai|仅 LoRA 微调|LoRA＋候选辅助|同一辅助模型关闭候选|A50 候选加权预测|PatchTST 直接预测|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('|'+r['dataset']+'|'+'|'.join(f'{r[m]:.6f}' if m in r else '—' for m in
            ['moirai_frozen','moirai_lora','moirai_rag_lora','rag_memory_disabled','a50_analog','patchtst_direct'])+'|')
    lines+=['','## 训练与最佳模型','',
        '两种微调均使用每变量最多 64 个训练窗口、8 轮、batch size 32、学习率 0.0001、同一随机种子和批次顺序。损失为查询历史尺度归一化的多预测长度分位数损失。按验证集 NMSE 选择模型，并允许选择微调前的第 0 轮。','',
        '|数据集|方案|可训练参数|选用轮次|初始验证 NMSE|最佳验证 NMSE|', '|---|---|---:|---:|---:|---:|']
    for r in training:
        lines.append(f"|{r['dataset']}|{r['variant']}|{r['trainable_parameters']}|{r['best_epoch']}|{r['initial_validation']:.6f}|{r['best_validation']:.6f}|")
    lines+=['','## 候选信息如何使用','',
        '- 冻结预训练权重，在 Moirai 最后两层注意力的 query/value 投影加入 rank-8 LoRA。辅助版本额外训练候选注意力模块。',
        '- 五个候选各自保留历史与后续走势，并用各自历史统计量归一化。候选未来不是当前查询的真实未来。',
        '- 候选分成 patch，附加位置和历史/未来标记；Moirai 的隐藏表示通过注意力读取这些 token，候选评分权重作为注意力先验。',
        '- 辅助模块输出以残差方式进入 Moirai 的预测头；初始化输出为零，微调前与原 Moirai 一致。',
        '- 召回器、重排模型和 0.8/0.2 权重始终冻结。训练、验证、测试的候选均限制在最早 25% 记忆库内。','',
        '## 验证与限制','',
        '- 所有候选已核对其完整未来结束位置，不得越过记忆库边界或查询历史起点。原始 CSV 未改变，原始 Moirai 的逐查询分数已复现。',
        '- 模型 forward 只接受查询历史和历史候选；当前查询真实未来只用于计算训练损失及评价误差。',
        '- 仅 LoRA 与辅助 LoRA 的训练预算相同，但辅助版本的参数量更多，因此不是参数量严格相等的消融。',
        '- 关闭候选是测试诊断，不用于选模型；其结果可能有分布变化，不能单独作为唯一因果证据。',
        '- 单一随机种子，每变量最多 4 个测试窗口；采用项目自定义划分，不是官方论文全量基准。',
        '- 选择第 0 轮表示验证集未支持采用微调更新，应按原始模型理解该结果。', '',
        '## 固定预测示例', '', '各子集 OT 变量最早的测试窗口，没有按效果挑选。总体指标仍使用所有变量和测试窗口。', '',
        '![原始、微调、检索辅助微调的预测曲线](forecast_examples.png)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='v7/runs/moirai_rag_finetune_20261008')
    p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    args=p.parse_args()
    report(Path(args.root),Path(args.source))
