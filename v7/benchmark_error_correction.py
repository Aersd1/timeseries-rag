"""Train/validate historical Moirai error transfer on paired96-to96 queries."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import fresh_dir,write_json,sha256
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool
from .error_correction import retrieve,features,innovations,ErrorCorrection,predict_delta

BETAS=[0.,.1,.25,.5,1.]


def prepare(source,queries,root,name,device,memory_scope='prefix',stride=1,value_mode='errors'):
    out=fresh_dir(root/name);previous=queries/name
    manifest=json.loads((previous/'manifest.json').read_text())
    store=Store(source/name/'store');adapter=MoiraiAdapter.pretrained(manifest['moirai'],device)
    if len(adapter.module.quantile_levels)!=9 or not np.allclose(adapter.module.quantile_levels,[.1,.2,.3,.4,.5,.6,.7,.8,.9]):
        raise ValueError('Expected pinned nine quantiles')
    pools={s['sid']:ResidualPool(store,s['sid'],manifest['window'],
        end=s['n'] if memory_scope=='observed' else None,stride=stride) for s in store.series}
    old_errors={};global_errors={};memory_files={}
    try:
        for sid,pool in pools.items():
            if value_mode=='innovations': continue
            std=max(store.series[sid]['memory_std'],1e-6);baselines=[];truth=[]
            for begin in range(0,len(pool.starts),128):
                starts=pool.starts[begin:begin+128]
                raw=np.stack([store.window(sid,int(p),96) for p in starts])
                baselines.extend(adapter.forecast(raw,96)[:,4].cpu().numpy())
                truth.extend([store.window(sid,int(p)+96,96) for p in starts])
            baseline=np.stack(baselines);errors=(np.stack(truth)-baseline)/std
            old_errors[sid]=errors.astype(np.float32);global_errors[sid]=errors.mean(0).astype(np.float32)
            path=out/f'memory_errors_{sid}.npz'
            np.savez_compressed(path,starts=pool.starts,baseline=baseline,errors=old_errors[sid])
            memory_files[str(sid)]=sha256(path)
            print(name,'old model errors',sid,len(pool.starts),flush=True)
        splits={};maximum_baseline_difference=0.
        for split in ['train','validation','test']:
            with np.load(previous/(split+'.npz'),allow_pickle=False) as f: data={k:f[k].copy() for k in f.files}
            arrays={}
            def add(k,v): arrays.setdefault(k,[]).append(v)
            for begin in range(0,len(data['raw_x']),32):
                quantiles=adapter.forecast(data['raw_x'][begin:begin+32],96).cpu().numpy()
                for j,raw_q in enumerate(quantiles,begin):
                    sid,start=int(data['sid'][j]),int(data['start'][j]);pool=pools[sid]
                    q,memory=retrieve(data['raw_x'][j],raw_q,pool,start,manifest['window'])
                    std=np.float32(max(store.series[sid]['memory_std'],1e-6))
                    if value_mode=='innovations':
                        errors,global_value=innovations(data['raw_x'][j],raw_q,pool,memory,std,manifest['window'],start)
                    else:
                        ids=np.searchsorted(pool.starts,memory['selected_starts']);errors=old_errors[sid][ids]
                        global_value=old_errors[sid][pool.starts+192<=min(start,pool.memory_end)].mean(0)
                    vector=features(data['raw_x'][j],raw_q,q,memory,errors,std)
                    base=raw_q[4];difference=float(np.max(abs(base-data['direct'][j])))
                    maximum_baseline_difference=max(maximum_baseline_difference,difference)
                    # Query targets are consumed only after features and selection are fixed.
                    record=dict(features=vector,keys=memory['examples'][:,:96],errors=errors,weights=memory['weights'],
                        sid=np.int64(sid),start=np.int64(start),base=base,raw_x=data['raw_x'][j],raw_y=data['raw_y'][j],
                        target=(data['raw_y'][j]-base)/std,std=std,global_error=global_value,
                        selected_starts=memory['selected_starts'],candidate_starts=memory['candidate_starts'],
                        history_scores=memory['history_scores'],future_scores=memory['future_scores'],raw_quantiles=raw_q)
                    for k,v in record.items():add(k,v)
            path=out/(split+'.npz');np.savez_compressed(path,**{k:np.asarray(v) for k,v in arrays.items()})
            splits[split]=dict(queries=len(data['raw_x']),sha256=sha256(path))
            print(name,'error cache',split,len(data['raw_x']),flush=True)
        write_json(out/'manifest.json',dict(complete=True,dataset=name,splits=splits,window=manifest['window'],
            source_data_id=store.meta['data_id'],model_identity=adapter.identity,moirai=manifest['moirai'],
            seed=manifest['seed'],memory_files=memory_files,baseline_max_abs_difference=maximum_baseline_difference,
            input='Raw96 history into frozen Moirai; only retrieval uses detrended residuals.',
            candidates=100,top_k=5,history_weight=.2,future_weight=.8,memory_scope=memory_scope,
            memory_stride=stride,value_mode=value_mode))
    finally:
        store.close();del adapter;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()


def load(folder,split):
    manifest=json.loads((folder/'manifest.json').read_text());path=folder/(split+'.npz')
    if sha256(path)!=manifest['splits'][split]['sha256']: raise ValueError('Cache changed')
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def train(folder,epochs=80):
    meta=json.loads((folder/'manifest.json').read_text());torch.manual_seed(meta['seed'])
    data={s:load(folder,s) for s in ['train','validation','test']};t,v=data['train'],data['validation']
    mean=t['features'].mean(0);scale=np.maximum(t['features'].std(0),.1)
    tensors={s:{k:torch.as_tensor((d[k]-mean)/scale if k=='features' else d[k])
        for k in ['features','keys','errors','weights','target']} for s,d in data.items()}
    model=ErrorCorrection();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.001)
    def forecast(items):return model(items['features'],items['keys'],items['errors'],items['weights'])
    def val_score():
        model.eval()
        with torch.no_grad():return float((forecast(tensors['validation'])-tensors['validation']['target']).square().mean())
    best=val_score();best_epoch=0;best_state={k:x.detach().clone() for k,x in model.state_dict().items()}
    history=[dict(epoch=0,validation_nmse=best)]
    for epoch in range(1,epochs+1):
        model.train();order=np.random.default_rng(meta['seed']+epoch).permutation(len(t['features']))
        for begin in range(0,len(order),64):
            ids=order[begin:begin+64];batch={k:x[ids] for k,x in tensors['train'].items()}
            optimizer.zero_grad(set_to_none=True);delta=forecast(batch)
            loss=(delta-batch['target']).square().mean()+.01*delta.square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        value=val_score()
        if value<best:best,best_epoch,best_state=value,epoch,{k:x.detach().clone() for k,x in model.state_dict().items()}
        history.append(dict(epoch=epoch,validation_nmse=value))
    weighted={s:(d['weights'][...,None]*d['errors']).sum(1) for s,d in data.items()}
    fixed_values={str(beta):float(np.mean((beta*weighted['validation']-v['target'])**2)) for beta in BETAS}
    beta=min(BETAS,key=lambda b:(fixed_values[str(b)],b));fixed=fixed_values[str(beta)]
    global_values={str(beta):float(np.mean((beta*v['global_error']-v['target'])**2)) for beta in BETAS}
    global_beta=min(BETAS,key=lambda b:(global_values[str(b)],b))
    adopted='learned' if best<fixed else 'fixed'
    state=dict(variant='retrieval_error_correction_96',adopted=adopted,beta=beta,global_beta=global_beta,
        mean=torch.tensor(mean),scale=torch.tensor(scale),model=best_state,epoch=best_epoch,
        validation_nmse=min(best,fixed),source_data_id=meta['source_data_id'],model_identity=meta['model_identity'],
        window=meta['window'],manifest_sha256=sha256(folder/'manifest.json'),
        memory_scope=meta.get('memory_scope','prefix'),memory_stride=meta.get('memory_stride',1),
        value_mode=meta.get('value_mode','errors'))
    torch.save(state,folder/'best.pt');model.load_state_dict(best_state);model.eval()
    test=data['test'];learned_state=dict(state,adopted='learned')
    learned=predict_delta(learned_state,test);deployed=predict_delta(state,test)
    # Break query-to-error pairing only; never use these ablations for model choice.
    shuffled=dict(test);permutation=np.arange(len(test['features']))
    rng=np.random.default_rng(meta['seed']+100)
    for sid in np.unique(test['sid']):
        ids=np.flatnonzero(test['sid']==sid);permutation[ids]=rng.permutation(ids)
    shuffled['errors']=test['errors'][permutation]
    methods=dict(moirai_direct=np.zeros_like(learned),rag_fixed=beta*weighted['test'],rag_learned=learned,
        rag_adopted=deployed,global_error_bias=global_beta*test['global_error'],
        rag_errors_shuffled=predict_delta(state,shuffled))
    np.savez_compressed(folder/'predictions.npz',**{m:test['base']+delta*test['std'][:,None] for m,delta in methods.items()})
    scores={m:float(np.mean((delta-test['target'])**2)) for m,delta in methods.items()}
    write_json(folder/'history.json',history)
    write_json(folder/'run.json',dict(complete=True,dataset=folder.name,adopted=adopted,beta=beta,
        global_beta=global_beta,best_epoch=best_epoch,epochs=epochs,parameters=sum(p.numel() for p in model.parameters()),
        validation_learned=best,validation_fixed=fixed,validation_fixed_grid=fixed_values,validation_global_grid=global_values,
        test_nmse=scores,queries=len(test['features']),train_queries=len(t['features']),validation_queries=len(v['features']),
        gain_vs_direct_percent=100*(1-scores['rag_adopted']/scores['moirai_direct']),
        checkpoint_sha256=sha256(folder/'best.pt'),predictions_sha256=sha256(folder/'predictions.npz')))
    print('ERROR-CORRECTION',folder.name,scores,'adopted',adopted,flush=True)


def report(root,names):
    records=[json.loads((root/n/'run.json').read_text()) for n in names]
    protocol=json.loads((root/'protocol.json').read_text());mode=protocol.get('value_mode','errors')
    write_json(root/'summary.json',dict(complete=True,length=96,horizon=96,results=records,value_mode=mode,
        memory_scope=protocol.get('memory_scope','prefix'),memory_stride=protocol.get('memory_stride',1),
        caveat='Single seed, existing custom chronological splits; test data do not choose coefficients/checkpoints.'))
    lines=['# 96→96：检索历史模型误差的修正实验','',
        '原始96点历史直接输入冻结Moirai。去趋势仅用于检索，不拆开基础模型的预测路径。',
        '对每个历史候选，计算当时Moirai预测与已知后续之间的96点误差。前100按波动历史检索，未来兼容/历史评分0.8/0.2重排选5。',
        '误差适配头以当前历史、基础预测及候选历史为条件，关注五个误差向量并输出96点修正。它在预训练Moirai外部，不是对Moirai本体的内部微调。',
        '修正头的输出层初始化为零；监督目标为训练查询的真实预测误差。验证集选择检查点，以及是否使用学习修正或固定误差迁移系数；固定系数包含0，即保留直接预测。',
        '所有候选完整历史+后续必须结束于查询历史开始前；查询真实未来不进入检索、重排或特征。','',
        '|子集|查询|直接Moirai|普通历史误差校准|检索误差固定修正|检索误差学习修正|最终采用|相对直接改善|打乱检索对应|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        s=r['test_nmse'];vals=[s[m] for m in ['moirai_direct','global_error_bias','rag_fixed','rag_learned','rag_adopted']]
        lines.append('|'+r['dataset']+'|'+str(r['queries'])+'|'+'|'.join(f'{v:.6f}' for v in vals)+f"|{r['gain_vs_direct_percent']:.2f}%|{s['rag_errors_shuffled']:.6f}|")
    lines+=['','指标为按历史库变量方差归一化的NMSE，越低越好。正改善表示超过直接预测；负改善表示退化。',
        '普通校准只用整个历史库的平均模型误差，不进行查询相似检索。如果它与检索修正一样好，则不能把收益归因于案例匹配。',
        '打乱对照将误差向量移交给同变量的其他查询；它只用于事后诊断，不影响采用策略。',
        '保留直接预测并不等于检索有效；需要检索方案同时超过直接模型和普通校准，并在破坏匹配时有明显损失。',
        '每个查询的原始预测、目标与修正保存在本地缓存，可独立复算。不能保证对所有未来样本优于基础模型。']
    if mode=='innovations':
        lines[0]='# 96→96：在线检索的波动修正实验'
        lines[2]='原始96点历史直接输入冻结Moirai。对其预测轨迹做同一个因果趋势投影，保留预测趋势；检索仅为预测波动提供修正建议。'
        lines[3]='候选值是“已知旧案例波动后续−当前Moirai预测波动”的96点差值，经历史尺度映射。这里不使用旧案例的模型预测误差。前100按波动历史检索，0.8/0.2重排选5。'
        lines.append('')
        lines.append('本实验的global_error_bias列表示：不做相似检索，将全部合格历史案例的平均归一化波动后续作为修正对照。它不是旧模型误差校准。')
    if protocol.get('memory_scope')=='observed':
        lines += ['','在线协议：检索库随时间扩展，只允许完整历史+后续结束于查询历史开始之前的案例。',
            '先前验证/测试窗口的真实值在已经观测后可以成为后续查询的历史案例；当前查询及之后的所有值严格排除。这与只允许训练库的离线协议不同。',
            f"索引步长固定为{protocol['memory_stride']}；候选数量仍为100。时间范围与索引步长共同改变，未单独隔离两者贡献。",
            '当前方法是在探索性开发过程中新增的对照，同一测试区间已用于此前方案评估，不能视作全新外部盲测。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--queries',default='v7/runs/trend_rag_96_20261013');p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',default=['ETTh1','ETTh2','ETTm1','ETTm2']);p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--memory-scope',choices=['prefix','observed'],default='prefix')
    p.add_argument('--memory-stride',type=int,default=1)
    p.add_argument('--value-mode',choices=['errors','innovations'],default='errors')
    args=p.parse_args();torch.set_num_threads(4)
    if args.epochs<1 or args.memory_stride<1:raise ValueError('Positive epochs/stride required')
    root=fresh_dir(args.output);source,queries=Path(args.source),Path(args.queries)
    write_json(root/'protocol.json',dict(length=96,horizon=96,epochs=args.epochs,betas=BETAS,
        history_weight=.2,future_weight=.8,candidates=100,top_k=5,query_input='original history',
        mechanism='known historical foundation-model error profiles, external learned correction head',
        source_queries=str(queries),memory_scope=args.memory_scope,memory_stride=args.memory_stride,value_mode=args.value_mode,
        code_sha256={f:sha256(f) for f in ['v7/error_correction.py','v7/benchmark_error_correction.py']}))
    for name in args.datasets:
        prepare(source,queries,root,name,args.device,args.memory_scope,args.memory_stride,args.value_mode);train(root/name,args.epochs)
    report(root,args.datasets)


if __name__=='__main__':main()
