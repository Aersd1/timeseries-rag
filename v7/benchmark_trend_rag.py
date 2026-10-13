"""Paired 96-to-96 Moirai, decomposition, and internal residual-RAG experiment."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from v5.common import valid_ranges, sample_ranges
from .common import fresh_dir, write_json, sha256
from .moirai import MoiraiAdapter
from .trend_rag import decompose, normalize, ResidualPool
from .retrieval_finetune import RetrievalFineTune, quantile_loss

SPLITS = ['train','validation','test']
METHODS = ['moirai_direct','decomposed_frozen','decomposed_plain_ft','decomposed_rag','rag_memory_disabled']


def refs_for(store, split, seed, train_per_series):
    rng = np.random.default_rng(seed+SPLITS.index(split))
    refs = []
    for s in store.series:
        lo,hi = {'train':(s['memory_end'],s['train_end']),
            'validation':(s['train_end'],s['validation_end']),
            'test':(s['validation_end'],s['n'])}[split]
        ranges = list(valid_ranges(s,96,96,lo,hi,32 if split=='train' else 192))
        starts = sample_ranges(ranges,train_per_series,rng) if split=='train' else np.array([p for a,b,st in ranges for p in range(a,b,st)])
        refs.extend((s['sid'],int(p)) for p in starts)
    if not refs: raise ValueError('No complete queries')
    return refs


def prepare(source, folder, name, device, window, count, batch_size):
    cfg = json.loads((source/name/'config.json').read_text(encoding='utf-8'))
    store = Store(source/name/'store')
    adapter = MoiraiAdapter.pretrained(cfg['moirai'],device)
    median = list(adapter.module.quantile_levels).index(.5)
    catalog = store.meta
    write_json(folder/'catalog.json',catalog)
    manifest = dict(complete=False,dataset=name,window=window,length=96,horizon=96,seed=cfg['seed'],
        model_identity=adapter.identity,moirai=cfg['moirai'],source_data_id=catalog['data_id'],
        splits={},retrieval='exact normalized residual-history NMSE, same variable, memory prefix only',
        candidates=100,top_k=5,future_weight=.8,history_weight=.2,diversity_weight=1.,
        quantile_levels=list(adapter.module.quantile_levels))
    try:
        pools = {s['sid']:ResidualPool(store,s['sid'],window) for s in store.series}
        for split in SPLITS:
            refs = refs_for(store,split,cfg['seed'],count)
            arrays = {}
            def add(key,value): arrays.setdefault(key,[]).append(value)
            for offset in range(0,len(refs),batch_size):
                current = refs[offset:offset+batch_size]
                x = np.stack([store.window(sid,p,96) for sid,p in current])
                trends,residuals = decompose(x,window)
                floors = np.array([pools[sid].floor for sid,p in current],dtype=np.float32)[:,None]
                normalized,means,scales = normalize(residuals,floors)
                # All three model calls and retrieval use history only.
                direct = adapter.forecast(x,96)[:,median].cpu().numpy()
                trend_forecast = adapter.forecast(trends,96)[:,median].cpu().numpy()
                residual_quantiles = adapter.forecast(normalized,96).cpu().numpy()
                results = [pools[sid].retrieve(normalized[j],p,residual_quantiles[j],adapter.module.quantile_levels)
                    for j,(sid,p) in enumerate(current)]
                # Query futures first read after all input features and retrieval choices exist.
                y = np.stack([store.window(sid,p+96,96) for sid,p in current])
                full_trend,full_residual = decompose(np.concatenate([x,y],axis=1),window)
                if not np.array_equal(full_trend[:,:96],trends): raise AssertionError('Future changed historical trend')
                for j,(sid,p) in enumerate(current):
                    values = dict(x=normalized[j],raw_x=x[j],raw_y=y[j],trend=trends[j],residual=residuals[j],
                        y=((full_residual[j,96:]-means[j])/scales[j]).astype(np.float32),
                        residual_mean=means[j,0],residual_scale=scales[j,0],floor=np.float32(.1),
                        direct=direct[j],trend_forecast=trend_forecast[j],residual_quantiles=residual_quantiles[j],
                        frozen=trend_forecast[j]+residual_quantiles[j,median]*scales[j,0]+means[j,0],
                        sid=np.int64(sid),start=np.int64(p),**results[j])
                    for k,v in values.items(): add(k,v)
                print(name,'prepare',split,min(offset+batch_size,len(refs)),'/',len(refs),flush=True)
            path = folder/(split+'.npz')
            np.savez_compressed(path,**{k:np.asarray(v) for k,v in arrays.items()})
            manifest['splits'][split] = dict(queries=len(refs),sha256=sha256(path))
        manifest['complete'] = True
        write_json(folder/'manifest.json',manifest)
    finally:
        store.close()
        del adapter
        gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()


def load_split(folder, split):
    manifest = json.loads((folder/'manifest.json').read_text())
    path = folder/(split+'.npz')
    if not manifest['complete'] or sha256(path)!=manifest['splits'][split]['sha256']:
        raise ValueError('Incomplete or changed cache')
    with np.load(path,allow_pickle=False) as data: return {k:data[k].copy() for k in data.files}


def batches(data,batch_size,device,order=None):
    ids = np.arange(len(data['x'])) if order is None else order
    keys = ['x','y','examples','weights','valid','floor','sid','raw_y','trend_forecast','residual_mean','residual_scale']
    for i in range(0,len(ids),batch_size):
        indices = ids[i:i+batch_size]
        yield indices,{k:torch.as_tensor(data[k][indices],device=device) for k in keys}


def predict(model,batch,use_memory):
    return model(batch['x'],batch['examples'],batch['weights'],batch['valid']) if use_memory else model(batch['x'])


def recompose(residual,batch):
    return batch['trend_forecast']+residual*batch['residual_scale'][:,None]+batch['residual_mean'][:,None]


def score(model,data,stds,batch_size,device,use_memory):
    median = list(model.module.quantile_levels).index(.5)
    model.eval(); errors = []
    with torch.no_grad():
        for _,batch in batches(data,batch_size,device):
            forecast = recompose(predict(model,batch,use_memory)[:,median],batch)
            error = (forecast.double()-batch['raw_y'].double()).square().mean(-1)/stds[batch['sid']].square()
            errors.extend(error.cpu().tolist())
    return float(np.mean(errors))


def train_variant(folder,variant,device,epochs,batch_size):
    manifest = json.loads((folder/'manifest.json').read_text())
    catalog = json.loads((folder/'catalog.json').read_text())
    seed = manifest['seed']; torch.manual_seed(seed)
    use_memory = variant=='rag'
    adapter = MoiraiAdapter.pretrained(manifest['moirai'],device)
    model = RetrievalFineTune(adapter,96,96,use_retrieval=use_memory,fusion='paired',
        candidate_dropout=.1 if use_memory else 0.,memory_dropout=.1 if use_memory else 0.).to(device)
    train,val = load_split(folder,'train'),load_split(folder,'validation')
    stds = torch.tensor([max(s['memory_std'],1e-6) for s in catalog['series']],device=device,dtype=torch.float64)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    out = fresh_dir(folder/variant); history=[]; start_time=time.perf_counter()
    best = score(model,val,stds,batch_size,device,use_memory); best_epoch = 0
    best_state = model.adaptation_state()
    history.append(dict(epoch=0,validation_nmse=best,train_loss=None))
    for epoch in range(1,epochs+1):
        began = time.perf_counter(); model.train(); total=0.; count=0
        order = np.random.default_rng(seed+epoch).permutation(len(train['x']))
        for _,batch in batches(train,batch_size,device,order):
            optimizer.zero_grad(set_to_none=True)
            with torch.autograd.graph.allow_mutation_on_saved_tensors():
                quantiles = predict(model,batch,use_memory)
                loss = quantile_loss(quantiles,batch['y'],batch['x'],batch['floor'],model.module.quantile_levels,[96])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True); optimizer.step()
            total += float(loss.detach())*len(batch['x']); count += len(batch['x'])
        value = score(model,val,stds,batch_size,device,use_memory)
        if value<best: best,best_epoch,best_state=value,epoch,model.adaptation_state()
        history.append(dict(epoch=epoch,validation_nmse=value,train_loss=total/count,seconds=time.perf_counter()-began))
        write_json(out/'history.json',history)
        print(folder.name,variant,history[-1],flush=True)
    # Checkpoint chosen on validation before test predictions are computed.
    state = dict(variant='trend_residual_rag_96',adaptation=best_state,architecture=model.architecture,
        identity=model.identity,use_retrieval=use_memory,fusion='paired',rank=model.rank,window=manifest['window'],
        length=96,horizon=96,epoch=best_epoch,validation_nmse=best,manifest_sha256=sha256(folder/'manifest.json'),
        source_data_id=manifest['source_data_id'],seed=seed)
    torch.save(state,out/'best.pt'); model.load_adaptation(best_state); model.eval()
    test = load_split(folder,'test'); median = list(model.module.quantile_levels).index(.5)
    predictions = []; without = []; gates=[]
    with torch.no_grad():
        for _,batch in batches(test,batch_size,device):
            predicted = predict(model,batch,use_memory)[:,median]
            predictions.extend(recompose(predicted,batch).cpu().numpy())
            gate = getattr(model.retrieval,'last_gate',None)
            gates.extend(gate.reshape(len(predicted),-1).mean(-1).cpu().tolist() if gate is not None else [0.]*len(predicted))
            if use_memory:
                without.extend(recompose(predict(model,batch,False)[:,median],batch).cpu().numpy())
    np.savez_compressed(out/'predictions.npz',prediction=np.stack(predictions),
        memory_disabled=np.stack(without) if without else np.stack(predictions),gate_mean=np.array(gates))
    write_json(out/'run.json',dict(complete=True,variant=variant,best_epoch=best_epoch,validation_nmse=best,
        train_queries=len(train['x']),validation_queries=len(val['x']),test_queries=len(test['x']),
        trainable_parameters=sum(p.numel() for p in parameters),epochs=epochs,seed=seed,learning_rate=1e-4,
        loss='Residual quantile pinball, horizon96; validation selection on recomposed raw NMSE',
        seconds=time.perf_counter()-start_time,checkpoint_sha256=sha256(out/'best.pt'),
        predictions_sha256=sha256(out/'predictions.npz'),memory_input='Old residual histories are attention keys; their continuations are values.' if use_memory else None))
    del model,adapter,optimizer
    gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()


def summarize(root,names):
    records=[]; details=[]
    for name in names:
        folder=root/name; data=load_split(folder,'test'); catalog=json.loads((folder/'catalog.json').read_text())
        plain=np.load(folder/'plain/predictions.npz',allow_pickle=False)
        rag=np.load(folder/'rag/predictions.npz',allow_pickle=False)
        forecasts=dict(moirai_direct=data['direct'],decomposed_frozen=data['frozen'],
            decomposed_plain_ft=plain['prediction'],decomposed_rag=rag['prediction'],rag_memory_disabled=rag['memory_disabled'])
        row=dict(dataset=name,queries=len(data['x']),variables=len(catalog['series']))
        for method,pred in forecasts.items():
            error=pred.astype(float)-data['raw_y'].astype(float)
            mse=np.mean(error**2,axis=-1); mae=np.mean(np.abs(error),axis=-1)
            variance=np.array([max(catalog['series'][sid]['memory_std'],1e-6)**2 for sid in data['sid']])
            row[method]=dict(nmse=float(np.mean(mse/variance)),mse=float(np.mean(mse)),mae=float(np.mean(mae)))
            for i in range(len(mse)):
                records.append(dict(dataset=name,query=i,sid=int(data['sid'][i]),start=int(data['start'][i]),
                    method=method,mse=float(mse[i]),mae=float(mae[i]),nmse=float(mse[i]/variance[i])))
        row['rag_gain_vs_direct_percent']=100*(1-row['decomposed_rag']['nmse']/row['moirai_direct']['nmse'])
        row['runs']={v:json.loads((folder/v/'run.json').read_text()) for v in ['plain','rag']}
        details.append(row); plain.close(); rag.close()
    write_json(root/'summary.json',dict(complete=True,length=96,horizon=96,results=details,
        caveats=['Single seed; custom25/60/80 percent chronological split, not official ETT protocol.',
        'Trailing25-point mean fixed across datasets; its physical duration differs with sampling rate.',
        'Only residual branch is adapted; trend branch uses frozen Moirai. Test data never select a checkpoint.']))
    import csv
    with (root/'query_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    from .report_trend_rag import report
    report(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008'); p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',default=['ETTh1','ETTh2','ETTm1','ETTm2'])
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--train-per-series',type=int,default=256)
    p.add_argument('--window',type=int,default=25)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.train_per_series,args.window)<1: raise ValueError('Positive budgets required')
    torch.set_num_threads(4)
    root=fresh_dir(args.output); source=Path(args.source)
    write_json(root/'protocol.json',dict(length=96,horizon=96,trend='causal trailing moving average with observed left padding',
        window=args.window,datasets=args.datasets,epochs=args.epochs,batch_size=args.batch_size,
        train_per_series=args.train_per_series,device=args.device,candidates=100,top_k=5,
        history_weight=.2,future_weight=.8,trend_branch='frozen Moirai',residual_branch='LoRA Moirai + internal paired retrieval attention',
        code_sha256={f:sha256(f) for f in ['v7/trend_rag.py','v7/benchmark_trend_rag.py','v7/retrieval_finetune.py']}))
    for name in args.datasets:
        folder=fresh_dir(root/name)
        prepare(source,folder,name,args.device,args.window,args.train_per_series,args.batch_size)
        for variant in ['plain','rag']: train_variant(folder,variant,args.device,args.epochs,args.batch_size)
    summarize(root,args.datasets)


if __name__=='__main__': main()
