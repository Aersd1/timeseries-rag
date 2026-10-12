"""Train scratch future-oriented encoders and audit matched chronological retrieval."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from v5.data import Store
from v5.common import valid_ranges
from v6.data import Windows,scale_floor
from v6.model import normalize
from v6.common import save_torch
from .common import fresh_dir,write_json,sha256
from .future_cluster import (FutureClusterEncoder,future_features,fit_clusters,assignments,
    eligible,encoder_loss,rank_from_history)


class Arrays(Dataset):
    def __init__(self,values): self.values=values
    def __len__(self): return len(self.values['x'])
    def __getitem__(self,i): return {k:torch.as_tensor(v[i]) for k,v in self.values.items()}


def collect(store,c,split,stride=None):
    m,h=c['length'],max(c['horizons']); refs=[]
    if stride is None:
        windows=Windows(store.path,c,split)
        refs=windows.refs; windows.store.close()
    else:
        for s in store.series:
            begin,end=(0,s['memory_end']) if split=='memory' else (s['memory_end'],s['train_end'])
            for a,b,step in valid_ranges(s,m,h,begin,end,stride):
                refs.extend((s['sid'],p) for p in range(a,b,step))
    if not refs: raise ValueError('No full windows in '+split)
    return dict(x=np.stack([store.window(s,p,m) for s,p in refs]),
        y=np.stack([store.window(s,p+m,h) for s,p in refs]),
        floor=np.asarray([scale_floor(store.series[s],c) for s,p in refs],np.float32),
        sid=np.asarray([s for s,p in refs],np.int64),start=np.asarray([p for s,p in refs],np.int64))


def normalized(data):
    x=torch.as_tensor(data['x']); floor=torch.as_tensor(data['floor'])
    past,center,scale=normalize(x,floor)
    future=(torch.as_tensor(data['y'])-center[:,None])/scale[:,None]
    return past.numpy(),future.numpy()


def training_candidates(train,bank,c):
    span=c['length']+max(c['horizons']); n=len(train['x'])
    candidates=np.zeros((n,64),np.int64); masks=np.zeros((n,64),bool)
    qpast,_=normalized(train); bpast,_=normalized(bank)
    rng=np.random.default_rng(c['seed'])
    for sid in np.unique(train['sid']):
        rows=np.flatnonzero(train['sid']==sid); ids=np.flatnonzero(bank['sid']==sid)
        distances=(qpast[rows]**2).mean(-1)[:,None]+(bpast[ids]**2).mean(-1)[None]-2*qpast[rows]@bpast[ids].T/c['length']
        allowed=eligible(train['sid'][rows],train['start'][rows],bank['sid'][ids],bank['start'][ids],span)
        for row,ds,ok in zip(rows,distances,allowed):
            choices=ids[ok]
            if len(choices)<2: raise ValueError('Fewer than two nonoverlapping historical candidates')
            hard=ids[np.argsort(np.where(ok,ds,np.inf),kind='stable')[:min(32,len(choices))]]
            others=np.setdiff1d(choices,hard)
            random=rng.choice(others,min(32,len(others)),replace=False)
            selected=np.r_[hard,random]; candidates[row,:len(selected)]=selected; masks[row,:len(selected)]=True
    all_allowed=eligible(train['sid'],train['start'],bank['sid'],bank['start'],span)
    assert all_allowed[np.arange(n)[:,None],candidates][masks].all()
    tf=future_features(*[torch.as_tensor(train[k]) for k in ['x','y','floor']],c['horizons'])
    bf=future_features(*[torch.as_tensor(bank[k]) for k in ['x','y','floor']],c['horizons'])
    distances=[]
    for i in range(0,n,128):
        distances.append(((tf[i:i+128,None]-bf[torch.as_tensor(candidates[i:i+128])])**2).sum(-1).numpy())
    train.update(candidates=candidates,eligible=masks,future_distance=np.concatenate(distances))
    return tf,bf,dict(training_queries=n,training_pairs=int(masks.sum()),self_or_overlap_pairs=0,
        chronology_violations=0,candidate_span=span,hard_history_candidates=32,random_candidates=32)


def encode(model,data,device,batch=512):
    values={'z':[],'probability':[],'forecast':[]}; model.eval()
    with torch.no_grad():
        for a in range(0,len(data['x']),batch):
            out=model(torch.as_tensor(data['x'][a:a+batch],device=device),torch.as_tensor(data['floor'][a:a+batch],device=device))
            for key in values:
                if out[key] is not None: values[key].append(out[key].cpu().numpy())
    return {k:np.concatenate(v) if v else None for k,v in values.items()}


def query_output(encoded,i,x,floor):
    center=float(np.mean(x)); scale=max(float(np.std(x)),float(floor))
    return dict(z=encoded['z'][i],probability=None if encoded['probability'] is None else encoded['probability'][i],
        normalized_history=(x-center)/scale,center=center,scale=scale,
        normalized_forecast=(encoded['forecast'][i]-center)/scale)


def validation(model,val,bank,bank_probability,c,store,device):
    q=encode(model,val,device); b=encode(model,bank,device)
    past,future=normalized(bank); scores=[]
    for i,x in enumerate(val['x']):
        ok=eligible(val['sid'][i:i+1],val['start'][i:i+1],bank['sid'],bank['start'],c['length']+max(c['horizons']))[0]
        out=query_output(q,i,x,val['floor'][i])
        result=rank_from_history(out,past[ok],future[ok],b['z'][ok],bank_probability[ok],
            'history_future_cluster' if model.use_clusters else 'embedding')
        variance=max(store.series[int(val['sid'][i])]['memory_std'],1e-6)**2
        error=(result['prediction'].astype(float)-val['y'][i])**2
        scores.append(np.mean([error[:h].mean()/variance for h in c['horizons']]))
    return float(np.mean(scores))


def train_model(folder,variant,train,val,bank,bf,centers,temperature,c,store,device,epochs):
    out=fresh_dir(folder/variant)
    mc=dict(length=c['length'],horizons=c['horizons'],width=64,dim=32,clusters=len(centers),dropout=.1)
    torch.manual_seed(c['seed']); model=FutureClusterEncoder(mc,variant=='cluster').to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    cp=torch.as_tensor(centers,device=device); probability=assignments(bf.to(device),cp,temperature)
    history=[]; best=float('inf'); best_epoch=0
    run=dict(complete=False,variant=variant,model_config=mc,data_id=store.meta['data_id'],source_config=c,
        epochs=epochs,batch_size=64,learning_rate=.0003,seed=c['seed'],training_windows=len(train['x']),
        validation_windows=len(val['x']),trainable_parameters=sum(p.numel() for p in model.parameters()),
        selection='Validation analog NMSE using coarse historical memory; no test input',
        no_pretrained_encoder=True,feature_policy='history_mean_scaled_ordered_future_v1',
        selection_mode='history_future_cluster' if variant=='cluster' else 'embedding')
    write_json(out/'run.json',run)
    for epoch in range(1,epochs+1):
        began=time.perf_counter()
        encoded=encode(model,bank,device)
        bank_z=torch.as_tensor(encoded['z'],device=device)
        model.train(); total=0.; count=0
        loader=DataLoader(Arrays(train),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(c['seed']+epoch))
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss,_=encoder_loss(model,batch,bank_z,probability,cp,temperature)
            if not torch.isfinite(loss): raise FloatingPointError('Invalid scratch training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            total+=float(loss.detach())*len(batch['x']); count+=len(batch['x'])
        score=validation(model,val,bank,probability.cpu().numpy(),c,store,device)
        if not np.isfinite(score): raise FloatingPointError('Invalid validation NMSE')
        if score<best:
            best,best_epoch=score,epoch
            save_torch(out/'best.pt',dict(family='future_cluster_scratch',model_config=mc,use_clusters=model.use_clusters,
                model=model.state_dict(),centers=torch.as_tensor(centers),cluster_temperature=temperature,
                feature_policy='history_mean_scaled_ordered_future_v1',
                data_id=store.meta['data_id'],source_config=c,epoch=epoch,validation_nmse=score))
        history.append(dict(epoch=epoch,loss=total/count,validation_nmse=score,seconds=time.perf_counter()-began))
        write_json(out/'history.json',history)
        print(folder.name,variant,history[-1],flush=True)
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model']); model.eval()
    run.update(complete=True,best_epoch=best_epoch,best_validation=best,checkpoint_sha256=sha256(out/'best.pt'))
    write_json(out/'run.json',run)
    return model


def evaluate(folder,models,bank,test,c,store,centers,temperature,device):
    past,future=normalized(bank)
    features=future_features(*[torch.as_tensor(bank[k]) for k in ['x','y','floor']],c['horizons'])
    probability=assignments(features,torch.as_tensor(centers),temperature).numpy()
    embeddings={v:encode(model,bank,device)['z'] for v,model in models.items()}
    queries={v:encode(model,test,device) for v,model in models.items()}
    np.savez(folder/'index.npz',past=past,future=future,probability=probability,
        sid=bank['sid'],start=bank['start'],scratch_z=embeddings['scratch'],cluster_z=embeddings['cluster'])
    manifest=dict(complete=True,data_id=store.meta['data_id'],source_config=c,windows=len(past),
        index_sha256=sha256(folder/'index.npz'),checkpoints={v:sha256(folder/v/'best.pt') for v in models},
        clusters_fit_split='train_only',training_features_sha256=sha256(folder/'train_future_features.npy'))
    manifest['feature_policy']='history_mean_scaled_ordered_future_v1'
    write_json(folder/'index_manifest.json',manifest)
    metrics=[]; details=[]; forecasts=[]
    for i,x in enumerate(test['x']):
        sid,start=int(test['sid'][i]),int(test['start'][i]); y=test['y'][i]
        ok=eligible([sid],[start],bank['sid'],bank['start'],c['length']+max(c['horizons']))[0]
        ids=np.flatnonzero(ok); out=query_output(queries['cluster'],i,x,test['floor'][i])
        histories=past[ok]; futures=future[ok]
        raw=np.mean((histories-out['normalized_history'])**2,-1)
        choices={}
        for name,variant,mode in [('scratch_embedding','scratch','embedding'),
                ('cluster_embedding','cluster','embedding'),('future_cluster','cluster','future_cluster'),
                ('history_future_cluster','cluster','history_future_cluster'),
                ('history_future_regression','cluster','history_future_regression')]:
            q=query_output(queries[variant],i,x,test['floor'][i])
            q['horizons']=c['horizons']
            choices[name]=rank_from_history(q,histories,futures,embeddings[variant][ok],probability[ok],mode)
        raw_order=np.argsort(raw,kind='stable')[:5]; ds=raw[raw_order]
        w=np.exp(-(ds-ds.min())/max(float(np.median(ds)),1e-6)); w/=w.sum()
        choices['history_only']=dict(indices=raw_order,weights=w,prediction=w@futures[raw_order]*out['scale']+out['center'])
        row=dict(query=i,sid=sid,start=start,methods={})
        predicted={v:r['prediction'].tolist() for v,r in choices.items()}
        predicted['scratch_direct']=queries['scratch']['forecast'][i].tolist()
        predicted['cluster_direct']=queries['cluster']['forecast'][i].tolist()
        variance=max(store.series[sid]['memory_std'],1e-6)**2
        for method,values in predicted.items():
            error=np.asarray(values,dtype=float)-y.astype(float)
            for h in c['horizons']:
                mse=float(np.mean(error[:h]**2))
                metrics.append(dict(query=i,sid=sid,start=start,horizon=h,method=method,mse=mse,
                    mae=float(np.abs(error[:h]).mean()),nmse=mse/variance))
        # Evaluation-only query futures diagnose retrieval quality, never select candidates.
        yn=(y-out['center'])/out['scale']
        for method,r in choices.items():
            selected=ids[r['indices']]
            assert eligible([sid],[start],bank['sid'][selected],bank['start'][selected],c['length']+max(c['horizons'])).all()
            mismatch=np.mean([np.mean((futures[r['indices'],:h]-yn[:h])**2,axis=-1) for h in c['horizons']],axis=0)
            row['methods'][method]=dict(starts=bank['start'][selected].tolist(),weights=r['weights'].tolist(),
                candidate_future_nmse_mean=float(mismatch.mean()),candidate_future_nmse_top1=float(mismatch[0]),
                candidate_history_nmse_mean=float(raw[r['indices']].mean()))
        details.append(row)
        forecasts.append(dict(query=i,sid=sid,start=start,predictions=predicted))
    for filename,records in [('metrics.jsonl',metrics),('retrieval.jsonl',details)]:
        (folder/filename).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    write_json(folder/'predictions.json',forecasts)
    np.savez(folder/'test.npz',**test)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',default=['ETTh1','ETTh2','ETTm1','ETTm2'])
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--clusters',type=int,default=16)
    args=p.parse_args(); root=fresh_dir(args.output); source=Path(args.source)
    torch.set_num_threads(4)
    write_json(root/'protocol.json',dict(datasets=args.datasets,epochs=args.epochs,clusters=args.clusters,
        source=str(source),device=args.device,train_stride=32,bank_train_stride=32,test_bank_stride=1,
        recall=50,top_k=5,future_weight=.8,history_weight=.2,
        feature_policy='history_mean_scaled_ordered_future_v1',cluster_selection_mode='history_future_cluster',
        code_sha256={f:sha256(f) for f in ['v7/future_cluster.py','v7/benchmark_future_cluster.py']}))
    for name in args.datasets:
        folder=fresh_dir(root/name); src=source/name
        c=json.loads((src/'config.json').read_text()); store=Store(src/'store')
        try:
            train=collect(store,c,'train',32); bank=collect(store,c,'memory',32)
            val=collect(store,c,'validation'); test=collect(store,c,'test')
            tf,bf,audit=training_candidates(train,bank,c)
            centers,temperature=fit_clusters(tf.numpy(),args.clusters,c['seed'])
            np.save(folder/'train_future_features.npy',tf.numpy())
            np.savez(folder/'training_pairs.npz',query_sid=train['sid'],query_start=train['start'],
                candidate_sid=bank['sid'][train['candidates']],candidate_start=bank['start'][train['candidates']],
                eligible=train['eligible'])
            write_json(folder/'training_audit.json',dict(**audit,clusters_fit_split='train_only',
                cluster_fit_windows=len(tf),validation_windows=len(val['x']),test_windows=len(test['x']),
                data_id=store.meta['data_id'],catalog_sha256=sha256(src/'store/catalog.json')))
            models={v:train_model(folder,v,train,val,bank,bf,centers,temperature,c,store,args.device,args.epochs)
                for v in ['scratch','cluster']}
            dense=collect(store,c,'memory',1)
            evaluate(folder,models,dense,test,c,store,centers,temperature,args.device)
        finally:
            store.close()
        del models,train,bank,dense; gc.collect()
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    write_json(root/'complete.json',dict(complete=True,datasets=args.datasets))
    from .report_future_cluster import report
    report(root,source)


if __name__=='__main__': main()
