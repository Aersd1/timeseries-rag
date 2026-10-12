"""Keep original A/Moirai and test 100 candidates, soft diversity and calibration."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from v6.data import Windows
from v5.common import valid_ranges
from .common import fresh_dir,write_json,sha256
from .learned_rerank import LearnedMoiraiReranker,rank_prepared

STRENGTHS=[0.,.1,.25,.5,1.]
ALPHAS=[0.,.25,.5,.75,1.]


def cache_dataset(source,root,name,split,device,batch_size=1,query_stride=None):
    src=source/name; folder=fresh_dir(root/name/split)
    engine=LearnedMoiraiReranker(src/'store',src/'train/best.pt',src/'index_a',device,candidates=100)
    dataset=Windows(src/'store',engine.c,split)
    if query_stride is not None:
        if split!='validation' or query_stride<1: raise ValueError('Dense calibration is validation-only')
        dataset.refs=[]
        for s in dataset.store.series:
            for lo,end,step in valid_ranges(s,engine.c['length'],max(engine.c['horizons']),s['train_end'],s['validation_end'],query_stride):
                dataset.refs.extend((s['sid'],int(p)) for p in range(lo,end,step))
    fields=['x','y','floor','sid','start','starts','distance','future_scores','history_scores','mapped','direct','quantiles']
    arrays={k:[] for k in fields}; timing=[]
    try:
        for begin in range(0,len(dataset),batch_size):
            items=[dataset[i] for i in range(begin,min(begin+batch_size,len(dataset)))]
            pools=engine.prepare_batch(np.stack([t['x'] for t in items]),[int(t['sid']) for t in items],[int(t['start']) for t in items])
            for i,(item,p) in enumerate(zip(items,pools),begin):
                if len(p['candidates'])!=100: raise ValueError('Require full top100 for matched benchmark')
                values=dict(item,starts=[h['start'] for h in p['candidates']],distance=[h['distance'] for h in p['candidates']],
                    future_scores=p['future_scores'],history_scores=p['history_scores'],mapped=p['mapped'],
                    direct=p['direct_prediction'],quantiles=p['quantiles'])
                for k in fields: arrays[k].append(values[k])
                timing.append(dict(query=i,**p['stats']))
            print(name,split,min(begin+batch_size,len(dataset)),'/',len(dataset),flush=True)
        np.savez(folder/'cache.npz',**{k:np.asarray(v) for k,v in arrays.items()})
        write_json(folder/'run.json',dict(complete=True,dataset=name,split=split,config=engine.c,queries=len(dataset),
            data_id=engine.store.meta['data_id'],checkpoint_sha256=sha256(src/'train/best.pt'),
            index_sha256=sha256(src/'index_a/manifest.json'),cache_sha256=sha256(folder/'cache.npz'),
            candidates=100,top_k=5,history_weight=.2,future_weight=.8,inference_batch_size=batch_size,query_stride=query_stride,
            quantile_levels=p['quantile_levels'].tolist()))
        write_json(folder/'timing.json',timing)
    finally: engine.close(); dataset.store.close()


def load_cache(folder):
    run=json.loads((folder/'run.json').read_text())
    if not run['complete'] or sha256(folder/'cache.npz')!=run['cache_sha256']: raise ValueError('Cache changed')
    with np.load(folder/'cache.npz',allow_pickle=False) as data: return run,{k:data[k].copy() for k in data.files}


def rank_cache(data,run,i,count,strength=0.):
    sid=int(data['sid'][i]); starts=data['starts'][i,:count]
    p=dict(candidates=[dict(sid=sid,start=int(s),distance=float(d),future_score=float(f),history_nmse=float(h))
        for s,d,f,h in zip(starts,data['distance'][i,:count],data['future_scores'][i,:count],data['history_scores'][i,:count])],
        mapped=data['mapped'][i,:count],future_scores=data['future_scores'][i,:count],
        history_scores=data['history_scores'][i,:count])
    c=run['config']; return rank_prepared(p,.2,5,c['length']+max(c['horizons']),strength)


def nmse(prediction,data,i,c,std):
    error=(np.asarray(prediction,dtype=float)-data['y'][i].astype(float))**2
    return float(np.mean([error[:h].mean()/max(float(std),1e-6)**2 for h in c['horizons']]))


def select_policy(tables):
    """Validation-only shared diversity strength, then per-dataset mixture alpha."""
    if not tables or any(t['split']!='validation' for t in tables.values()):
        raise ValueError('Policy selection accepts validation only')
    objectives={str(s):float(np.mean([t['diversity'][str(s)]/max(t['plain100'],1e-12) for t in tables.values()])) for s in STRENGTHS}
    strength=min(STRENGTHS,key=lambda s:(objectives[str(s)],s))
    alphas={name:min(ALPHAS,key=lambda a:(t['mixture'][str(strength)][str(a)],a)) for name,t in tables.items()}
    return dict(diversity_weight=strength,analogue_alpha=alphas,history_weight=.2,future_weight=.8,
        selected_on='validation',diversity_grid=STRENGTHS,alpha_grid=ALPHAS,diversity_objectives=objectives)


def validation_tables(root,source,datasets):
    tables={}
    for name in datasets:
        run,data=load_cache(root/name/'validation'); c=run['config']
        catalog=json.loads((source/name/'store/catalog.json').read_text())
        base=[]; diverse={str(s):[] for s in STRENGTHS}; mix={str(s):{str(a):[] for a in ALPHAS} for s in STRENGTHS}
        for i in range(len(data['x'])):
            std=catalog['series'][int(data['sid'][i])]['memory_std']
            base.append(nmse(rank_cache(data,run,i,100)['prediction'],data,i,c,std))
            for s in STRENGTHS:
                pred=rank_cache(data,run,i,100,s)['prediction']
                diverse[str(s)].append(nmse(pred,data,i,c,std))
                for a in ALPHAS:
                    mix[str(s)][str(a)].append(nmse((1-a)*data['direct'][i]+a*pred,data,i,c,std))
        tables[name]=dict(split='validation',plain100=float(np.mean(base)),
            diversity={s:float(np.mean(v)) for s,v in diverse.items()},
            mixture={s:{a:float(np.mean(v)) for a,v in av.items()} for s,av in mix.items()})
    return tables


def evaluate_cache(folder,policy,catalog):
    run,data=load_cache(folder); c=run['config']; metrics=[]; details=[]; forecasts=[]
    strength=policy['diversity_weight']; alpha=policy['analogue_alpha'][run['dataset']]
    from .quality_gate import gate_predict
    state=torch.load(folder.parent/'quality_gate.pt',map_location='cpu',weights_only=True)
    for i in range(len(data['x'])):
        sid,start=int(data['sid'][i]),int(data['start'][i])
        selected={m:rank_cache(data,run,i,k,s) for m,k,s in [('a50',50,0.),('a100',100,0.),('a100_diverse',100,strength)]}
        predictions={m:r['prediction'] for m,r in selected.items()}
        predictions['moirai_direct']=data['direct'][i]
        predictions['a100_calibrated']=(1-alpha)*data['direct'][i]+alpha*predictions['a100_diverse']
        from .quality_gate import quality_features
        lookup={int(s):j for j,s in enumerate(data['starts'][i])}; hits=selected['a100_diverse']['hits']
        ids=[lookup[h['start']] for h in hits]
        feature=quality_features(data['x'][i],data['floor'][i],data['quantiles'][i],run['quantile_levels'],
            data['mapped'][i,ids],hits,c['length']+max(c['horizons']),catalog['series'][sid]['memory_std'])
        raw_alpha,deployed_alpha=gate_predict(state,feature[None])
        predictions['a100_gate']=(1-float(raw_alpha[0]))*data['direct'][i]+float(raw_alpha[0])*predictions['a100_diverse']
        predictions['a100_adaptive']=(1-float(deployed_alpha[0]))*data['direct'][i]+float(deployed_alpha[0])*predictions['a100_diverse']
        variance=max(catalog['series'][sid]['memory_std'],1e-6)**2
        for method,pred in predictions.items():
            error=np.asarray(pred,dtype=float)-data['y'][i].astype(float)
            for h in c['horizons']:
                mse=float(np.mean(error[:h]**2))
                metrics.append(dict(query=i,sid=sid,start=start,horizon=h,method=method,mse=mse,
                    mae=float(np.abs(error[:h]).mean()),nmse=mse/variance))
        details.append(dict(query=i,sid=sid,start=start,selected={m:r['hits'] for m,r in selected.items()}))
        forecasts.append(dict(query=i,sid=sid,start=start,predictions={m:v.tolist() for m,v in predictions.items()},
            raw_gate_alpha=float(raw_alpha[0]),deployed_alpha=float(deployed_alpha[0])))
    write_json(folder/'predictions.json',forecasts)
    write_json(folder/'retrieval.json',details)
    (folder/'metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in metrics),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--datasets',nargs='+',default=['ETTh1','ETTh2','ETTm1','ETTm2'])
    p.add_argument('--calibration-stride',type=int)
    p.add_argument('--reuse-root',help='Reuse immutable train/test forecast caches, not model-selection outcomes')
    args=p.parse_args(); root=fresh_dir(args.output); source=Path(args.source); torch.set_num_threads(4)
    write_json(root/'protocol.json',dict(datasets=args.datasets,candidates=100,top_k=5,history_weight=.2,future_weight=.8,
        source=str(source),device=args.device,retrain_base=False,gate_epochs=200,gate_parameters=113,
        calibration_stride=args.calibration_stride,reused_caches=args.reuse_root,
        diversity_grid=STRENGTHS,alpha_grid=ALPHAS,
        code_sha256={p:sha256(p) for p in ['v7/learned_rerank.py','v7/benchmark_a100.py','v7/quality_gate.py']}))
    if args.reuse_root:
        old=Path(args.reuse_root)
        for name in args.datasets:
            for split in ['train','test']:
                run,_=load_cache(old/name/split)
                if run['checkpoint_sha256']!=sha256(source/name/'train/best.pt') or run['index_sha256']!=sha256(source/name/'index_a/manifest.json'):
                    raise ValueError('Cannot reuse mismatched A/forecast caches')
                shutil.copytree(old/name/split,root/name/split)
    for name in args.datasets:
        cache_dataset(source,root,name,'validation',args.device,batch_size=32 if args.calibration_stride else 1,
            query_stride=args.calibration_stride)
    tables=validation_tables(root,source,args.datasets); write_json(root/'validation_selection.json',tables)
    policy=select_policy(tables)
    policy['validation_hashes']={n:sha256(root/n/'validation/cache.npz') for n in args.datasets}
    write_json(root/'policy.json',policy); policy_sha=sha256(root/'policy.json')
    print('FROZEN POLICY',policy,flush=True)
    from .quality_gate import train_gate
    for name in args.datasets:
        if not args.reuse_root: cache_dataset(source,root,name,'train',args.device,batch_size=32)
        train_gate(root/name,source/name,policy)
    for name in args.datasets:
        if not args.reuse_root: cache_dataset(source,root,name,'test',args.device)
        assert sha256(root/'policy.json')==policy_sha
        catalog=json.loads((source/name/'store/catalog.json').read_text())
        for split in ['validation','test']: evaluate_cache(root/name/split,policy,catalog)
    write_json(root/'complete.json',dict(complete=True,datasets=args.datasets,policy_sha256=policy_sha))
    from .report_a100 import report
    report(root,source)


if __name__=='__main__': main()
