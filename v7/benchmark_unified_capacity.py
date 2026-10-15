"""Additional query/query ensemble control for the two-network deployment gate."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .prepare_unified_residual import read
from .benchmark_unified_residual import SETTINGS,trace,train_one,predict,confirmed,apply_confirmed
from .benchmark_state_recall import metric,interval

SEEDS=[s+100 for s in SETTINGS['seeds']]


def experiment(folder,device='cuda'):
    if (folder/'capacity_run.json').exists():raise ValueError('Capacity control is immutable')
    run=read(folder/'run.json');meta=read(folder/'preparation.json');stats=read(folder/'stats.json')
    train=trace(folder,'train');val=trace(folder,'validation');models=[];records=[];vp=[]
    for seed in SEEDS:
        model,losses=train_one(train,stats,meta['period'],'query_only',seed,device)
        path=folder/f'query_replica_{seed}.pt';torch.save(dict(state_dict=model.cpu().state_dict(),variables=len(stats['std']),width=SETTINGS['width']),path);model.to(device)
        models.append(model);vp.append(predict(model,val,stats,meta['period'],'query_only',device))
        records.append(dict(file=path.name,sha256=sha256(path),seed=seed,losses=losses,parameters=sum(p.numel() for p in model.parameters())))
    raw_val=np.mean(vp,0).astype(np.float32)
    with np.load(folder/'validation_predictions.npz',allow_pickle=False) as f:q=f['query_confirmed'].copy()
    _,policy=confirmed(val,raw_val,stats,q)
    write_json(folder/'capacity_policy.json',policy)
    write_json(folder/'capacity_contract.json',dict(source_contract_sha256=run['contract_sha256'],
        source_run_sha256=sha256(folder/'run.json'),policy_sha256=sha256(folder/'capacity_policy.json'),models=records,
        settings=SETTINGS,seeds=SEEDS,code_sha256=sha256('v7/benchmark_unified_capacity.py'),
        rule='Same base query ensemble plus three additional query-only models; same beta grid and VAL guards as real/history/shuffled/random',
        caveat='Additional capacity control specified after inspecting initial results; fixed seed offset100, no TEST fitting or selection'))
    np.savez_compressed(folder/'capacity_validation.npz',raw=raw_val)
    del train
    test=trace(folder,'test');raw=[]
    for model in models:raw.append(predict(model,test,stats,meta['period'],'query_only',device))
    raw_replica=np.mean(raw,0).astype(np.float32)
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:q=f['query_confirmed'].copy();real=f['extra_real'].copy();single=f['raw_query_only'].copy()
    final=apply_confirmed(test,raw_replica,q,policy)
    np.savez_compressed(folder/'capacity_predictions.npz',raw_replica=raw_replica,query_query=final,
        six_query_equal_mean=.5*(single+raw_replica),**{f'seed_{seed}':v for seed,v in zip(SEEDS,raw)})
    record=dict(complete=True,dataset=folder.name,queries=len(q),metrics=metric(final,test),raw_replica_metrics=metric(raw_replica,test),
        real_vs_query_query_nmse_interval=interval(final,real,test),
        contract_sha256=sha256(folder/'capacity_contract.json'),validation_sha256=sha256(folder/'capacity_validation.npz'),prediction_sha256=sha256(folder/'capacity_predictions.npz'))
    write_json(folder/'capacity_run.json',record);print('UNIFIED CAPACITY',folder.name,record['metrics'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:experiment(Path(a.root)/name,a.device)
