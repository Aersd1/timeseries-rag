"""Stronger independently trained shuffle: no historical case keeps its residual."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .prepare_unified_residual import read
from .benchmark_unified_residual import SETTINGS,trace,train_one,predict,confirmed,apply_confirmed
from .benchmark_state_recall import metric


def derange(data):
    order=np.empty_like(data['selected_starts'])
    for row,(sid,start) in enumerate(zip(data['sid'],data['start'])):
        rng=np.random.default_rng(20261015+int(sid)*1000003+int(start)+53)
        for j in [0,5]:
            perm=rng.permutation(5)
            while np.any(perm==np.arange(5)):perm=rng.permutation(5)
            order[row,j:j+5]=j+perm
    return dict(data,shuffle_order=order)


def experiment(folder,device='cuda'):
    if (folder/'derangement_run.json').exists():raise ValueError('Control is immutable')
    meta=read(folder/'preparation.json');stats=read(folder/'stats.json');train=derange(trace(folder,'train'));val=derange(trace(folder,'validation'))
    models=[];records=[];vp=[]
    for seed in SETTINGS['seeds']:
        m,losses=train_one(train,stats,meta['period'],'shuffled',seed,device)
        path=folder/f'deranged_{seed}.pt';torch.save(dict(state_dict=m.cpu().state_dict(),variables=len(stats['std']),width=SETTINGS['width']),path);m.to(device)
        models.append(m);vp.append(predict(m,val,stats,meta['period'],'shuffled',device));records.append(dict(seed=seed,file=path.name,sha256=sha256(path),losses=losses,parameters=sum(p.numel() for p in m.parameters())))
    raw_val=np.mean(vp,0).astype(np.float32)
    with np.load(folder/'validation_predictions.npz',allow_pickle=False) as f:q=f['query_confirmed'].copy()
    _,own=confirmed(val,raw_val,stats,val['raw_quantiles'][:,4]);_,extra=confirmed(val,raw_val,stats,q)
    write_json(folder/'derangement_policy.json',dict(own=own,extra=extra))
    np.savez_compressed(folder/'derangement_validation.npz',raw=raw_val,order=val['shuffle_order'])
    write_json(folder/'derangement_contract.json',dict(source_run_sha256=sha256(folder/'run.json'),settings=SETTINGS,models=records,
        policy_sha256=sha256(folder/'derangement_policy.json'),validation_sha256=sha256(folder/'derangement_validation.npz'),
        code_sha256=sha256('v7/benchmark_unified_derangement.py'),
        caveat='Additional full-mismatch control after reviewing ordinary shuffle; TRAIN/VAL-only fit and policy; previously viewed TEST'))
    del train
    test=derange(trace(folder,'test'));raw=[]
    for model in models:raw.append(predict(model,test,stats,meta['period'],'shuffled',device))
    mean=np.mean(raw,0).astype(np.float32)
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:q=f['query_confirmed'].copy()
    outputs=dict(raw=mean,own=apply_confirmed(test,mean,test['raw_quantiles'][:,4],own),extra=apply_confirmed(test,mean,q,extra))
    np.savez_compressed(folder/'derangement_predictions.npz',**outputs,order=test['shuffle_order'],**{f'seed_{seed}':v for seed,v in zip(SETTINGS['seeds'],raw)})
    write_json(folder/'derangement_run.json',dict(complete=True,dataset=folder.name,queries=len(q),metrics={k:metric(v,test) for k,v in outputs.items()},contract_sha256=sha256(folder/'derangement_contract.json'),predictions_sha256=sha256(folder/'derangement_predictions.npz')))
    print('FULL DERANGEMENT',folder.name,{k:metric(v,test) for k,v in outputs.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:experiment(Path(a.root)/name,a.device)
