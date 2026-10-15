"""Refit the SAME five forecast experts at three historical cutoffs and TRAIN."""
import argparse,gc
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .dense_forward import read,make_index,materialize
from .unified_residual import UnifiedResidualFusion
from .benchmark_unified_residual import train_one,predict,SETTINGS

ARMS=['query_only','history_only','real','shuffled','matched_random']
EXPERT_PROTOCOL=dict(seed=13,epochs=16,architecture='UnifiedResidualFusion width32',arms=ARMS,
    training='At most64 uniformly spread origins per variable and prefix',
    selection='No early stopping or VAL selection; all five experts are refitted from scratch before each held block')


def model_from(saved,device):
    model=UnifiedResidualFusion(saved['variables'],32);model.load_state_dict(saved['state_dict']);return model.to(device).eval()


def fit_phase(source,out,phase,cutoff,held_starts,device='cuda'):
    path=out/(phase+'_experts.json')
    if path.exists():
        saved=read(path)
        for file,key in [(phase+'_predictions.npz','predictions_sha256')]:
            if sha256(out/file)!=saved[key]:raise ValueError('Changed prefix prediction')
        print('FORWARD EXPERT REUSE',out.name,phase,flush=True);return
    meta=read(out/'preparation.json');store=Store(meta['store']);count=len(store.series)
    try:
        train_starts=meta['layout']['expert_starts'][str(cutoff)]
        make_index(source,out,cutoff,{sid:train_starts for sid in range(count)},phase+'_train')
        make_index(source,out,cutoff,{sid:held_starts for sid in range(count)},phase+'_held')
    finally:store.close()
    train=materialize(source,out,phase+'_train');stats=read(out/f'prefix_{cutoff}.json')
    if np.any(train['start']+192>cutoff):raise ValueError('Expert fit label beyond prefix')
    train_hash=sha256(out/(phase+'_train_index.npz'));stats_hash=sha256(out/f'prefix_{cutoff}.json');models={};records=[]
    for mode in ARMS:
        target=out/f'{phase}_{mode}.pt'
        if target.exists():
            saved=torch.load(target,map_location='cpu',weights_only=True)
            if saved['train_index_sha256']!=train_hash or saved['stats_sha256']!=stats_hash:raise ValueError('Changed prefix fit inputs')
            model=model_from(saved,device);losses=saved['losses']
        else:
            model,losses=train_one(train,stats,meta['period'],mode,13,device)
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},variables=len(stats['std']),losses=losses,
                train_index_sha256=train_hash,stats_sha256=stats_hash,seed=13),target)
        records.append(dict(mode=mode,file=target.name,sha256=sha256(target),losses=losses))
        models[mode]=model;print('FORWARD EXPERT FIT',out.name,phase,mode,flush=True)
    del train;gc.collect()
    held=materialize(source,out,phase+'_held')
    if np.any(held['start']<cutoff):raise ValueError('Held history precedes fit cutoff')
    predictions={mode:predict(model,held,stats,meta['period'],mode,device) for mode,model in models.items()}
    np.savez_compressed(out/(phase+'_predictions.npz'),**predictions,sid=held['sid'],start=held['start'])
    write_json(path,dict(complete=True,cutoff=cutoff,protocol=EXPERT_PROTOCOL,models=records,train_queries=read(out/(phase+'_train_index.json'))['queries'],
        last_train_label_end=int(np.max(np.asarray(train_starts))+192),first_held_start=int(held['start'].min()),held_queries=len(held['sid']),
        train_index_sha256=train_hash,held_index_sha256=sha256(out/(phase+'_held_index.npz')),stats_sha256=stats_hash,
        foundation_manifest_sha256=sha256(out/'foundation_manifest.json'),predictions_sha256=sha256(out/(phase+'_predictions.npz'))))
    del models,held,predictions;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()


def prepare(source,out,device='cuda'):
    out=Path(out);source=Path(source);meta=read(out/'preparation.json');manifest=read(out/'foundation_manifest.json');store=Store(meta['store'])
    try:
        if len(manifest['files'])!=len(store.series):raise ValueError('Finish foundation cache before experts')
    finally:store.close()
    if SETTINGS['epochs']!=16:raise ValueError('Changed inherited expert budget')
    protocol=out/'expert_protocol.json'
    if protocol.exists():
        if read(protocol)!=EXPERT_PROTOCOL:raise ValueError('Changed expert protocol')
    else:write_json(protocol,EXPERT_PROTOCOL)
    cuts=meta['layout']['cuts']
    for i,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        held=[p for p in meta['layout']['oof_starts'] if a<=p and p+192<=b]
        fit_phase(source,out,f'fold{i}',a,held,device)
    fit_phase(source,out,'final',cuts[-1],meta['layout']['validation_starts'],device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:prepare(Path(a.source)/name,Path(a.output)/name,a.device)
